"""MetricWriter: TB SummaryWriter + 可选 tensorboard Web 服务子进程 + 可选 JSONL 文本日志。"""

from __future__ import annotations

import atexit
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Mapping

from torch.utils.tensorboard import SummaryWriter


class MetricWriter:
    """向 TensorBoard 写入标量指标，并可选在本地端口起 Web UI / 落 JSONL 文本日志。

    Args:
        logdir: TB 服务的扫描根目录（``tensorboard --logdir``）。
        name: run 子目录名（TB UI 里显示的 run 标识）。非空时 events 写到
            ``{logdir}/{name}/``；为 ``None`` 时写 logdir 根（TB 里 run 名显示为 ``.``）。
        autostart_server: 是否自动拉起 ``tensorboard`` 子进程。
        port: Web 服务端口（默认 6006）。
        host: Web 服务监听地址（默认 ``0.0.0.0``，局域网可访问）。
        flush_every_s: SummaryWriter 异步 flush 间隔（秒）。
        overwrite: 启动时清理当前 run events 目录下旧的 events 文件（只看本次 run）。
            不影响 ``logdir`` 下其他同级 run 子目录。
        log: 非空时在 events 目录下开启 JSONL 文本日志 ``{log}.log``（追加写）。
            每行一个事件: ``{"ts", "scope", "step", "metrics": {...}}``。
    """

    def __init__(
        self,
        logdir: str | Path,
        *,
        name: str | None = None,
        autostart_server: bool = True,
        port: int = 6006,
        host: str = "0.0.0.0",
        flush_every_s: int = 5,
        overwrite: bool = False,
        log: str | None = None,
    ) -> None:
        self._logdir = Path(logdir)
        self._events_dir = self._logdir / name if name else self._logdir
        self._events_dir.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._purge_events()

        self._writer: SummaryWriter | None = SummaryWriter(
            log_dir=str(self._events_dir), flush_secs=int(flush_every_s)
        )
        self._lock = threading.Lock()
        self._log_file: IO[str] | None = None
        if log:
            log_path = self._events_dir / f"{log}.log"
            self._log_file = open(log_path, "a", encoding="utf-8", buffering=1)

        self._proc: subprocess.Popen | None = None
        self._port = int(port)
        self._host = str(host)
        if autostart_server:
            self._start_server()
        atexit.register(self.close)

    def _purge_events(self) -> None:
        """只清 ``_events_dir`` 下文件名以 ``events.out.tfevents`` 开头的文件；
        绝不删除目录、不递归子目录、不触碰任何非 events 文件（比如 XLA 编译缓存、
        checkpoint、同目录的 JSONL 日志都不会动）。"""
        removed = 0
        for f in self._events_dir.iterdir():
            if f.is_file() and f.name.startswith("events.out.tfevents"):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            sys.stdout.write(
                f"[rltrack] overwrite=True: 清理 {removed} 个旧 events 文件 "
                f"({self._events_dir})\n"
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        step: int,
        scope: str | None = None,
    ) -> None:
        """批量写入标量。``scope`` 作为 tag 前缀，TB UI 会按前缀分组。"""
        if self._writer is None:
            return
        prefix = f"{scope}/" if scope else ""
        step_i = int(step)

        coerced = {name: float(value) for name, value in metrics.items()}

        with self._lock:
            for name, value in coerced.items():
                self._writer.add_scalar(f"{prefix}{name}", value, step_i)

            if self._log_file is not None:
                record = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "scope": scope or "",
                    "step": step_i,
                    "metrics": coerced,
                }
                self._log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.flush()
            if self._log_file is not None:
                self._log_file.flush()

    # ------------------------------------------------------------------
    # Web 服务
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable, "-m", "tensorboard.main",
                    "--logdir", str(self._logdir),
                    "--port", str(self._port),
                    "--host", self._host,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError) as e:
            sys.stderr.write(
                f"[rltrack] 警告: 启动 tensorboard 失败 ({e!r})，"
                "请确认 `pip install tensorboard`\n"
            )
            sys.stderr.flush()
            self._proc = None
            return

        # 展示用地址：浏览器不能把 0.0.0.0 当目标，统一显示 localhost。
        display_host = "localhost" if self._host in ("0.0.0.0", "") else self._host

        sys.stdout.write(
            f"[rltrack] TensorBoard 启动中: http://{display_host}:{self._port}"
            f" (logdir={self._logdir})\n"
        )
        sys.stdout.flush()

        # 后台线程轮询端口，就绪或超时后打印消息；init 不阻塞，写入不受影响。
        # TB 冷启动在 Linux 上几秒、Windows 上可能到 ~20s，给足 60s 余量。
        threading.Thread(
            target=self._wait_for_ready,
            args=(60.0, display_host),
            daemon=True,
        ).start()

    def _wait_for_ready(self, timeout_s: float, display_host: str) -> None:
        probe_host = "127.0.0.1" if self._host in ("0.0.0.0", "") else self._host
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((probe_host, self._port), timeout=0.3):
                    sys.stdout.write(
                        f"[rltrack] TensorBoard 已就绪: http://{display_host}:{self._port}\n"
                    )
                    sys.stdout.flush()
                    return
            except OSError:
                time.sleep(0.5)
        sys.stderr.write(
            f"[rltrack] 警告: TensorBoard 在 {int(timeout_s)}s 内仍未就绪"
            f"（port={self._port}）\n"
        )
        sys.stderr.flush()

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.flush()
                    self._writer.close()
                finally:
                    self._writer = None
            if self._log_file is not None:
                try:
                    self._log_file.close()
                finally:
                    self._log_file = None
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass


# ----------------------------------------------------------------------
# 全局单例：一个训练进程一个 writer，多组件跨线程共享
# ----------------------------------------------------------------------

_singleton: MetricWriter | None = None


def init_writer(
    logdir: str | Path,
    *,
    name: str | None = None,
    autostart_server: bool = True,
    port: int = 6006,
    host: str = "0.0.0.0",
    flush_every_s: int = 10,
    overwrite: bool = False,
    log: str | None = None,
) -> MetricWriter:
    """构造全局 writer。参数与 ``MetricWriter.__init__`` 一致。重复调用会报错。"""
    global _singleton
    if _singleton is not None:
        raise RuntimeError("rltrack.init_writer: writer already initialized")
    _singleton = MetricWriter(
        logdir=logdir,
        name=name,
        autostart_server=autostart_server,
        port=port,
        host=host,
        flush_every_s=flush_every_s,
        overwrite=overwrite,
        log=log,
    )
    return _singleton


def get_writer() -> MetricWriter | None:
    return _singleton


def close_writer() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.close()
        _singleton = None
