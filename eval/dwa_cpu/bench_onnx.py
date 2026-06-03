from __future__ import annotations

"""Measure single-step CPU inference latency of an exported policy.onnx.

Companion to the serial DWA timing: lets you compare the learned policy's
per-step cost against the DWA candidate-scan cost on the same CPU.

The ONNX (cnn_zeropad policy) takes ``obs[batch,112]`` (105 rays + 7 pose) and
``limits[batch,2]`` ([vx_max, omega_max]); it returns ``action/mu/log_std``.
This times one forward pass at batch=1 with warmup + min/median/percentile
stats, under both 1 intra-op thread (apples-to-apples with the serial DWA) and
ONNX Runtime's default thread pool.

Usage:
    python -m eval.dwa_cpu.bench_onnx
    python -m eval.dwa_cpu.bench_onnx --model "C:\\path\\to\\policy.onnx" \
        --batch 1 --iters 3000 --warmup 300 --output mu
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort


DEFAULT_MODEL = Path(r"C:\Users\MECHREVO\Desktop\毕业设计模型\20260527-100603-cnn_zeropad\model\policy.onnx")


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def make_session(model: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])


def bench(model: Path, batch: int, iters: int, warmup: int, threads: int,
          output: str, vx_max: float, omega_max: float) -> dict:
    sess = make_session(model, threads)
    in_names = [i.name for i in sess.get_inputs()]
    out_all = [o.name for o in sess.get_outputs()]
    out_names = out_all if output == "all" else [output]

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((batch, 112)).astype(np.float32)
    limits = np.tile(np.array([[vx_max, omega_max]], np.float32), (batch, 1))
    feeds = {}
    if "obs" in in_names:
        feeds["obs"] = obs
    if "limits" in in_names:
        feeds["limits"] = limits
    # fallback: positional if names differ
    for nm in in_names:
        if nm not in feeds:
            shp = [batch if (isinstance(d, str) or d is None) else d
                   for d in sess.get_inputs()[in_names.index(nm)].shape]
            feeds[nm] = rng.standard_normal(shp).astype(np.float32)

    for _ in range(warmup):
        sess.run(out_names, feeds)

    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(out_names, feeds)
        samples.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "threads": threads if threads > 0 else "default",
        "batch": batch,
        "output": ",".join(out_names),
        "iters": iters,
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "p90_us": _percentile(samples, 0.90),
        "p99_us": _percentile(samples, 0.99),
        "max_us": max(samples),
    }


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-step ONNX policy latency benchmark")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--output", type=str, default="mu",
                   help="which output(s) to fetch: action / mu / log_std / all")
    p.add_argument("--threads", type=int, nargs="*", default=[1, 0],
                   help="intra-op thread counts to test (0 = ORT default)")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    if not args.model.is_file():
        print(f"model not found: {args.model}")
        return 2

    vx_max, omega_max = 0.6, 1.5
    meta = args.model.parent / "meta.json"
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        lim = m.get("limits", {})
        vx_max = float(lim.get("vx_max", vx_max))
        omega_max = float(lim.get("omega_max", omega_max))

    print(f"Model:   {args.model}")
    print(f"ORT:     {ort.__version__}  providers={ort.get_available_providers()}")
    print(f"Inputs:  obs[*,112], limits[*,2]=[{vx_max},{omega_max}]   "
          f"batch={args.batch}  output={args.output}")
    print(f"Bench:   {args.iters} iters, {args.warmup} warmup")
    print()
    print(f"{'threads':>8}  {'min':>9}  {'median':>9}  {'mean':>9}  {'p90':>9}  {'p99':>9}  {'1/mean':>10}")
    print("-" * 74)
    rows = []
    for th in args.threads:
        r = bench(args.model, args.batch, args.iters, args.warmup, th,
                  args.output, vx_max, omega_max)
        rows.append(r)
        hz = 1e6 / r["mean_us"] if r["mean_us"] > 0 else float("inf")
        print(f"{str(r['threads']):>8}  {r['min_us']:>8.1f}u  {r['median_us']:>8.1f}u  "
              f"{r['mean_us']:>8.1f}u  {r['p90_us']:>8.1f}u  {r['p99_us']:>8.1f}u  {hz:>9,.0f}/s")
    print()
    print("(u = microseconds per single forward pass at the given batch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
