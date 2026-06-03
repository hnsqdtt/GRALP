from __future__ import annotations

"""Time the serial CPU DWA across candidate-grid densities (timing only).

The port's numerical correctness was already verified against
eval/dwa/planner.py -- C(double) reproduced torch(float64) with 0% action
mismatch, and the residual vs float32 equals float32's own near-tie rounding.
This script keeps only the timing path: it runs dwa_cpu.exe in a SINGLE process
over a grid ladder on the recorded snapshots fixture (consistent CPU state ->
per-plan cost is monotonic/comparable) and prints a latency table relative to
the 9x17 baseline. No torch / numpy needed.

Usage:
    python -m eval.dwa_cpu.bench_dwa
    python -m eval.dwa_cpu.bench_dwa --grids 3x5 9x17 21x41 --repeat 8 --warmup 2
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXE = HERE / "dwa_cpu.exe"
BUILD = HERE / "build.bat"
DEFAULT_SNAP = HERE / "snapshots.bin"
DEFAULT_DWA_CFG = REPO / "eval" / "dwa_config.json"
DEFAULT_GRIDS = ["3x5", "5x9", "7x13", "9x17", "11x21", "21x41"]
BASELINE = "9x17"


def ensure_built(force: bool = False) -> None:
    if EXE.is_file() and not force:
        return
    print("[bench_dwa] building dwa_cpu.exe ...", file=sys.stderr)
    proc = subprocess.run(["cmd", "/c", str(BUILD)], capture_output=True, text=True)
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0 or not EXE.is_file():
        raise RuntimeError("build failed; run eval\\dwa_cpu\\build.bat manually")


def _parse_result(line: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for kv in line.split()[1:]:
        k, _, v = kv.partition("=")
        d[k] = v
    return d


def run_sweep(snap: Path, cfg: Path, grids: List[str],
              repeat: int, warmup: int) -> List[Dict[str, Any]]:
    cmd = [str(EXE), str(snap), str(cfg), "--sweep", ",".join(grids),
           "--repeat", str(repeat), "--warmup", str(warmup), "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"dwa_cpu.exe --sweep failed ({proc.returncode}):\n"
                           f"{proc.stdout}\n{proc.stderr}")
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            d = _parse_result(line)
            rows.append({
                "grid": d.get("grid"),
                "nc": int(d.get("NC", 0)),
                "us": float(d.get("per_plan_us", 0.0)),
            })
    return rows


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serial CPU DWA timing across grid densities")
    p.add_argument("--snapshots", type=Path, default=DEFAULT_SNAP)
    p.add_argument("--dwa-config", type=Path, default=DEFAULT_DWA_CFG)
    p.add_argument("--grids", nargs="*", default=None, help="e.g. 3x5 9x17 21x41")
    p.add_argument("--repeat", type=int, default=8, help="timed passes (min is reported)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    if not args.snapshots.is_file():
        print(f"snapshots fixture not found: {args.snapshots}", file=sys.stderr)
        return 2
    ensure_built(args.rebuild)

    grids = [g.replace(",", "x") for g in args.grids] if args.grids else list(DEFAULT_GRIDS)
    rows = run_sweep(args.snapshots, args.dwa_config, grids, args.repeat, args.warmup)
    base = next((r for r in rows if r["grid"] == BASELINE), None)

    bar = "=" * 64
    print(bar)
    print(f"Serial CPU DWA timing  (single process, min of {args.repeat}, warmup {args.warmup})")
    print(f"snapshots = {args.snapshots.name}")
    print(bar)
    print(f"{'grid':>8}  {'NC':>5}  {'us/plan':>10}  {'plans/s':>12}  {'x baseline':>11}")
    print("-" * 64)
    for r in rows:
        pps = 1e6 / r["us"] if r["us"] > 0 else float("inf")
        rel = (r["us"] / base["us"]) if (base and base["us"] > 0) else float("nan")
        mark = "  <- baseline" if r["grid"] == BASELINE else ""
        print(f"{r['grid']:>8}  {r['nc']:>5}  {r['us']:>10.2f}  {pps:>12,.0f}  {rel:>10.2f}x{mark}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
