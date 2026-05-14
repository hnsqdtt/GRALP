from __future__ import annotations

"""Build dwa.c into a shared library next to this script.

Outputs:
    Windows:  eval/dwa/dwa.dll
    Linux:    eval/dwa/libdwa.so
    macOS:    eval/dwa/libdwa.dylib

Picks a compiler in this order (first one on PATH wins):
    Windows: cl.exe (MSVC) -> gcc (MinGW) -> clang
    Linux/macOS: cc -> gcc -> clang

Usage:
    python -m eval.dwa.setup_dwa
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC = HERE / "dwa.c"


def _windows_out_path() -> Path:
    return HERE / "dwa.dll"


def _unix_out_path() -> Path:
    if sys.platform == "darwin":
        return HERE / "libdwa.dylib"
    return HERE / "libdwa.so"


def _msvc_env_and_cl() -> tuple[str, dict]:
    """Use setuptools' bundled distutils helpers to (a) locate cl.exe and (b)
    obtain the INCLUDE/LIB/PATH env vars vcvarsall would set. Returns
    (cl_path, env_dict)."""
    try:
        from setuptools._distutils.compilers.C.msvc import _get_vc_env  # type: ignore
    except ImportError:
        try:
            from setuptools._distutils._msvccompiler import _get_vc_env  # type: ignore
        except ImportError:
            from distutils._msvccompiler import _get_vc_env  # type: ignore

    # plat_spec strings vary across versions; 'x64' works on modern setuptools.
    try:
        vc_env = _get_vc_env("x64")
    except Exception:
        vc_env = _get_vc_env("x86_amd64")

    # Merge over current env so PATH gains the MSVC bin directory.
    env = os.environ.copy()
    for k, v in vc_env.items():
        env[k.upper()] = v

    # Find cl.exe on the augmented PATH.
    cl_path = shutil.which("cl", path=env.get("PATH", ""))
    if not cl_path:
        # vc_env keys are typically lowercase; ensure PATH is in there.
        cl_path = shutil.which("cl", path=env.get("Path", "") or env.get("path", ""))
    if not cl_path:
        raise RuntimeError("Could not locate cl.exe after distutils MSVC setup")
    return cl_path, env


def _build_via_distutils(out: Path) -> Path:
    """Drive the build directly with `cl /LD` using the MSVC env distutils
    discovered for us. We bypass distutils' own link step because it drops the
    /DLL flag and fails LNK1561 on this toolchain."""
    cl_path, env = _msvc_env_and_cl()
    cmd = [
        cl_path, "/nologo", "/O2", "/LD",
        str(SRC),
        f"/Fe:{out}",
        f"/Fo:{HERE / 'dwa.obj'}",
    ]
    print(f"[setup_dwa] cl /LD via distutils-discovered MSVC at {cl_path}")
    subprocess.check_call(cmd, env=env)
    for stale in (HERE / "dwa.obj", HERE / "dwa.exp", HERE / "dwa.lib"):
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
    return out


def _build_windows() -> Path:
    out = _windows_out_path()
    cl = shutil.which("cl")
    if cl:
        # MSVC: /LD = create DLL, /O2 = optimize, /Fe = output exe/dll name, /Fo = obj path.
        # /nologo silences the banner. Place obj in the same dir to keep things tidy.
        obj = HERE / "dwa.obj"
        cmd = [
            cl, "/nologo", "/O2", "/LD",
            str(SRC),
            f"/Fe:{out}",
            f"/Fo:{obj}",
        ]
        print(f"[setup_dwa] cl: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        for stale in (obj, HERE / "dwa.exp", HERE / "dwa.lib"):
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass
        return out

    gcc = shutil.which("gcc") or shutil.which("clang")
    if gcc:
        cmd = [gcc, "-O2", "-shared", "-o", str(out), str(SRC), "-lm"]
        print(f"[setup_dwa] {gcc}: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return out

    # Last resort: let setuptools-bundled distutils find MSVC via the registry.
    print("[setup_dwa] No compiler on PATH; falling back to setuptools' MSVC detection")
    return _build_via_distutils(out)


def _build_unix() -> Path:
    out = _unix_out_path()
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("No C compiler on PATH (need cc, gcc, or clang).")
    flags = ["-O2", "-fPIC", "-shared", "-o", str(out), str(SRC), "-lm"]
    cmd = [cc, *flags]
    print(f"[setup_dwa] {cc}: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    return out


def build() -> Path:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source: {SRC}")
    if sys.platform == "win32":
        out = _build_windows()
    else:
        out = _build_unix()
    if not out.exists():
        raise RuntimeError(f"Compiler reported success but {out} is missing")
    print(f"[setup_dwa] Built {out}  ({out.stat().st_size:,} bytes)")
    return out


def main() -> int:
    try:
        build()
    except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError) as e:
        print(f"[setup_dwa] FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
