"""Builds lib/lib_jaxsplat_ffi.so.

    python -m jaxsplat_ffi.build                 # all supported GPU generations
    JAXSPLAT_CUDA_ARCHS="86;89" python -m jaxsplat_ffi.build   # only some

The library is a "fat binary": it contains machine code for every listed
architecture, so one build runs on all common GPUs
(RTX 20xx = 75, A100 = 80, RTX 30xx = 86, RTX 40xx/L40S = 89, H100 = 90,
B200 = 100, RTX 50xx/RTX PRO Blackwell = 120). PTX for the newest architecture
is embedded as well, so future GPUs can JIT-compile it.

The CUDA runtime is linked statically, so the library does not depend on the
CUDA toolkit at run time. The GPU driver must support the CUDA major version
of the nvcc used for the build (CUDA 12.x: driver >= 525, CUDA 13.x: >= 580).
The output is named lib/lib_jaxsplat_ffi.cuda<major>.so, so builds for
different CUDA versions can coexist; the newest one the driver supports is
loaded at import.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jax

PACKAGE_DIR = Path(__file__).parent
CSRC = PACKAGE_DIR / "csrc"
LIB_DIR = PACKAGE_DIR / "lib"

# (architecture, minimum nvcc version that supports it, maximum (exclusive) or None)
ARCHS = [
    ("75", (11, 0), None),
    ("80", (11, 0), None),
    ("86", (11, 1), None),
    ("89", (11, 8), None),
    ("90", (11, 8), None),
    ("100", (12, 8), None),
    ("120", (12, 8), None),
]


def find_nvcc() -> str:
    candidates = [
        os.environ.get("CUDACXX"),
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError("nvcc not found. Set CUDACXX or add CUDA's bin dir to PATH.")


def nvcc_version(nvcc: str) -> tuple[int, int]:
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"release (\d+)\.(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse nvcc version from:\n{out}")
    return int(m.group(1)), int(m.group(2))


def select_archs(version: tuple[int, int]) -> list[str]:
    override = os.environ.get("JAXSPLAT_CUDA_ARCHS")
    if override:
        return [a.strip() for a in override.replace(",", ";").split(";") if a.strip()]
    return [a for a, lo, hi in ARCHS if version >= lo and (hi is None or version < hi)]


def build(verbose: bool = True) -> Path:
    nvcc = find_nvcc()
    version = nvcc_version(nvcc)
    archs = select_archs(version)
    if not archs:
        raise RuntimeError(f"No CUDA architectures selected for nvcc {version}")

    gencode = []
    for a in archs:
        gencode += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    newest = archs[-1]
    gencode += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    sources = [CSRC / "ffi.cu", *sorted((CSRC / "kernels").glob("*.cu"))]
    # One file per CUDA major version; _ffi.py picks the one the driver supports.
    out = LIB_DIR / f"lib_jaxsplat_ffi.cuda{version[0]}.so"
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.name + f".tmp{os.getpid()}")

    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-cudart", "static",
        "--threads", "0",
        # Silence warnings coming from JAX's own FFI headers.
        "-diag-suppress", "940,2473",
        "-Xcompiler", "-Wno-return-type",
        "-I", str(CSRC),
        "-I", str(CSRC / "kernels"),
        "-I", os.environ.get("JAXSPLAT_XLA_INCLUDE", jax.ffi.include_dir()),
        *gencode,
        *map(str, sources),
        "-o", str(tmp_out),
    ]
    if verbose:
        print(f"nvcc {version[0]}.{version[1]} ({nvcc}), architectures: {', '.join(archs)}")
        print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    # Atomic replace, so jobs that load the library while another job builds
    # never see a half-written file.
    os.replace(tmp_out, out)
    if verbose:
        print(f"Built {out}")
    return out


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
