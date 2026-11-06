"""Loads the compiled CUDA library and registers its XLA FFI targets.

Several builds can live side by side in ``lib/``, one per CUDA major version
(``lib_jaxsplat_ffi.cuda12.so``, ``...cuda13.so``). A library built with
CUDA X needs a driver that supports CUDA X, so on import we ask the driver
which CUDA version it supports and load the newest compatible build. This lets
one installation serve machines with different drivers.
"""

import ctypes
import os
import re
from pathlib import Path

import jax

_PACKAGE_DIR = Path(__file__).parent
LIB_DIR = _PACKAGE_DIR / "lib"
LIB_PATTERN = re.compile(r"lib_jaxsplat_ffi\.cuda(\d+)\.so$")

TARGETS = {
    "jaxsplat_ffi_project_fwd": "JaxsplatProjectFwd",
    "jaxsplat_ffi_project_bwd": "JaxsplatProjectBwd",
    "jaxsplat_ffi_rasterize_fwd": "JaxsplatRasterizeFwd",
    "jaxsplat_ffi_rasterize_bwd": "JaxsplatRasterizeBwd",
}


def library_filename(cuda_major: int) -> str:
    return f"lib_jaxsplat_ffi.cuda{cuda_major}.so"


def driver_cuda_version() -> tuple[int, int] | None:
    """Highest CUDA version the installed driver supports, e.g. (12, 4)."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        if libcuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
        return version.value // 1000, (version.value % 1000) // 10
    except OSError:
        return None


def available_builds() -> dict[int, Path]:
    builds = {}
    if LIB_DIR.exists():
        for p in LIB_DIR.iterdir():
            m = LIB_PATTERN.match(p.name)
            if m:
                builds[int(m.group(1))] = p
    return builds


def library_path() -> Path:
    """Picks the library to load (override with JAXSPLAT_FFI_LIB)."""
    override = os.environ.get("JAXSPLAT_FFI_LIB")
    if override:
        return Path(override)

    builds = available_builds()
    if not builds:
        raise ImportError(
            f"jaxsplat_ffi: no compiled library in {LIB_DIR}.\n"
            "Build it with:  python -m jaxsplat_ffi.build"
        )
    driver = driver_cuda_version()
    if driver is None:  # cannot tell; take the oldest build (most compatible)
        return builds[min(builds)]
    compatible = [major for major in builds if major <= driver[0]]
    if not compatible:
        raise ImportError(
            f"jaxsplat_ffi: the GPU driver supports CUDA {driver[0]}.{driver[1]}, but "
            f"only builds for CUDA {sorted(builds)} exist in {LIB_DIR}. Build with an "
            f"older CUDA toolkit, see "
            "README.md."
        )
    return builds[max(compatible)]


_lib = None
_lib_path = None


def ensure_loaded():
    """Loads the library and registers the FFI targets (once)."""
    global _lib, _lib_path
    if _lib is None:
        path = library_path()
        lib = ctypes.cdll.LoadLibrary(str(path))
        for target, symbol in TARGETS.items():
            jax.ffi.register_ffi_target(
                target, jax.ffi.pycapsule(getattr(lib, symbol)), platform="CUDA"
            )
        _lib, _lib_path = lib, path
    return _lib


def loaded_library() -> Path | None:
    """Path of the library in use (None before the first render)."""
    return _lib_path
