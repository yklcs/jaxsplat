"""Compiles the CUDA library while pip/uv builds the wheel.

So `pip install git+https://...` gives a working package. Needs `nvcc`
(found via CUDACXX, PATH or /usr/local/cuda). Environment variables:

    JAXSPLAT_CUDA_ARCHS="86;89"   build only these GPU architectures (faster)
    JAXSPLAT_SKIP_BUILD=1         do not compile (then run `python -m jaxsplat_ffi.build` later)

Editable installs compile only if `lib/` has no library yet, so `uv sync`
does not recompile every time.
"""

import importlib.util
import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CudaBuildHook(BuildHookInterface):
    PLUGIN_NAME = "jaxsplat-cuda"

    def initialize(self, version, build_data):
        if self.target_name != "wheel" or os.environ.get("JAXSPLAT_SKIP_BUILD"):
            return

        spec = importlib.util.spec_from_file_location(
            "_jaxsplat_build", Path(self.root) / "src" / "jaxsplat_ffi" / "build.py"
        )
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)

        existing = sorted(build.LIB_DIR.glob("lib_jaxsplat_ffi.cuda*.so"))
        if version == "editable" and existing:
            return  # the editable package loads lib/ from the source tree directly

        lib = build.build()
        build_data["pure_python"] = False  # platform-specific wheel tag
        build_data["infer_tag"] = True
        for so in sorted(build.LIB_DIR.glob("lib_jaxsplat_ffi.cuda*.so")):
            if version != "editable":
                build_data["force_include"][str(so)] = f"jaxsplat_ffi/lib/{so.name}"
        self.app.display_info(f"jaxsplat_ffi: built {lib.name}")
