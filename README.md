# jaxsplat_ffi

![](./docs/jaxsplat.gif)

A port of 3D Gaussian Splatting to JAX.
Fully differentiable, CUDA accelerated.

This is a fork of [jaxsplat](https://github.com/yklcs/jaxsplat) that runs the
same CUDA kernels through JAX's **XLA FFI**, so it works with current JAX
(≥ 0.6, tested with 0.12). The original registers its kernels with
`register_custom_call_target`, which JAX 0.6 removed. The package is called
`jaxsplat_ffi`, so it can be installed next to the original, and it keeps the
original API: `import jaxsplat_ffi as jaxsplat` is a drop-in switch.

## Installation

Requires a working CUDA toolchain (`nvcc`, CUDA ≥ 11.8) to install.
Simply `pip install`ing directly from source builds and installs jaxsplat_ffi:

```shell
$ python -m venv venv && . venv/bin/activate
$ pip install "jaxsplat-ffi @ git+https://github.com/dwoiwode/jaxsplat"
```

Install JAX with GPU support as usual (e.g. `jax[cuda12]`), or use the extras
`jaxsplat-ffi[cuda12]` / `jaxsplat-ffi[cuda13]`.

The build compiles one library for all common GPU generations (sm_75 to
sm_120) and takes a few minutes. Options (environment variables):

- `CUDACXX=/path/to/nvcc`: which `nvcc` to use (default: `PATH`, then `/usr/local/cuda/bin/nvcc`).
- `JAXSPLAT_CUDA_ARCHS="86;89"`: compile only for these GPU architectures (faster).
- `JAXSPLAT_SKIP_BUILD=1`: install without compiling; build later with `python -m jaxsplat_ffi.build`.

## Usage

The primary function of this library is `jaxsplat.render`:

```python
import jaxsplat_ffi as jaxsplat

img = jaxsplat.render(
    means3d,   # jax.Array (N, 3)
    scales,    # jax.Array (N, 3)
    quats,     # jax.Array (N, 4) normalized
    colors,    # jax.Array (N, 3)
    opacities, # jax.Array (N, 1)
    viewmat=viewmat,         # jax.Array (4, 4)
    background=background,   # jax.Array (3,)
    img_shape=img_shape,     # tuple[int, int] = (H, W)
    f=f,                     # tuple[float, float] = (fx, fy)
    c=c,                     # tuple[float, float] = (cx, cy)
    glob_scale=glob_scale,   # float
    clip_thresh=clip_thresh, # float
    block_size=block_size,   # int <= 16
)
```

The rendered output is differentiable w.r.t. `means3d`, `scales`, `quats`, `colors`, `opacities` and `background`.
`jax.vmap` over any of the inputs renders the whole batch in one launch per kernel.

Alternatively, `jaxsplat.project` projects 3D Gaussians to 2D, and `jaxsplat.rasterize` sorts and rasterizes 2D Gaussians.
`jaxsplat.render` successively calls `jaxsplat.project` and `jaxsplat.rasterize` under the hood.

## Examples

See [/examples](./examples) for examples.
These can be ran like the following:

```shell
$ python -m venv venv && . venv/bin/activate
$ pip install -r examples/requirements.txt

# Train Gaussians on a single image
$ python -m examples.single_image input.png
```

## Differences from the original jaxsplat

| | jaxsplat | jaxsplat_ffi |
|---|---|---|
| JAX | < 0.6 | ≥ 0.6 (tested 0.12.0.1) |
| Binding | legacy custom call (pybind11) | XLA FFI (`jax.ffi`) |
| Build | CMake + pybind11, for the local GPU | hatch build hook, one binary for sm_75–sm_120 |
| Images | – | bit-identical |
| Gradients | backward pass caps alpha at 0.99, forward at 0.999 (up to 14% error for nearly opaque Gaussians) | 0.999 in both; matches finite differences |
| `vmap` | not supported | native, also `vmap` of `grad`; one launch per kernel for the whole batch |
| `background` | no gradient | differentiable |
| Temporary GPU memory | `cudaMalloc`/`cudaFree` per buffer, up to 10 stream syncs per call | XLA's allocator, one sync per rasterize call |
| Invisible Gaussians | sorted and rasterized | culled before sorting (opacity < 1/255) |
| Speed (RTX PRO 4500 Blackwell) | – | 1.7–3.9× faster forward, 1.6–3.4× faster forward+backward |

Measurements and how to reproduce them: [benchmarks/RESULTS.md](./benchmarks/RESULTS.md).

## Method

We use modified versions of [gsplat](https://github.com/nerfstudio-project/gsplat)'s kernels.
The [original INRIA implementation](https://github.com/graphdeco-inria/diff-gaussian-rasterization) uses a custom license and contains dynamically shaped tensors which are harder to port to JAX/XLA.

## Development

```shell
$ git clone https://github.com/dwoiwode/jaxsplat && cd jaxsplat
$ pip install -e .                  # compiles into src/jaxsplat_ffi/lib/ (only if no library is there yet)
$ python tests/test_jaxsplat_ffi.py
```

Rebuild after changing the CUDA code with `python -m jaxsplat_ffi.build`.

```
src/jaxsplat_ffi/
  __init__.py     render / project / rasterize with custom VJPs
  _ffi.py         picks and loads the compiled library, registers the FFI targets
  build.py        compiles csrc/ into lib/lib_jaxsplat_ffi.cuda<major>.so
  csrc/ffi.cu     XLA FFI handlers (project fwd/bwd, rasterize fwd/bwd)
  csrc/kernels/   CUDA kernels
hatch_build.py    runs build.py when pip builds the wheel
tests/            unit tests (gradients vs. finite differences, vmap, edge cases)
benchmarks/       correctness and speed against the original jaxsplat
examples/         training on a single image; speed against gsplat and the INRIA rasterizer
docs/             documentation (Sphinx)
```

### CUDA versions

The CUDA runtime is linked statically, so at run time only the driver
matters: a build with CUDA 12.x needs driver ≥ 525, CUDA 13.x needs ≥ 580.
The file name carries the CUDA major version (`lib_jaxsplat_ffi.cuda12.so`,
`...cuda13.so`), and several builds can sit in `lib/` side by side. On import
the package loads the newest build the driver supports.
`jaxsplat_ffi._ffi.library_path()` tells which one; `JAXSPLAT_FFI_LIB`
overrides the choice.

### Comparing with the original

Needs a second environment with JAX 0.5 and the original jaxsplat:

```shell
$ python -m venv legacy-venv
$ legacy-venv/bin/pip install "jax[cuda12]>=0.5,<0.6" "jaxsplat @ git+https://github.com/yklcs/jaxsplat"
```

Then, with `venv` being the environment with jaxsplat_ffi:

```shell
$ export PYTHONPATH=benchmarks
$ legacy-venv/bin/python benchmarks/dump_outputs.py --impl jaxsplat --out results/legacy.npz
$ venv/bin/python        benchmarks/dump_outputs.py --impl ffi      --out results/ffi.npz
$ venv/bin/python        benchmarks/compare_outputs.py results/legacy.npz results/ffi.npz
$ legacy-venv/bin/python benchmarks/benchmark.py --impl jaxsplat --out results/bench_legacy.json
$ venv/bin/python        benchmarks/benchmark.py --impl ffi      --out results/bench_ffi.json
$ venv/bin/python        benchmarks/benchmark.py --report results/bench_legacy.json results/bench_ffi.json
```

Images should be identical; gradients differ by ~1% because of the alpha cap fix.

### Implementation notes

- **Batched rendering**: all FFI calls use `vmap_method="broadcast_all"`.
  Under `vmap` the CUDA side receives the whole batch and renders it in one
  launch per kernel (`blockIdx.z` = image, tile ids offset per image), with
  one host sync per pass instead of one per image. With a single image, the
  rasterizer uses the projection's `cum_tiles_hit`. For a batch it recounts
  tile hits itself (`count_tiles_hit`), because JAX may have broadcast one
  projection to several images, and then those sums restart in every image.
- **Opacity culling**: `render` passes the opacities to the projection, which
  drops Gaussians with opacity < 1/255. The rasterizer would skip them anyway
  (alpha ≤ opacity), so the image is unchanged, but they no longer go through
  sorting and every tile loop. This matters when many Gaussians are fully
  transparent.
- **Alpha cap fix**: forward and backward both cap alpha at 0.999. The
  original backward pass used 0.99, which gave wrong gradients for nearly
  opaque Gaussians (up to 14% error in colour gradients, see
  `tests/test_jaxsplat_ffi.py::test_gradients_nearly_opaque_gaussians`).
  Images are unchanged; gradients differ from the original by ~1% on typical
  scenes because of this fix.
- **Fewer launches**: kernels initialise their own outputs (no per-output
  memsets); the unused `v_xy_abs` gradient (3DGS densification) is gone.
- **Scratch memory** (sort keys, tile bins, CUB workspaces) comes from XLA's
  allocator through the FFI. XLA rejects explicit alignments ("Unsupported
  alignment"), so we request alignment 1 and check that the buffer is at
  least 16-byte aligned (XLA's GPU buffers are 256-byte aligned).
- **One host sync per rasterize call**: the number of tile/Gaussian
  intersections decides how big the sort buffers are, so it must be read on
  the host. It is copied into a pinned host buffer: a copy into ordinary
  (pageable) memory kept the stream blocked for 0.5–2 ms afterwards.
- **Non-determinism**: the backward kernel accumulates with `atomicAdd`
  (as in the original), so gradients differ in the last float bits between
  runs.
- No gradient w.r.t. `viewmat` (as in the original).
- **Allocator**: don't set `XLA_PYTHON_CLIENT_ALLOCATOR=platform`. It frees
  memory with a device sync on every free, and this renderer allocates its
  scratch buffers through XLA on every call, so it becomes very slow.
