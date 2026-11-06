# jaxsplat_ffi vs. original jaxsplat – results

Measured 2026-11-02 on an **NVIDIA RTX PRO 4500 Blackwell** (driver 595.91, CUDA 13.2).

| | Original | New |
|---|---|---|
| Package | `jaxsplat` ([VersesTech fork](https://github.com/VersesTech/jaxsplat), commit 2ec2a05) | `src/jaxsplat_ffi` |
| JAX | 0.5.3 | 0.12.0.1 |
| Binding | legacy custom call (`register_custom_call_target`) | XLA FFI (`jax.ffi`) |
| CUDA kernels | gsplat-derived | same kernels, with the changes listed below |

The VersesTech fork is the same code as the original jaxsplat this repository
is forked from, apart from one minor bug fix.

## Correctness

`dump_outputs.py` renders the test scenes with both implementations and saves
image + gradients of `sum(image * W)` w.r.t. means, scales, quaternions,
colours and opacities. Scenes: 1k Gaussians on a non-square 128×160 image, 5k
with tile size 8, 50k at 512², and a scene with everything behind the camera.

- **Images: bit-identical** in all scenes (max |diff| = 0).
- **Gradients** differ from the original by ~0.5–1.3% (relative). All of it
  comes from the alpha cap fix (see below): with only the old 0.99 cap
  restored, the gradients match the original within float noise (relative
  error ≤ 8.3e-7, the same size as the original's own run-to-run noise from
  `atomicAdd`). Batching, culling and the launch changes are exact.
- `tests/test_jaxsplat_ffi.py` (11 tests) additionally checks gradients
  against finite differences, `vmap` vs. a Python loop, zero Gaussians, nothing
  visible, background gradient, tile sizes, nearly opaque Gaussians and invalid
  arguments. It also passes with `CUDA_FORCE_PTX_JIT=1` (the path GPUs without
  native code take). The nearly-opaque test fails on the original.

## Speed

Median of repeated jitted calls, including device sync. "fwd+bwd" =
`value_and_grad` w.r.t. all five Gaussian parameters. Random scenes from
`scenes.py`.

| Case | original fwd | ffi fwd | speed-up | original fwd+bwd | ffi fwd+bwd | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| N=1k, 512x512 | 0.54 ms | 0.30 ms | 1.80x | 1.33 ms | 0.55 ms | 2.42x |
| N=10k, 512x512 | 1.43 ms | 0.37 ms | 3.86x | 3.06 ms | 0.91 ms | 3.36x |
| N=100k, 512x512 | 1.47 ms | 0.53 ms | 2.77x | 3.50 ms | 1.64 ms | 2.13x |
| N=1M, 512x512 | 2.50 ms | 1.46 ms | 1.71x | 6.73 ms | 4.24 ms | 1.59x |
| N=100k, 256x256 | 1.52 ms | 0.45 ms | 3.38x | 2.90 ms | 1.26 ms | 2.30x |
| N=100k, 1024x1024 | 2.04 ms | 0.95 ms | 2.15x | 5.87 ms | 3.69 ms | 1.59x |

These scenes have no invisible Gaussians, so opacity culling does not show
here; batched (`vmap`) rendering is not measured either. Measured on an
otherwise idle GPU, before the pinned-copy change (below).

Where the speed-up comes from: the original synchronises the GPU stream up to
10 times per call and calls `cudaMalloc`/`cudaFree` for every temporary buffer.
The port uses XLA's allocator and syncs once per rasterize call (to read the
number of tile intersections). The gain is largest for small and medium scenes,
where this overhead dominates; at 1M Gaussians the kernels themselves dominate.

## Changes to the kernels and the FFI layer

| Change | Effect |
|---|---|
| Cull Gaussians with opacity < 1/255 in the projection | They no longer go through sorting and every tile loop. Image unchanged (the rasterizer skips them anyway) |
| Native batched rendering (`vmap_method="broadcast_all"`, `blockIdx.z` = image) | One launch per kernel and one host sync per pass for the whole batch |
| Pinned host buffer for the intersection count | A copy into pageable memory kept the stream blocked 0.5–2 ms after each sync |
| Kernels initialise their outputs; removed unused `v_xy_abs` | Fewer memsets, fewer atomics/warp reductions per Gaussian |
| `background` passed as a device pointer | `background` is differentiable |
| **Bug fix:** backward alpha cap 0.99 → 0.999 (= forward) | The original has up to 14% error in colour gradients for nearly opaque Gaussians (regression test added) |

## Behavioural differences to the original

- Works with JAX ≥ 0.6 (tested 0.12.0.1).
- `vmap` works natively, including `vmap` of `grad`.
- `background` gets a gradient (original: none).
- Principal point is passed as float (some earlier ports truncated it to int).

## Reproduce

See `README.md` ("Comparing with the original").

## Possible next steps

1. **Sync-free rasterization.** The sort buffers are sized by the intersection
   count, which forces a host read per rasterize call. Options: a capacity
   estimate learned on the host with an exact on-device fallback; or caching
   the forward's sorted lists for the backward pass (saves one sync and one sort).
2. **Deterministic backward pass** (optional): replace the `atomicAdd`
   accumulation so gradients are bit-reproducible.
3. Kernel micro-optimisations (shared-memory colours in the forward pass,
   fewer atomics via block-level reduction) for large scenes.
