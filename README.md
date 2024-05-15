# jaxsplat

![](./docs/jaxsplat.gif)

A port of 3D Gaussian Splatting to JAX.
Fully differentiable, CUDA accelerated.

## Installation

Requires a working CUDA toolchain to install.
Simply `pip install`ing directly from source should build and install jaxsplat:

```shell
$ python -m venv venv && . venv/bin/activate
$ pip install git+https://github.com/yklcs/jaxsplat
```

## Usage

The primary function of this library is `jaxsplat.render`:

```python
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
    c=c,                     # tuple[int, int] = (cx, cy)
    glob_scale=glob_scale,   # float
    clip_thresh=clip_thresh, # float
    block_size=block_size,   # int <= 16
)
```

The rendered output is differentiable w.r.t. `means3d`, `scales`, `quats`, `colors`, and `opacities`.

Alternatively, `jaxsplat.project` projects 3D Gaussians to 2D, and `jaxsplat.rasterize` sorts and rasterizes 2D Gaussians.
`jaxsplat.render` successively calls `jaxsplat.project` and `jaxsplat.rasterize` under the hood.

## Examples

See [/examples](./examples) for examples.
These can be ran like the following:

```shell
$ python -m venv venv && . venv/bin/activate
$ pip install -r examples/requirements.txt

# Train Gaussians on a single image
$ python -m examples.single_image.py input.png
```

## Method

We use modified versions of [gsplat](https://github.com/nerfstudio-project/gsplat)'s kernels.
The [original INRIA implementation](https://github.com/graphdeco-inria/diff-gaussian-rasterization) uses a custom license and contains dynamically shaped tensors which are harder to port to JAX/XLA.
