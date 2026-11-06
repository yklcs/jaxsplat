# jaxsplat_ffi

![](./jaxsplat.gif)

jaxsplat_ffi is a port of 3D Gaussian Splatting to [JAX](https://github.com/google/jax).
Fully differentiable, CUDA accelerated.

It is a fork of [jaxsplat](https://github.com/yklcs/jaxsplat) that runs the same kernels through JAX's XLA FFI, so it works with JAX ≥ 0.6.
`import jaxsplat_ffi as jaxsplat` is a drop-in switch.

## Installation

Requires a working CUDA toolchain to install. Simply `pip install`ing directly from source should build and install jaxsplat_ffi:

```shell
$ python -m venv venv && . venv/bin/activate
$ pip install "jaxsplat-ffi @ git+https://github.com/dwoiwode/jaxsplat"
```

## Usage

The primary function of jaxsplat_ffi is `jaxsplat.render`, which renders 3D Gaussians to a 2D image differentiably.
View [the rendering API docs](./render/index) for more complete docs.

```python
import jaxsplat_ffi as jaxsplat

img = jaxsplat.render(
    means3d,
    scales,
    quats,
    colors,
    opacities,
    viewmat=viewmat,
    background=background,
    img_shape=img_shape,
    f=f,
    c=c,
    glob_scale=glob_scale,
    clip_thresh=clip_thresh,
    block_size=block_size,
)
```

## Bibliography

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., SIGGRAPH 2023)
- [gsplat](https://github.com/nerfstudio-project/gsplat)

```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:

    self
    GitHub <https://github.com/dwoiwode/jaxsplat/blob/main/examples/single_image.py>

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API Reference:

    render/index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Examples:

    Single image <https://github.com/dwoiwode/jaxsplat/blob/main/examples/single_image.py>
```
