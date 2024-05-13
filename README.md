# jaxsplat

A port of 3D Gaussian Splatting to JAX.
Fully differentiable, CUDA accelerated.

## Setup

Requires a working CUDA toolchain to install.
Simply `pip install`ing should work:

```shell
$ pip install git+https://github.com/yklcs/jaxsplat
```

## Method

We use modified versions of [gsplat](https://github.com/nerfstudio-project/gsplat)'s kernels.
The [original INRIA implementation](https://github.com/graphdeco-inria/diff-gaussian-rasterization) uses a custom license and contains dynamically shaped tensors which are harder to port to JAX/XLA.
