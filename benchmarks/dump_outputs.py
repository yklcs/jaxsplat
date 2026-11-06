"""Renders the correctness scenes and saves images + gradients to an .npz file.

Run once per implementation, then compare with compare_outputs.py:

    # environment with JAX 0.5 + original jaxsplat
    python benchmarks/dump_outputs.py --impl jaxsplat --out results/legacy.npz
    # environment with jaxsplat_ffi
    python benchmarks/dump_outputs.py --impl ffi --out results/ffi.npz

Loss used for the gradients: sum(image * W) with a fixed random W, so every
pixel contributes. Each scene is evaluated twice to measure run-to-run noise
(the backward kernel uses atomicAdd, so gradients are not bit-reproducible).
"""

import argparse
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import scenes


def load_renderer(name):
    return importlib.import_module({"jaxsplat": "jaxsplat", "ffi": "jaxsplat_ffi"}[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["jaxsplat", "ffi"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    renderer = load_renderer(args.impl)
    results = {"meta/jax_version": np.array(jax.__version__), "meta/impl": np.array(args.impl)}

    for name, scene in scenes.correctness_scenes().items():
        render = scenes.make_render_fn(renderer, scene)
        weights = jnp.asarray(scenes.loss_weights(scene["img_shape"]))
        inputs = [jnp.asarray(scene[k]) for k in scenes.DIFF_KEYS]

        def loss(*xs):
            return jnp.sum(render(*xs) * weights)

        img = jax.jit(render)(*inputs)
        grad_fn = jax.jit(jax.grad(loss, argnums=tuple(range(len(inputs)))))
        for run in (0, 1):
            grads = grad_fn(*inputs)
            for key, g in zip(scenes.DIFF_KEYS, grads):
                results[f"{name}/run{run}/grad_{key}"] = np.asarray(g)
        results[f"{name}/img"] = np.asarray(img)
        print(f"{name:16s} img mean {float(img.mean()):.5f}  "
              f"|grad means3d| {float(jnp.linalg.norm(grads[0])):.4g}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **results)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
