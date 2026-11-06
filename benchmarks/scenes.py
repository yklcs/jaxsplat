"""Test scenes shared by the reference, comparison and benchmark scripts.

Plain numpy so the exact same inputs are produced in an environment with
JAX 0.5 + the original jaxsplat and in one (JAX 0.12 + jaxsplat_ffi).
"""

import numpy as np


def _quats(rng, n):
    q = rng.normal(size=(n, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _view(distance):
    v = np.eye(4, dtype=np.float32)
    v[2, 3] = distance
    return v


def random_scene(n, img_shape, seed=0, spread=3.0, scale=(0.05, 0.3), distance=8.0, block_size=16):
    rng = np.random.RandomState(seed)
    h, w = img_shape
    return dict(
        means3d=rng.uniform(-spread, spread, size=(n, 3)).astype(np.float32),
        scales=rng.uniform(*scale, size=(n, 3)).astype(np.float32),
        quats=_quats(rng, n),
        colors=rng.uniform(0, 1, size=(n, 3)).astype(np.float32),
        opacities=rng.uniform(0.05, 1, size=(n, 1)).astype(np.float32),
        viewmat=_view(distance),
        background=np.array([0.1, 0.2, 0.3], np.float32),
        img_shape=(h, w),
        f=(0.8 * w, 0.8 * w),
        c=(w / 2, h / 2),
        glob_scale=1.0,
        clip_thresh=0.01,
        block_size=block_size,
    )


def behind_camera_scene():
    """Every Gaussian is behind the camera: nothing is drawn."""
    s = random_scene(500, (64, 96), seed=3)
    s["means3d"][:, 2] = -50.0
    return s


def correctness_scenes():
    return {
        "small_nonsquare": random_scene(1_000, (128, 160), seed=0),
        "medium_block8": random_scene(5_000, (200, 300), seed=1, block_size=8),
        "large": random_scene(50_000, (512, 512), seed=2, spread=4.0, scale=(0.01, 0.1)),
        "behind_camera": behind_camera_scene(),
    }


ARRAY_KEYS = ("means3d", "scales", "quats", "colors", "opacities", "viewmat", "background")
DIFF_KEYS = ("means3d", "scales", "quats", "colors", "opacities")


def loss_weights(img_shape, seed=123):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(*img_shape, 3)).astype(np.float32)


def make_render_fn(renderer, scene):
    """Returns render(means3d, scales, quats, colors, opacities) for a scene."""
    import jax.numpy as jnp

    static = dict(
        viewmat=jnp.asarray(scene["viewmat"]),
        background=jnp.asarray(scene["background"]),
        img_shape=tuple(int(x) for x in scene["img_shape"]),
        f=tuple(float(x) for x in scene["f"]),
        c=tuple(float(x) for x in scene["c"]),
        glob_scale=float(scene["glob_scale"]),
        clip_thresh=float(scene["clip_thresh"]),
        block_size=int(scene["block_size"]),
    )

    def render(means3d, scales, quats, colors, opacities):
        return renderer.render(means3d, scales, quats, colors, opacities, **static)

    return render
