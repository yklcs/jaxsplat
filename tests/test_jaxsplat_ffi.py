"""Tests for jaxsplat_ffi (needs a CUDA GPU).

    PYTHONPATH=src python tests/test_jaxsplat_ffi.py      # plain
    PYTHONPATH=src python -m pytest tests/test_jaxsplat_ffi.py   # or with pytest

Comparison against the original jaxsplat lives in benchmarks/.
"""

import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import jaxsplat_ffi as jaxsplat


def make_scene(n=300, img_shape=(48, 64), seed=0):
    rng = np.random.RandomState(seed)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    params = dict(
        means3d=jnp.asarray(rng.uniform(-2, 2, (n, 3)), jnp.float32),
        scales=jnp.asarray(rng.uniform(0.1, 0.4, (n, 3)), jnp.float32),
        quats=jnp.asarray(q),
        colors=jnp.asarray(rng.uniform(0, 1, (n, 3)), jnp.float32),
        opacities=jnp.asarray(rng.uniform(0.1, 0.9, (n, 1)), jnp.float32),
        background=jnp.asarray([0.2, 0.5, 0.7], jnp.float32),
    )
    h, w = img_shape
    static = dict(
        viewmat=jnp.eye(4).at[2, 3].set(8.0),
        img_shape=img_shape,
        f=(1.2 * w, 1.2 * w),
        c=(w / 2, h / 2),
        glob_scale=1.0,
        clip_thresh=0.01,
    )
    return params, static


def render(params, static):
    return jaxsplat.render(
        params["means3d"], params["scales"], params["quats"], params["colors"],
        params["opacities"], background=params["background"], **static,
    )


def test_output_shape_and_range():
    params, static = make_scene()
    img = jax.jit(lambda p: render(p, static))(params)
    assert img.shape == (48, 64, 3) and img.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(img)))
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0 + 1e-5


def test_gradients_match_finite_differences():
    """Analytic gradients vs. central finite differences.

    The rendered image is only piecewise smooth: depth re-sorting and the
    "skip alpha < 1/255" cut-off cause small jumps. Moving all Gaussians at
    once crosses many of them, so geometry is checked one coordinate at a time
    (the most influential ones). Colours and background are smooth and are
    checked along a random direction.
    """
    params, static = make_scene(n=150, img_shape=(40, 40), seed=1)
    weights = jnp.asarray(np.random.RandomState(7).normal(size=(40, 40, 3)), jnp.float32)

    def loss(p):
        return jnp.sum(render(p, static) * weights)

    grads = jax.jit(jax.grad(loss))(params)
    loss_jit = jax.jit(loss)

    def central_diff(key, direction, eps):
        plus = dict(params, **{key: params[key] + eps * direction})
        minus = dict(params, **{key: params[key] - eps * direction})
        return (float(loss_jit(plus)) - float(loss_jit(minus))) / (2 * eps)

    rng = np.random.RandomState(3)
    for key in ("colors", "background"):
        v = jnp.asarray(rng.normal(size=params[key].shape), jnp.float32)
        fd, an = central_diff(key, v, 1e-2), float(jnp.vdot(grads[key], v))
        assert abs(fd - an) <= 1e-3 * max(abs(an), 1.0), f"{key}: fd {fd:.5g} vs analytic {an:.5g}"

    for key in ("means3d", "scales", "quats", "opacities"):
        g = np.asarray(grads[key])
        flat_idx = np.argsort(-np.abs(g).ravel())[:6]
        for fi in flat_idx:
            idx = np.unravel_index(fi, g.shape)
            e = jnp.zeros_like(params[key]).at[idx].set(1.0)
            fd, an = central_diff(key, e, 1e-2), float(g[idx])
            err = abs(fd - an) / max(abs(an), 1e-2)
            assert err < 0.15, f"{key}{idx}: fd {fd:.5g} vs analytic {an:.5g} (rel err {err:.3g})"


def test_gradients_nearly_opaque_gaussians():
    """Regression test: the original backward pass capped alpha at 0.99 while the
    forward pass used 0.999, so gradients were wrong whenever a Gaussian was
    nearly opaque (common in trained scenes, where many opacities sit at 1)."""
    # Large, fully opaque, overlapping Gaussians. The image is linear in the
    # colours, so finite differences are exact there: the original jaxsplat is
    # off by up to 14% on this scene, the fixed kernel by 0.
    params, static = make_scene(n=40, img_shape=(40, 40), seed=4)
    params["scales"] = jnp.asarray(np.random.RandomState(5).uniform(0.3, 0.6, (40, 3)), jnp.float32)
    params["opacities"] = jnp.ones((40, 1), jnp.float32)
    weights = jnp.asarray(np.random.RandomState(8).normal(size=(40, 40, 3)), jnp.float32)
    loss = lambda p: jnp.sum(render(p, static) * weights)  # noqa: E731
    color_errors = fd_errors(loss, params, ("colors",), top=8)
    assert max(color_errors) < 1e-3, f"colour gradient errors {np.round(color_errors, 4)}"
    mean_errors = fd_errors(loss, params, ("means3d",), top=8)
    assert max(mean_errors) < 0.1, f"position gradient errors {np.round(mean_errors, 3)}"


def fd_errors(loss, params, keys, eps=1e-2, top=6):
    """Relative error |fd - analytic| / |analytic| for the `top` largest gradient entries."""
    grads = jax.jit(jax.grad(loss))(params)
    loss_jit = jax.jit(loss)
    errors = []
    for key in keys:
        g = np.asarray(grads[key])
        for fi in np.argsort(-np.abs(g).ravel())[:top]:
            idx = np.unravel_index(fi, g.shape)
            e = jnp.zeros_like(params[key]).at[idx].set(eps)
            fd = (float(loss_jit(dict(params, **{key: params[key] + e})))
                  - float(loss_jit(dict(params, **{key: params[key] - e})))) / (2 * eps)
            errors.append(abs(fd - float(g[idx])) / max(abs(float(g[idx])), 1e-2))
    return errors


def test_culling_does_not_change_output():
    """Gaussians with opacity < 1/255 are skipped early; image and gradients must not change."""
    params, static = make_scene(n=400, seed=6)
    op = np.asarray(params["opacities"]).copy()
    op[::2] = 1.0 / 300.0  # half of them invisible
    params["opacities"] = jnp.asarray(op)

    def manual(p):  # rasterize everything, no culling
        xys, depths, radii, conics, _, cum = jaxsplat.project(
            p["means3d"], p["scales"], p["quats"], static["viewmat"], img_shape=static["img_shape"],
            f=static["f"], c=static["c"], glob_scale=1.0, clip_thresh=0.01)
        return jaxsplat.rasterize(p["colors"], p["opacities"], p["background"], xys, depths, radii,
                                  conics, cum, img_shape=static["img_shape"])

    np.testing.assert_array_equal(np.asarray(render(params, static)), np.asarray(manual(params)))
    g1 = jax.grad(lambda p: render(p, static).sum())(params)
    g2 = jax.grad(lambda p: manual(p).sum())(params)
    for key in ("means3d", "scales", "colors"):
        np.testing.assert_allclose(np.asarray(g1[key]), np.asarray(g2[key]), rtol=1e-4, atol=1e-6)


def test_background_gradient():
    """d img / d background = final transmittance; nothing drawn => gradient = #pixels."""
    params, static = make_scene()
    params["means3d"] = params["means3d"].at[:, 2].set(-50.0)  # all behind camera
    g = jax.grad(lambda b: jnp.sum(render(dict(params, background=b), static)))(params["background"])
    np.testing.assert_allclose(np.asarray(g), 48 * 64, rtol=1e-6)


def test_nothing_visible_gives_background():
    params, static = make_scene()
    params["means3d"] = params["means3d"].at[:, 2].set(-50.0)
    img = render(params, static)
    np.testing.assert_allclose(np.asarray(img), np.broadcast_to(params["background"], img.shape))
    grads = jax.grad(lambda p: render(p, static).sum())(params)
    for key in ("means3d", "scales", "quats", "colors", "opacities"):
        assert float(jnp.abs(grads[key]).max()) == 0.0


def test_zero_gaussians():
    params, static = make_scene(n=0)
    img = render(params, static)
    np.testing.assert_allclose(np.asarray(img), np.broadcast_to(params["background"], img.shape))
    jax.grad(lambda p: render(p, static).sum())(params)  # must not crash


def test_vmap_matches_loop():
    scenes = [make_scene(seed=s)[0] for s in range(3)]
    _, static = make_scene()
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *scenes)

    imgs = jax.jit(jax.vmap(lambda p: render(p, static)))(batched)
    for i, p in enumerate(scenes):
        np.testing.assert_array_equal(np.asarray(imgs[i]), np.asarray(render(p, static)))

    grad_fn = jax.grad(lambda p: jnp.sum(render(p, static) ** 2))
    bgrads = jax.jit(jax.vmap(grad_fn))(batched)
    for i, p in enumerate(scenes):
        g = grad_fn(p)
        for key in g:
            np.testing.assert_allclose(np.asarray(bgrads[key][i]), np.asarray(g[key]), rtol=1e-4, atol=1e-5)


def test_vmap_shared_camera_params():
    """Only some arguments batched (e.g. a batch of colours for one geometry)."""
    params, static = make_scene()
    colors = jnp.stack([params["colors"], 1 - params["colors"]])
    imgs = jax.vmap(lambda c: render(dict(params, colors=c), static))(colors)
    np.testing.assert_array_equal(np.asarray(imgs[1]), np.asarray(render(dict(params, colors=colors[1]), static)))


def test_block_sizes_agree():
    params, static = make_scene()
    ref = np.asarray(render(params, static))
    for bs in (4, 8, 16):
        img = jaxsplat.render(
            params["means3d"], params["scales"], params["quats"], params["colors"],
            params["opacities"], background=params["background"], block_size=bs, **static,
        )
        np.testing.assert_allclose(np.asarray(img), ref, atol=1e-5)


def test_invalid_block_size():
    params, static = make_scene()
    try:
        jaxsplat.render(
            params["means3d"], params["scales"], params["quats"], params["colors"],
            params["opacities"], background=params["background"], block_size=32, **static,
        )
    except ValueError:
        return
    raise AssertionError("block_size=32 should raise ValueError")


if __name__ == "__main__":
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
