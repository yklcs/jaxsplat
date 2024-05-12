import jaxsplat
import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio
import gsplat
import torch


def params():
    num_points = 50000
    key = jax.random.key(2)

    key, subkey = jax.random.split(key)
    means3d = jax.random.normal(subkey, (num_points, 3))

    key, subkey = jax.random.split(key)
    scales = jax.random.uniform(subkey, (num_points, 3)) + 0.1

    key, subkey = jax.random.split(key)
    quats = jax.random.normal(subkey, (num_points, 4))
    quats /= jnp.linalg.norm(quats, axis=-1, keepdims=True)

    viewmat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    key, subkey = jax.random.split(key)
    colors = jax.random.uniform(subkey, (num_points, 3))

    key, subkey = jax.random.split(key)
    opacities = jax.random.uniform(subkey, (num_points, 1))

    background = jnp.array([1, 1, 1], dtype=jnp.float32)

    W, H = 1600, 900
    fx, fy = W / 2, W / 2
    cx, cy = W / 2, H / 2
    glob_scale = 0.1
    clip_thresh = 0.01
    block_size = 16

    return (
        means3d,
        scales,
        quats,
        viewmat,
        colors,
        opacities,
        background,
        (W, H),
        (fx, fy),
        (cx, cy),
        glob_scale,
        clip_thresh,
        block_size,
    )


def params_to_torch(
    means3d,
    scales,
    quats,
    viewmat,
    colors,
    opacities,
    background,
    img_shape,
    f,
    c,
    glob_scale,
    clip_thresh,
    block_size,
):
    return (
        torch.from_numpy(np.asarray(means3d)).cuda().detach().clone().requires_grad_(),
        torch.from_numpy(np.asarray(scales)).cuda().detach().clone().requires_grad_(),
        torch.from_numpy(np.asarray(quats)).cuda().detach().clone().requires_grad_(),
        torch.from_numpy(np.asarray(viewmat)).cuda().detach().clone(),
        torch.from_numpy(np.asarray(colors)).cuda().detach().clone().requires_grad_(),
        torch.from_numpy(np.asarray(opacities))
        .cuda()
        .detach()
        .clone()
        .requires_grad_(),
        torch.from_numpy(np.asarray(background))
        .cuda()
        .detach()
        .clone()
        .requires_grad_(),
        img_shape,
        f,
        c,
        glob_scale,
        clip_thresh,
        block_size,
    )


def test_jax(
    means3d,
    scales,
    quats,
    viewmat,
    colors,
    opacities,
    background,
    img_shape,
    f,
    c,
    glob_scale,
    clip_thresh,
    block_size,
):
    print("\n\n\njaxsplat")

    def render(means3d, scales, quats, viewmat, colors, opacities, background):
        (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit) = jaxsplat.project(
            means3d,
            scales,
            quats,
            viewmat,
            img_shape=img_shape,
            f=f,
            c=c,
            glob_scale=glob_scale,
            clip_thresh=clip_thresh,
            block_width=block_size,
        )

        img = jaxsplat.rasterize(
            colors,
            opacities,
            background,
            xys,
            depths,
            radii,
            conics,
            cum_tiles_hit,
            img_shape=img_shape,
            block_width=block_size,
        )

        return img

    grad = jax.grad(lambda *x: render(*x).min(), argnums=(0, 1, 2, 4, 5))
    # print(
    grad(
        means3d,
        scales,
        quats,
        viewmat,
        colors,
        opacities,
        background,
    )
    # )
    # img = render(means3d, scales, quats, viewmat, colors, opacities, background)
    # iio.imwrite("jaxsplat.png", (img * 255).astype(jnp.uint8))


def test_torch(
    means3d,
    scales,
    quats,
    viewmat,
    colors,
    opacities,
    background,
    img_shape,
    f,
    c,
    glob_scale,
    clip_thresh,
    block_size,
):
    print("\n\n\ngsplat")

    def render(means3d, scales, quats, viewmat, colors, opacities, background):
        (xys, depths, radii, conics, compensation, num_tiles_hit, covs3d) = (
            gsplat.project_gaussians(
                means3d,
                scales,
                glob_scale,
                quats,
                viewmat,
                fx=f[0],
                fy=f[1],
                cx=c[0],
                cy=c[1],
                img_width=img_shape[0],
                img_height=img_shape[1],
                clip_thresh=clip_thresh,
                block_width=block_size,
            )
        )

        img = gsplat.rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            opacities,
            img_width=img_shape[0],
            img_height=img_shape[1],
            block_width=block_size,
        )

        return img

    min = render(
        means3d,
        scales,
        quats,
        viewmat,
        colors,
        opacities,
        background,
    ).min()

    min.backward()

    # img = render(means3d, scales, quats, viewmat, colors, opacities, background)
    # iio.imwrite("gsplat.png", (img * 255).cpu().type(torch.uint8))


if __name__ == "__main__":
    p = params()
    test_jax(*p)
    test_torch(*params_to_torch(*p))
