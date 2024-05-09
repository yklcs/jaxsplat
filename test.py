import jaxsplat
import jax
import jax.numpy as jnp
import numpy as np
import gsplat
import torch


def test_project_gaussians_forward():
    num_points = 10

    key = jax.random.key(2)

    key, subkey = jax.random.split(key)
    means3d = jax.random.normal(subkey, (num_points, 3))

    key, subkey = jax.random.split(key)
    scales = jax.random.uniform(subkey, (num_points, 3)) + 0.2

    key, subkey = jax.random.split(key)
    quats = jax.random.normal(subkey, (num_points, 4))

    quats /= jnp.linalg.norm(quats, axis=-1, keepdims=True)

    glob_scale = 0.1
    H, W = 512, 512
    cx, cy = W / 2, H / 2
    # 90 degree FOV
    fx, fy = W / 2, W / 2
    clip_thresh = 0.01
    viewmat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # viewmat[:3, :3] = quat_to_rotmat(torch.randn(4))
    BLOCK_SIZE = 16

    # (
    #     covs3d,
    #     xys,
    #     depths,
    #     radii,
    #     conics,
    #     compensation,
    #     num_tiles_hit,
    # ) = jaxsplat.project_gaussians(
    #     means3d,
    #     scales,
    #     quats,
    #     viewmat,
    #     glob_scale=glob_scale,
    #     f=(fx, fy),
    #     c=(cx, cy),
    #     img_shape=(W, H),
    #     block_width=BLOCK_SIZE,
    #     clip_thresh=clip_thresh,
    # )

    # print(xys)

    def test_grad_jax(
        means3d,
        scales,
        quats,
        viewmat,
    ):
        (
            covs3d,
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = jaxsplat.project_gaussians(
            means3d,
            scales,
            quats,
            viewmat,
            glob_scale=glob_scale,
            f=(fx, fy),
            c=(cx, cy),
            img_shape=(W, H),
            block_width=BLOCK_SIZE,
            clip_thresh=clip_thresh,
        )

        return xys.mean()

    grad = jax.grad(test_grad_jax, argnums=(0,))
    print(grad(means3d, scales, quats, viewmat))

    # torch

    means3d_torch = torch.from_numpy(np.array(means3d)).cuda().requires_grad_(True)
    scales_torch = torch.from_numpy(np.array(scales)).cuda().requires_grad_(True)
    quats_torch = torch.from_numpy(np.array(quats)).cuda().requires_grad_(True)
    viewmat_torch = torch.from_numpy(np.array(viewmat)).cuda()

    (
        xys,
        depths,
        radii,  # no grad
        conics,
        compensation,
        num_tiles_hit,  # no grad
        cov3d,
    ) = gsplat.project_gaussians(
        means3d_torch,
        scales_torch,
        glob_scale,
        quats_torch,
        viewmat_torch,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_width=W,
        img_height=H,
        block_width=BLOCK_SIZE,
        clip_thresh=clip_thresh,
    )

    z = xys.mean()
    z.backward()
    print(means3d_torch.grad)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    test_project_gaussians_forward()
