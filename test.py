import jaxsplat
import jax
import jax.numpy as jnp


def test():
    num_points = 10

    key = jax.random.key(2)

    key, subkey = jax.random.split(key)
    means3d = jax.random.normal(subkey, (num_points, 3))

    key, subkey = jax.random.split(key)
    scales = jax.random.uniform(subkey, (num_points, 3)) + 0.2

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

    glob_scale = 0.1
    H, W = 512, 512
    cx, cy = W / 2, H / 2
    fx, fy = W / 2, W / 2
    clip_thresh = 0.01

    block_size = 16

    def test_grad_jax(means3d, scales, quats, colors, opacities, background):
        rendered = jaxsplat.render(
            means3d,
            scales,
            quats,
            viewmat,
            colors,
            opacities,
            background,
            glob_scale=glob_scale,
            f=(fx, fy),
            c=(cx, cy),
            img_shape=(W, H),
            block_width=block_size,
            clip_thresh=clip_thresh,
        )

        return rendered.mean()

    grad = jax.grad(test_grad_jax, argnums=(0,))
    print(grad(means3d, scales, quats, colors, opacities, background))


if __name__ == "__main__":
    test()
