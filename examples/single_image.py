import jaxsplat
import jax
import jax.numpy as jnp
import imageio.v3 as iio
import optax
import argparse
import time


def main(
    iterations: int,
    num_points: int,
    lr: float,
    gt_path: str,
    out_img_path: str,
    out_vid_path: str,
):
    gt = jnp.array(iio.imread(gt_path)).astype(jnp.float32)[..., :3] / 255

    key = jax.random.key(0)
    params, coeffs = init(key, num_points, gt.shape[:2])

    optimizer = optax.adam(lr)
    optimizer_state = optimizer.init(params)

    def loss_fn(params):
        output = render_fn(params, coeffs)
        loss = jnp.mean(jnp.square(output - gt))
        return loss

    # @jax.jit
    def train_step(
        params,
        optimizer_state: optax.OptState,
    ):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    log_every = 50
    with iio.imopen(out_vid_path, "w", plugin="pyav") as video:
        video.init_video_stream("h264")

        cum_time = 0
        cum_time_split = 0
        for i in range(iterations):
            img = (render_fn(params, coeffs) * 255).astype(jnp.uint8)
            video.write_frame(img)

            start = time.perf_counter()
            params, optimizer_state, loss = train_step(params, optimizer_state)
            end = time.perf_counter()

            cum_time += end - start
            cum_time_split += end - start

            if i % log_every == 0:
                print(
                    f"iter {i} loss {loss:.4f}, {cum_time_split/log_every*1000:.3f}ms avg per step"
                )
                cum_time_split = 0
        print(
            f"done training in {cum_time:.3f}s ({cum_time/iterations*1000:.3f}ms avg per step)"
        )

    out = render_fn(params, coeffs)
    iio.imwrite(out_img_path, (out * 255).astype(jnp.uint8))


def init(key, num_points, img_shape):
    key, subkey = jax.random.split(key)
    means3d = jax.random.uniform(
        subkey,
        (num_points, 3),
        minval=jnp.array([-6, -6, -1]),
        maxval=jnp.array([6, 6, 1]),
        dtype=jnp.float32,
    )

    key, subkey = jax.random.split(key)
    scales = jax.random.uniform(
        subkey, (num_points, 3), dtype=jnp.float32, minval=0, maxval=0.5
    )

    key, subkey = jax.random.split(key)
    u, v, w = jax.random.uniform(subkey, (3, num_points, 1))
    quats = jnp.hstack(
        [
            jnp.sqrt(1 - u) * jnp.sin(2 * jnp.pi * v),
            jnp.sqrt(1 - u) * jnp.cos(2 * jnp.pi * v),
            jnp.sqrt(u) * jnp.sin(2 * jnp.pi * w),
            jnp.sqrt(u) * jnp.cos(2 * jnp.pi * w),
        ]
    )

    viewmat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    key, subkey = jax.random.split(key)
    colors = jax.random.uniform(subkey, (num_points, 3), dtype=jnp.float32)

    key, subkey = jax.random.split(key)
    opacities = jax.random.uniform(subkey, (num_points, 1), minval=0.5)

    background = jnp.array([0, 0, 0], dtype=jnp.float32)

    H, W = img_shape
    fx, fy = W / 2, H / 2
    cx, cy = W / 2, H / 2
    glob_scale = 1
    clip_thresh = 0.01
    block_size = 16

    return (
        {
            "means3d": means3d,
            "scales": scales,
            "quats": quats,
            "colors": colors,
            "opacities": opacities,
        },
        {
            "viewmat": viewmat,
            "background": background,
            "img_shape": img_shape,
            "f": (fx, fy),
            "c": (cx, cy),
            "glob_scale": glob_scale,
            "clip_thresh": clip_thresh,
            "block_size": block_size,
        },
    )


def render_fn(params, coeffs):
    means3d = params["means3d"]
    quats = params["quats"] / (jnp.linalg.norm(params["quats"], axis=-1, keepdims=True))
    scales = params["scales"]
    colors = jax.nn.sigmoid(params["colors"])
    opacities = jax.nn.sigmoid(params["opacities"])

    img = jaxsplat.render(
        means3d=means3d,
        scales=scales,
        quats=quats,
        colors=colors,
        opacities=opacities,
        viewmat=coeffs["viewmat"],
        background=coeffs["background"],
        img_shape=coeffs["img_shape"],
        f=coeffs["f"],
        c=coeffs["c"],
        glob_scale=coeffs["glob_scale"],
        clip_thresh=coeffs["clip_thresh"],
        block_size=coeffs["block_size"],
    )

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m examples.single_image",
        description="Fits 3D Gaussians to single 2D image",
    )
    parser.add_argument("input")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--num_points", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out_image", default="out.png")
    parser.add_argument("--out_video", default="out.mp4")

    args = parser.parse_args()
    main(
        args.iters, args.num_points, args.lr, args.input, args.out_image, args.out_video
    )
