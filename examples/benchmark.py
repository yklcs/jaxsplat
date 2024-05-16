import jax
import jax.numpy as jnp
import numpy as np
import torch
import imageio.v3 as iio

import time
import argparse

import gsplat
import jaxsplat
import diff_gaussian_rasterization


def main(
    iterations: int,
    num_points: int,
):
    coeffs_jax = init_coeffs()
    experiments: dict[str, Experiment] = {
        "jaxsplat": JaxsplatExperiment(coeffs_jax),
        "gsplat": GsplatExperiment(coeffs_jax),
        "diff_gaussian_rasterization": DiffGaussianRasterizationExperiment(coeffs_jax),
    }

    for name, experiment in experiments.items():
        key = jax.random.key(0)
        params_jax = init_params(key, num_points)
        _, img = experiment.run(params_jax)

        iio.imwrite(f"test-{name}.png", (img * 255).astype(jnp.uint8))

        for i in range(iterations):
            key, subkey = jax.random.split(key)
            params_jax = init_params(subkey, num_points)
            _, img = experiment.run(params_jax)
        print(f"{name} avg {experiment.total_time/iterations*1000:.4f}ms")


class Experiment:
    _start_time: float
    total_time: float = 0

    def _start(self):
        self._start_time = time.perf_counter()

    def _end(self) -> float:
        end = time.perf_counter()
        return end - self._start_time

    def run(self, params_jax: dict[str, jax.Array]) -> tuple[float, jax.Array]: ...


class JaxsplatExperiment(Experiment):
    _coeffs: dict

    def __init__(self, coeffs_jax):
        self._coeffs = coeffs_jax
        self.render = self.renderer()

    def run(self, params_jax):
        self._start()
        img = self.render(params_jax).block_until_ready()
        delta = self._end()

        self.total_time += delta
        return delta, img

    def renderer(self):
        viewmat = self._coeffs["viewmat"]
        background = self._coeffs["background"]
        img_shape = self._coeffs["img_shape"]
        f = self._coeffs["f"]
        c = self._coeffs["c"]
        glob_scale = self._coeffs["glob_scale"]
        clip_thresh = self._coeffs["clip_thresh"]
        block_size = self._coeffs["block_size"]

        def render(params: dict[str, jax.Array]) -> jax.Array:
            img = jaxsplat.render(
                means3d=params["means3d"],
                scales=params["scales"],
                quats=params["quats"],
                colors=params["colors"],
                opacities=params["opacities"],
                viewmat=viewmat,
                background=background,
                img_shape=img_shape,
                f=f,
                c=c,
                glob_scale=glob_scale,
                clip_thresh=clip_thresh,
                block_size=block_size,
            )
            return img

        return render


class GsplatExperiment(Experiment):
    _coeffs: dict

    def __init__(self, coeffs_jax):
        self._coeffs = jax_to_torch_dict(coeffs_jax)
        self.render = self.renderer()

    def run(self, params_jax):
        params = jax_to_torch_dict(params_jax)
        self._start()
        img = self.render(params)
        torch.cuda.synchronize()
        delta = self._end()

        self.total_time += delta
        return delta, jnp.asarray(img.cpu())

    def renderer(self):
        glob_scale = self._coeffs["glob_scale"]
        viewmat = self._coeffs["viewmat"]
        img_height = self._coeffs["img_shape"][0]
        img_width = self._coeffs["img_shape"][1]
        fx = self._coeffs["f"][0]
        fy = self._coeffs["f"][1]
        cx = self._coeffs["c"][0]
        cy = self._coeffs["c"][1]
        clip_thresh = self._coeffs["clip_thresh"]
        block_width = self._coeffs["block_size"]
        background = self._coeffs["background"]

        def render(params: dict[str, torch.Tensor]) -> torch.Tensor:
            (xys, depths, radii, conics, compensation, num_tiles_hit, cov3ds) = (
                gsplat.project_gaussians(
                    means3d=params["means3d"],
                    scales=params["scales"],
                    glob_scale=glob_scale,
                    quats=params["quats"],
                    viewmat=viewmat,
                    img_height=img_height,
                    img_width=img_width,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    clip_thresh=clip_thresh,
                    block_width=block_width,
                )
            )

            img = gsplat.rasterize_gaussians(
                xys=xys,
                depths=depths,
                radii=radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,
                colors=params["colors"],
                opacity=params["opacities"],
                img_height=img_height,
                img_width=img_width,
                block_width=block_width,
                background=background,
                return_alpha=False,
            )

            return img

        return render


class DiffGaussianRasterizationExperiment(Experiment):
    _coeffs: dict
    _settings: diff_gaussian_rasterization.GaussianRasterizationSettings

    def __init__(self, coeffs_jax):
        self._coeffs = jax_to_torch_dict(coeffs_jax)
        h, w = self._coeffs["img_shape"]
        fx, fy = self._coeffs["f"]
        far, near = 1000, self._coeffs["clip_thresh"]
        viewmat = self._coeffs["viewmat"].T
        projmat = torch.tensor(
            [
                [2 * fx / w, 0, 0, 0],
                [0, 2 * fy / h, 0, 0],
                [0, 0, (far + near) / (far - near), 1],
                [0, 0, -2 * far * near / (far - near), 0],
            ]
        ).cuda()
        self._settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=0.5 * w / fx,
            tanfovy=0.5 * h / fy,
            bg=self._coeffs["background"],
            scale_modifier=self._coeffs["glob_scale"],
            viewmatrix=viewmat,
            projmatrix=viewmat @ projmat,
            sh_degree=0,
            campos=viewmat[:3, 3],
            prefiltered=False,
            debug=False,
        )
        self.render = self.renderer()

    def run(self, params_jax):
        params = jax_to_torch_dict(params_jax)
        self._start()
        img = self.render(params)
        torch.cuda.synchronize()
        delta = self._end()

        self.total_time += delta
        return delta, jnp.asarray(img.permute(1, 2, 0).cpu())

    def renderer(self):
        settings = self._settings

        def render(
            params: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            img, _ = diff_gaussian_rasterization.rasterize_gaussians(
                means3D=params["means3d"],
                means2D=torch.zeros_like(params["means3d"]),
                sh=torch.Tensor([]),
                colors_precomp=params["colors"],
                opacities=params["opacities"],
                scales=params["scales"],
                rotations=params["quats"],
                cov3Ds_precomp=torch.Tensor([]),
                raster_settings=settings,
            )  # type: ignore

            return img

        return render


def jax_to_torch(array_jax: jax.Array) -> torch.Tensor:
    array_np = np.asarray(array_jax)
    tensor = torch.from_numpy(array_np.copy()).cuda()
    return tensor


def jax_to_torch_dict(dict_jax) -> dict:
    return {
        k: (jax_to_torch(v) if isinstance(v, jax.Array) else v)
        for k, v in dict_jax.items()
    }


def init_params(key, num_points: int) -> dict[str, jax.Array]:
    subkeys = jax.random.split(key, 5)

    means3d = jax.random.uniform(subkeys[0], (num_points, 3), minval=-3, maxval=3)
    scales = jax.random.uniform(subkeys[1], (num_points, 3), maxval=0.5)
    quats = jax.random.normal(subkeys[2], (num_points, 4))
    quats /= jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(subkeys[3], (num_points, 3))
    opacities = jax.random.uniform(subkeys[4], (num_points, 1))

    return {
        "means3d": means3d,
        "scales": scales,
        "quats": quats,
        "colors": colors,
        "opacities": opacities,
    }


def init_coeffs():
    viewmat = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 8],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    W, H = 1600, 900

    return {
        "viewmat": viewmat,
        "background": jnp.ones((3,), dtype=jnp.float32),
        "img_shape": (H, W),
        "f": (W / 2, H / 2),
        "c": (W / 2, H / 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
        "block_size": 16,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m examples.benchmark",
        description="Benchmarks jaxsplat and other methods",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num_points", type=int, default=50_000)

    args = parser.parse_args()
    main(args.iters, args.num_points)
