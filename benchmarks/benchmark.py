"""Speed benchmark: original jaxsplat (JAX 0.5) vs. jaxsplat_ffi (JAX 0.12).

    # environment with JAX 0.5 + original jaxsplat
    PYTHONPATH=src:benchmarks python benchmarks/benchmark.py --impl jaxsplat --out results/bench_legacy.json
    # environment with jaxsplat_ffi
    PYTHONPATH=src:benchmarks python benchmarks/benchmark.py --impl ffi --out results/bench_ffi.json
    # table
    python benchmarks/benchmark.py --report results/bench_legacy.json results/bench_ffi.json

Every case is jitted; times are the median over repeated calls after warm-up,
including the device sync (block_until_ready).
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

import scenes

# (name, scene factory)
CASES = [
    ("N=1k, 512x512", lambda: scenes.random_scene(1_000, (512, 512), seed=10)),
    ("N=10k, 512x512", lambda: scenes.random_scene(10_000, (512, 512), seed=11)),
    ("N=100k, 512x512", lambda: scenes.random_scene(100_000, (512, 512), seed=12, spread=4.0, scale=(0.01, 0.08))),
    ("N=1M, 512x512", lambda: scenes.random_scene(1_000_000, (512, 512), seed=13, spread=4.0, scale=(0.005, 0.03))),
    ("N=100k, 256x256", lambda: scenes.random_scene(100_000, (256, 256), seed=12, spread=4.0, scale=(0.01, 0.08))),
    ("N=100k, 1024x1024", lambda: scenes.random_scene(100_000, (1024, 1024), seed=12, spread=4.0, scale=(0.01, 0.08))),
]


def timeit(fn, *args, min_time=2.0, max_reps=200, warmup=3):
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and (time.perf_counter() - start < min_time or len(times) < 5):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def gpu_name():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def run(impl):
    import jax
    import jax.numpy as jnp

    if impl == "jaxsplat":
        import jaxsplat as renderer
    else:
        import jaxsplat_ffi as renderer

    results = {
        "impl": impl,
        "jax": jax.__version__,
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "cases": {},
    }
    for name, make_scene in CASES:
        scene = make_scene()
        render = scenes.make_render_fn(renderer, scene)
        inputs = [jnp.asarray(scene[k]) for k in scenes.DIFF_KEYS]
        weights = jnp.asarray(scenes.loss_weights(scene["img_shape"]))
        argnums = tuple(range(len(inputs)))

        def loss(*xs):
            return jnp.sum(render(*xs) * weights)

        fwd = render
        fwd_bwd = jax.value_and_grad(loss, argnums=argnums)

        entry = {}
        try:
            entry["forward_ms"] = timeit(jax.jit(fwd), *inputs)
            entry["forward_backward_ms"] = timeit(jax.jit(fwd_bwd), *inputs)
        except Exception as e:  # e.g. out of memory
            entry["error"] = f"{type(e).__name__}: {(str(e).splitlines() or [repr(e)])[0][:200]}"
        results["cases"][name] = entry
        print(f"{name:40s} " + "  ".join(
            f"{k} {v:8.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()
        ), flush=True)
    return results


def report(paths):
    data = [json.loads(Path(p).read_text()) for p in paths]
    base, new = data[0], data[1]
    print(f"GPU: {base['gpu']}\n")
    print(f"| Case | {base['impl']} (JAX {base['jax']}) fwd | {new['impl']} (JAX {new['jax']}) fwd | speed-up "
          f"| {base['impl']} fwd+bwd | {new['impl']} fwd+bwd | speed-up |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in base["cases"]:
        a, b = base["cases"][name], new["cases"].get(name, {})
        row = [name]
        for key in ("forward_ms", "forward_backward_ms"):
            if key in a and key in b:
                row += [f"{a[key]:.2f} ms", f"{b[key]:.2f} ms", f"{a[key] / b[key]:.2f}x"]
            else:
                row += [a.get(key, a.get("error", "-")), b.get(key, b.get("error", "-")), "-"]
        print("| " + " | ".join(str(x) for x in row) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["jaxsplat", "ffi"])
    parser.add_argument("--out")
    parser.add_argument("--report", nargs=2, metavar=("BASELINE_JSON", "NEW_JSON"))
    args = parser.parse_args()
    if args.report:
        report(args.report)
    else:
        res = run(args.impl)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=2))
            print(f"Saved {args.out}")
