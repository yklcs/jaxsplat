"""Compares two dump_outputs.py results (needs only numpy).

    python benchmarks/compare_outputs.py results/legacy.npz results/ffi.npz

For gradients it reports the relative error ||a - b|| / ||a|| next to the
reference implementation's own run-to-run noise. The check passes if images
match to 1e-5 and every gradient error is within 1e-4 or within 10x the
reference noise (atomicAdd makes float sums order-dependent).
"""

import sys

import numpy as np

IMG_ATOL = 1e-5
GRAD_RTOL = 1e-4
NOISE_FACTOR = 10.0


def rel(a, b):
    na = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / na) if na > 0 else float(np.linalg.norm(b))


def main(ref_path, new_path):
    ref = np.load(ref_path)
    new = np.load(new_path)
    print(f"reference: {ref['meta/impl']} (JAX {ref['meta/jax_version']})")
    print(f"candidate: {new['meta/impl']} (JAX {new['meta/jax_version']})\n")

    scenes = sorted({k.split("/")[0] for k in ref.files if not k.startswith("meta")})
    ok = True
    print(f"{'scene':16s} {'quantity':16s} {'max |diff|':>11s} {'rel err':>10s} {'ref noise':>10s}  status")
    for s in scenes:
        a, b = ref[f"{s}/img"], new[f"{s}/img"]
        d = float(np.abs(a - b).max())
        passed = d <= IMG_ATOL
        ok &= passed
        print(f"{s:16s} {'image':16s} {d:11.3g} {rel(a, b):10.3g} {'':>10s}  {'ok' if passed else 'FAIL'}")
        for key in sorted(k.split("/")[-1] for k in ref.files if k.startswith(f"{s}/run0/")):
            a0, a1 = ref[f"{s}/run0/{key}"], ref[f"{s}/run1/{key}"]
            b0 = new[f"{s}/run0/{key}"]
            noise = rel(a0, a1)
            err = rel(a0, b0)
            passed = bool(np.all(np.isfinite(b0))) and (err <= GRAD_RTOL or err <= NOISE_FACTOR * noise)
            ok &= passed
            print(f"{s:16s} {key:16s} {float(np.abs(a0 - b0).max()):11.3g} {err:10.3g} {noise:10.3g}  "
                  f"{'ok' if passed else 'FAIL'}")
    print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
