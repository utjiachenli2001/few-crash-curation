"""Check the README's "runs are deterministic on one CPU thread": run the same commands twice, each in a fresh folder
with the same synthetic data, and compare every file the two runs write. JSON and other text must match byte for byte,
NumPy archives array by array, PyTorch models tensor by tensor (a torch.save file also carries a random id).

  drone       prep.py, nat_probe.py prop50j8 1, mis_drone.py (FACT and adding the failure, Table 1)
  crazyflie   build_cf.py, cf_table.py (Table 9)
  mujoco      make_dry_logs.py and mj_major.py on its logs (Table 8), if MUSHR_CARS is set (see smoke_mujoco.py)

With the bootstrap of the smoke tests (small MPPI, few dives, a weak plant) and NT=1, as in the README.

usage (from anywhere): python tests/determinism.py [--only drone,crazyflie,mujoco]
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import smoke_crazyflie as SC
import smoke_drone as SD
import smoke_mujoco as SM
from smoke_common import ROOT, prepare, run_steps

PLATFORMS = {
    "drone": (os.path.join(ROOT, "code", "drone", "work"), "work", SD.synthetic_neurobem, SD.BOOT, {}, [
        SD.PREP,
        ("nat_probe", ["nat_probe.py", "prop50j8", "1"], None, ["prep"]),
        ("mis_drone", ["mis_drone.py", "--fault", "prop50j8", "--seed", "1", "--k", "40"], None, ["nat_probe"]),
    ]),
    "crazyflie": (os.path.join(ROOT, "code", "crazyflie"), ".", SC.synthetic_nanobench, SC.BOOT,
                  {"CF_VOLTS": "3.1", "CF_EQ": "1"}, SC.STEPS[:2]),
    "mujoco": (os.path.join(ROOT, "code", "mujoco"), ".", SM.copy_car, SM.BOOT, {}, [
        SM.STEPS[0],
        ("mj_major", ["mj_major.py", "--logs", "logs_V2.5_dry100.npz", "--seed", "1", "--k", "12"], None,
         ["make_dry_logs"]),
    ]),
}


def same(a, b):
    """None if files a and b hold the same results, else what differs."""
    if a.endswith(".pt"):
        import torch
        x, y = torch.load(a), torch.load(b)
        if x.keys() != y.keys():
            return "different tensors"
        bad = [k for k in x if not torch.equal(x[k], y[k])]
        return f"tensors differ: {', '.join(bad)}" if bad else None
    if a.endswith(".npz"):
        x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        if sorted(x.files) != sorted(y.files):
            return "different arrays"
        bad = [k for k in x.files if not np.array_equal(x[k], y[k])]
        return f"arrays differ: {', '.join(bad)}" if bad else None
    return None if open(a, "rb").read() == open(b, "rb").read() else "bytes differ"


def outputs(folder):
    out = set()
    for d, _, fs in os.walk(folder):
        for f in fs:
            if not f.endswith(".py") and "__pycache__" not in d:
                out.add(os.path.relpath(os.path.join(d, f), folder))
    return out


def check(name, tmp):
    src, sub, make_data, boot, env, steps = PLATFORMS[name]
    if name == "mujoco" and not os.environ.get("MUSHR_CARS"):
        print("  skip mujoco: MUSHR_CARS is not set")
        return []
    runs = []
    for r in ("a", "b"):
        folder = os.path.join(tmp, name, r)
        here = prepare(src, sub, make_data, folder)
        print(f"  run {r}:", flush=True)
        if run_steps(steps, "", folder, here, sub, boot, env):
            return [f"{name}: run {r} failed"]
        runs.append(folder)
    fa, fb = outputs(runs[0]), outputs(runs[1])
    bad = [f"{name}: {f} written by one run only" for f in sorted(fa ^ fb)]
    n_same = 0
    for f in sorted(fa & fb):
        why = same(os.path.join(runs[0], f), os.path.join(runs[1], f))
        if why is None:
            n_same += 1
            if f.endswith((".json", ".pt", ".npz")):  # results; the inputs are the same by construction
                print(f"  same {f}")
        else:
            print(f"  DIFF {f}: {why}")
            bad.append(f"{name}: {f} ({why})")
    print(f"  {n_same} of {len(fa | fb)} files identical")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", default="", help="comma-separated platforms (default: all)")
    names = [x for x in p.parse_args().only.split(",") if x] or list(PLATFORMS)
    t0, bad = time.time(), []
    tmp = tempfile.mkdtemp(prefix="determinism_")
    try:
        for name in names:
            print(f"== {name}", flush=True)
            bad += check(name, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'NOT DETERMINISTIC: ' + '; '.join(bad) if bad else 'two runs wrote identical results'} "
          f"({time.time() - t0:.0f}s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
