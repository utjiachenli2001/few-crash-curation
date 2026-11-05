"""Smoke test of the quadrotor scripts on synthetic data, without NeuroBEM.

Writes a small synthetic stand-in for work/nb100.npz (24 flights on 4 days, the NeuroBEM columns the code reads, a
quadratic thrust map plus drag), copies code/drone/work/ into a temporary folder and runs the real experiment scripts
there, unchanged, in the order the README gives them:

  nat_probe.py                 faulty and oracle models
  mis_drone.py                 FACT (top-k, sign rule, worst 2 s) and adding the failure (Table 1)
  sign3add_drone.py            sign rule from 3 dives, then add them
  base_drone.py --smoke        robust losses, worst-fit and crash-like removal, ensembles
  controls_drone.py --smoke    learning to reweight and the other controls (Table 3), 20 updates each
  reach_drone.py --smoke       every --model, including the gradient-free score and L2RW (Table 2)

Before each script a short bootstrap makes the work small: MPPI samples 32 rollouts over a 10-step horizon, every
evaluation set has 2 dives of at most 40 steps, and the plant keeps 40% of its thrust, so the drone cannot hover,
every dive crashes and there is always a failure to curate. The scripts themselves are not changed.

This checks that the pipeline runs end to end with the installed packages (torch.func, scikit-learn, NumPy) and writes
results of the expected shape. The numbers mean nothing.

usage (from anywhere): python tests/smoke_drone.py [--only mis_drone,reach_drone] [--keep DIR]
"""
import json
import os

import numpy as np

from smoke_common import ROOT, expect, rates_ok, run

FAULT, SEED, K = "prop50j2", "1", 4  # propeller fault on the 2 flights per day with the most axial descent (8 of 24)

BOOT = r"""
import drone as Dr
import attr_data as AD
Dr.K, Dr.H = 32, 10                                    # MPPI rollouts and horizon
_tasks, _evaluate = Dr.tasks, Dr.evaluate
def tasks(n_each=4, *a, **kw):                         # 2 dives per evaluation set
    return _tasks(min(n_each, 2), *a, **kw)
def evaluate(model, *a, **kw):                         # at most 40 steps per evaluation dive
    kw["T"] = min(kw.get("T", 200), 40)
    return _evaluate(model, *a, **kw)
Dr.tasks, Dr.evaluate = tasks, evaluate
AD.PLANT0 = Dr.PLANT = Dr.Tweak(Dr.PLANT, a=-0.6)    # 40% thrust: cannot hover, so every dive crashes
"""


def synthetic_nb100(folder, days=("2021-02-03", "2021-02-05", "2021-02-18", "2021-02-23"), per_day=6, segs=2,
                    rows=440, seed=0):
    """nb100.npz with the arrays prep.py writes, from a made-up quadrotor: thrust 2.45 * sum((mot/1000)^2) along the
    body z axis, linear drag, and flights that differ in how much axial descent they contain."""
    rng = np.random.default_rng(seed)
    cols = ["vel x", "vel y", "vel z", "ang vel x", "ang vel y", "ang vel z", "quat x", "quat y", "quat z", "quat w",
            "mot 1", "mot 2", "mot 3", "mot 4", "acc x", "acc y", "acc z"]
    flights = [f"{d}-{10 + i:02d}-00-00" for d in days for i in range(per_day)]
    data, flight, seg, test, names = [], [], [], [], []
    t = np.arange(rows) * 0.01
    for f, name in enumerate(flights):
        descent = np.linspace(-1.8, 0.4, per_day)[f % per_day]  # mean axial velocity: some flights descend a lot
        for s in range(segs):
            ph = rng.uniform(0, 2 * np.pi, 7)
            v = np.stack([np.sin(0.7 * t + ph[0]), np.sin(0.5 * t + ph[1]),
                          descent + 1.2 * np.sin(0.9 * t + ph[2])], 1) + rng.normal(0, 0.05, (rows, 3))
            w = 0.6 * np.stack([np.sin(1.3 * t + ph[3]), np.sin(1.1 * t + ph[4]), np.sin(0.8 * t + ph[5])], 1)
            q = rng.normal(0, 0.1, (rows, 4)); q[:, 3] = 1.0
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            mot = 1000 + 150 * np.sin(1.5 * t[:, None] + ph[6] + np.arange(4)) + rng.normal(0, 20, (rows, 4))
            c = 2.45 * ((mot / 1000) ** 2).sum(1)
            acc = np.stack([-0.35 * v[:, 0], -0.35 * v[:, 1], c - 0.25 * v[:, 2]], 1) + rng.normal(0, 0.2, (rows, 3))
            data.append(np.concatenate([v, w, q, mot, acc], 1).astype(np.float32))
            flight.append(np.full(rows, f)); seg.append(np.full(rows, len(names)))
            test.append(np.full(rows, f % per_day == 0 and s == segs - 1))
            names.append(f"{name}_seg_{s}")
    np.savez_compressed(os.path.join(folder, "nb100.npz"), data=np.concatenate(data), flight=np.concatenate(flight),
                        seg=np.concatenate(seg), test=np.concatenate(test), names=np.array(names),
                        flights=np.array(flights), desc=np.array([f"synthetic flight {i}" for i in range(len(flights))]),
                        cols=np.array(cols))


def check_nat(here):
    for f in (f"model3_{FAULT}_s{SEED}.pt", f"model3_{FAULT}_dropharm_s{SEED}.pt", f"nat_{FAULT}_s{SEED}.json"):
        expect(os.path.exists(os.path.join(here, f)), f"nat_probe.py did not write {f}")
    return "faulty and oracle models saved"


def check_mis(here):
    d = json.load(open(os.path.join(here, f"mis_{FAULT}_s{SEED}.json")))
    rows = {r["model"]: r for r in d["rows"]}
    expect(d["n_crash_dives"] >= 1, "no crash dives to curate from")
    for name in ("faulty", "clean (bad flights removed)", "add 1 crash dives, W=20", "ours (sign rule) from 1 dives",
                 f"ours from 1 dives, remove {K}", f"ours (worst 2 s) from 1 dives, remove {K}",
                 "ours (sign rule) from 1 dives, and add them (W=20)", "ours (sign rule) from 3 dives, and add them (W=20)"):
        expect(name in rows, f"no row {name!r}")
        expect(rates_ok(rows[name]), f"bad crash/reach rates in row {name!r}: {rows[name]}")
    r = rows["ours (sign rule) from 1 dives"]
    expect(0 <= r["bad_removed"] <= r["removed"] <= 24 and 0 <= r["auroc"] <= 1, f"bad sign-rule row {r}")
    return f"{len(rows)} rows; the sign rule removes {r['removed']} of 24 flights ({r['bad_removed']} of 8 bad)"


def check_base(here):
    d = json.load(open(os.path.join(here, f"smoke_base_{FAULT}_s{SEED}.json")))
    names = [r["model"] for r in d["rows"]]
    want = ("faulty", "huber loss", "l1 loss", f"remove {K} worst-fit flights", "remove all axial-descent samples",
            "ensemble of 5, mean", "ensemble of 5, disagreement penalty beta=10")
    expect(all(n in names for n in want) and all(rates_ok(r) for r in d["rows"]), f"rows: {names}")
    expect(np.isfinite(d["spread"]["train_median"]), "no ensemble spread")
    return f"{len(names)} rows"


ARGS = ["--fault", FAULT, "--seed", SEED]
STEPS = [
    ("nat_probe", ["nat_probe.py", FAULT, SEED], check_nat, []),
    ("mis_drone", ["mis_drone.py", *ARGS, "--k", str(K)], None, ["nat_probe"]),
    ("sign3add_drone", ["sign3add_drone.py", *ARGS], check_mis, ["mis_drone"]),
    ("base_drone", ["base_drone.py", *ARGS, "--k", str(K), "--smoke"], check_base, ["nat_probe"]),
    ("controls_drone", ["controls_drone.py", *ARGS, "--smoke"], None, ["nat_probe"]),
] + [("reach_drone", ["reach_drone.py", *ARGS, "--model", m, "--smoke"], None, ["nat_probe"])
     for m in ("unfixed", "oracle", "add", "fact", "fact_matched", "gradfree", "l2rw", "l2rw_la")]

if __name__ == "__main__":
    run(os.path.join(ROOT, "code", "drone", "work"), "work", synthetic_nb100, BOOT, STEPS, doc=__doc__)
