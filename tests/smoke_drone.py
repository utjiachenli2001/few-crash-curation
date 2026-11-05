"""Smoke test of the quadrotor scripts on synthetic data, without NeuroBEM.

Writes small synthetic flights in NeuroBEM's processed format (processed_data/**/merged_<flight>_seg_<n>.csv at 400 Hz,
Flights.txt and testset.txt; 48 flights on 4 days, a quadratic thrust map plus drag), copies code/drone/work/ into a
temporary folder and runs the real scripts there, unchanged, in the order the README gives them:

  prep.py                      the flights -> work/nb100.npz
  nat_probe.py                 faulty and oracle models
  mis_drone.py                 FACT (top-k, sign rule, worst 2 s) and adding the failure (Table 1)
  sign3add_drone.py            sign rule from 3 dives, then add them
  base_drone.py --smoke        robust losses, worst-fit and crash-like removal, ensembles
  controls_drone.py --smoke    learning to reweight and the other controls (Table 3), 20 updates each
  reach_drone.py --smoke       every --model, including the gradient-free score and L2RW (Table 2)

Before each script a short bootstrap makes the work small: MPPI samples 64 rollouts over a 10-step horizon, every
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

FAULT, SEED, K = "prop50j8", "1", 40  # as in the README: the propeller fault on 8 of the 12 flights of each day
N_FLIGHTS, N_BAD = 48, 32

PATCH = """
import drone as Dr
import attr_data as AD
Dr.K, Dr.H = 64, 10                                    # MPPI rollouts (cupid_drone.py keeps the top 64) and horizon
_tasks, _evaluate = Dr.tasks, Dr.evaluate
def tasks(n_each=4, *a, **kw):                         # 2 dives per evaluation set
    return _tasks(min(n_each, 2), *a, **kw)
def evaluate(model, *a, **kw):                         # at most 40 steps per evaluation dive
    kw["T"] = min(kw.get("T", 200), 40)
    return _evaluate(model, *a, **kw)
Dr.tasks, Dr.evaluate = tasks, evaluate
AD.PLANT0 = Dr.PLANT = Dr.Tweak(Dr.PLANT, a=-0.6)    # 40% thrust: cannot hover, so every dive crashes
"""
# drone.py loads work/nb100.npz on import, so prep.py, which writes it, runs without the patch
BOOT = 'if os.path.basename(sys.argv[1]) != "prep.py":\n' + "".join("    " + x + "\n" for x in PATCH.strip().split("\n"))


def synthetic_neurobem(here, days=("2021-02-03", "2021-02-05", "2021-02-18", "2021-02-23"), per_day=12, segs=2,
                       rows=440, seed=0):
    """NeuroBEM's processed flights as prep.py reads them, next to work/ (here), from a made-up quadrotor: thrust
    2.45 * sum((mot/1000)^2) along the body z axis, linear drag, and flights that differ in how much axial descent they
    contain. 400 Hz, so prep.py's every-4th-sample keeps `rows` per segment. 12 flights a day are enough for every
    fault in the README (prop50j8m takes the 12 with the most descent, local_drone.py 11)."""
    root = os.path.dirname(here)
    rng = np.random.default_rng(seed)
    cols = ["t", "vel x", "vel y", "vel z", "ang vel x", "ang vel y", "ang vel z", "quat x", "quat y", "quat z",
            "quat w", "mot 1", "mot 2", "mot 3", "mot 4", "acc x", "acc y", "acc z"]
    flights = [f"{d}-{10 + i:02d}-00-00" for d in days for i in range(per_day)]
    t = np.arange(4 * rows) * 0.0025
    test = []
    for f, name in enumerate(flights):
        os.makedirs(os.path.join(root, "processed_data", name[:10]), exist_ok=True)
        descent = np.linspace(-1.8, 0.4, per_day)[f % per_day]  # mean axial velocity: some flights descend a lot
        for s in range(segs):
            ph = rng.uniform(0, 2 * np.pi, 7)
            v = np.stack([np.sin(0.7 * t + ph[0]), np.sin(0.5 * t + ph[1]),
                          descent + 1.2 * np.sin(0.9 * t + ph[2])], 1) + rng.normal(0, 0.05, (len(t), 3))
            w = 0.6 * np.stack([np.sin(1.3 * t + ph[3]), np.sin(1.1 * t + ph[4]), np.sin(0.8 * t + ph[5])], 1)
            q = rng.normal(0, 0.1, (len(t), 4)); q[:, 3] = 1.0
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            mot = 1000 + 150 * np.sin(1.5 * t[:, None] + ph[6] + np.arange(4)) + rng.normal(0, 20, (len(t), 4))
            c = 2.45 * ((mot / 1000) ** 2).sum(1)
            acc = np.stack([-0.35 * v[:, 0], -0.35 * v[:, 1], c - 0.25 * v[:, 2]], 1) + rng.normal(0, 0.2, (len(t), 3))
            np.savetxt(os.path.join(root, "processed_data", name[:10], f"merged_{name}_seg_{s}.csv"),
                       np.concatenate([t[:, None], v, w, q, mot, acc], 1), fmt="%.6g", delimiter=",", header=",".join(cols),
                       comments="")
            if f % per_day == 0 and s == segs - 1:
                test.append(f"{name}_seg_{s}")
    with open(os.path.join(root, "Flights.txt"), "w") as fh:
        fh.writelines(f'"{name}"; % synthetic flight {i}\n' for i, name in enumerate(flights))
    with open(os.path.join(root, "testset.txt"), "w") as fh:
        fh.writelines(x + "\n" for x in test)


def check_prep(here):
    z = np.load(os.path.join(here, "nb100.npz"))
    expect(len(z["flights"]) == N_FLIGHTS and len(np.unique(z["flight"])) == N_FLIGHTS, "wrong flight count")
    expect(len(z["data"]) == N_FLIGHTS * 2 * 440 and z["test"].sum() == 4 * 440, f"wrong rows: {len(z['data'])}")
    return f"{len(z['data'])} rows of {len(z['flights'])} flights"

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
    expect(0 <= r["bad_removed"] <= r["removed"] <= N_FLIGHTS and 0 <= r["auroc"] <= 1, f"bad sign-rule row {r}")
    return (f"{len(rows)} rows; the sign rule removes {r['removed']} of {N_FLIGHTS} flights "
            f"({r['bad_removed']} of the {N_BAD} bad)")


def check_base(here):
    d = json.load(open(os.path.join(here, f"smoke_base_{FAULT}_s{SEED}.json")))
    names = [r["model"] for r in d["rows"]]
    want = ("faulty", "huber loss", "l1 loss", f"remove {K} worst-fit flights", "remove all axial-descent samples",
            "ensemble of 5, mean", "ensemble of 5, disagreement penalty beta=10")
    expect(all(n in names for n in want) and all(rates_ok(r) for r in d["rows"]), f"rows: {names}")
    expect(np.isfinite(d["spread"]["train_median"]), "no ensemble spread")
    return f"{len(names)} rows"


ARGS = ["--fault", FAULT, "--seed", SEED]
PREP = ("prep", ["prep.py"], check_prep, [])
STEPS = [
    PREP,
    ("nat_probe", ["nat_probe.py", FAULT, SEED], check_nat, ["prep"]),
    ("mis_drone", ["mis_drone.py", *ARGS, "--k", str(K)], None, ["nat_probe"]),
    ("sign3add_drone", ["sign3add_drone.py", *ARGS], check_mis, ["mis_drone"]),
    ("base_drone", ["base_drone.py", *ARGS, "--k", str(K), "--smoke"], check_base, ["nat_probe"]),
    ("controls_drone", ["controls_drone.py", *ARGS, "--smoke"], None, ["nat_probe"]),
] + [("reach_drone", ["reach_drone.py", *ARGS, "--model", m, "--smoke"], None, ["nat_probe"])
     for m in ("unfixed", "oracle", "add", "fact", "fact_matched", "gradfree", "l2rw", "l2rw_la")]

if __name__ == "__main__":
    run(os.path.join(ROOT, "code", "drone", "work"), "work", synthetic_neurobem, BOOT, STEPS, doc=__doc__)
