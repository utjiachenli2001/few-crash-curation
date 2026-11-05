"""Smoke test of the MuJoCo car scripts, with the logs made smaller.

Needs the MuSHR car model, as the README says (prl-mushr/mushr_mujoco_ros, BSD-3): point MUSHR_CARS at its
models/cars folder, e.g.

  git clone --depth 1 https://github.com/prl-mushr/mushr_mujoco_ros /tmp/mushr
  MUSHR_CARS=/tmp/mushr/models/cars python tests/smoke_mujoco.py

Copies code/mujoco/ and the car into a temporary folder and runs, unchanged, the commands the README gives for
Table 8 and Section 7.8: make_dry_logs.py and mk_matched.py (the logs), mj_major.py, mj_crude.py, mj_out.py and
mj_sign_ana.py on the majority fault, then mj_seeds.py and mj_out.py on the minority fault. Before each script a
short bootstrap makes the work small: a few logs of each kind (60 exploration, 40 on-ice, 3 route and 6 dry-day logs,
and 4 + 8 from mk_matched.py), 3 training epochs, MPPI with 64 rollouts over a 10-step horizon, and evaluation runs of
at most 40 steps from at most 3 starts per set. The few majority-fault logs are saved under the name the scripts read,
logs_V2.5_dry300_m100.npz. The numbers mean nothing; this checks that the code runs and writes rows of the right shape.

usage (from anywhere): MUSHR_CARS=<mushr_mujoco_ros>/models/cars python tests/smoke_mujoco.py [--only mj_out] [--keep DIR]
"""
import json
import os
import shutil
import sys

from smoke_common import ROOT, expect, run

SEED, K = "1", 12

BOOT = r"""
import core as C
import attrlib as A
C.K, C.H = 64, 10                                      # MPPI rollouts (attrlib keeps the top 64) and horizon
_make_logs, _fit, _starts, _run = C.make_logs, C.fit, C.starts, C.run
def make_logs(seed, n_explore=400, n_route=10, n_dry=10, n_ice=0, **kw):   # a few logs of each kind
    return _make_logs(seed, n_explore=min(n_explore, 60), n_route=min(n_route, 3), n_dry=min(n_dry, 6),
                      n_ice=min(n_ice, 40), **kw)
def fit(D, keep_logs=None, seed=0, epochs=60, **kw):   # 3 epochs
    return _fit(D, keep_logs, seed=seed, epochs=min(epochs, 3), **kw)
def starts(n, seed):                                   # at most 3 starts
    return _starts(min(n, 3), seed)
def run(model, S0, seed=0, T=200, **kw):               # evaluation: at most 3 starts per set, 40 steps
    return _run(model, S0[:3], seed=seed, T=min(T, 40), **kw)
C.make_logs, C.fit, C.starts, C.run = make_logs, fit, starts, run
"""


def copy_car(here):
    src = os.environ.get("MUSHR_CARS", "")
    if not os.path.exists(os.path.join(src, "base_car", "buddy.xml")):
        sys.exit("\n\n".join(__doc__.split("\n\n")[1:3]) + "\n\nMUSHR_CARS is not set to a folder with base_car/buddy.xml.")
    shutil.copytree(src, os.path.join(here, "mushr", "cars"))


def majority_name(here):
    """mv logs_V2.5_dry108_m4.npz logs_V2.5_dry300_m100.npz (the name mj_crude.py and mj_sign_ana.py read)"""
    os.replace(os.path.join(here, "logs_V2.5_dry108_m4.npz"), os.path.join(here, "logs_V2.5_dry300_m100.npz"))


def exists(*files):
    def f(here):
        for x in files:
            expect(os.path.exists(os.path.join(here, x)), f"{x} was not written")
        return ""
    return f


def rows(file, *names):
    def f(here):
        d = json.load(open(os.path.join(here, file)))
        r = {x["model"]: x for x in d["rows"]}
        for n in names:
            expect(n in r, f"no row {n!r} in {file}")
            for s in ("test", "late entry", "ice-edge entry"):
                if s in r[n]:
                    expect(0 <= r[n][s]["crash"] <= 1, f"bad crash rate in {n!r}: {r[n][s]}")
        return f"{len(r)} rows" + (f", {d['n_crash']} crash runs" if "n_crash" in d else "")
    return f


MAJ = "logs_V2.5_dry300_m100.npz"
STEPS = [
    ("make_dry_logs", ["make_dry_logs.py", "100"], exists("logs_V2.5_dry100.npz", "m_V2.5_dry100_faulty_s1.pt"), []),
    ("mk_matched", ["mk_matched.py", "4", "8"], exists("logs_V2.5_dry108_m4.npz"), ["make_dry_logs"]),
    ("rename", majority_name, exists(MAJ), ["mk_matched"]),
    ("mj_major", ["mj_major.py", "--logs", MAJ, "--seed", SEED, "--k", str(K)],
     rows(f"major_V2.5_dry300_m100_s{SEED}.json", "faulty", f"ours from 1 runs, remove {K}"), ["rename"]),
    ("mj_crude", ["mj_crude.py", "--seed", SEED, "--k", str(K)],
     rows(f"crude_V2.5_dry300_m100_s{SEED}.json", f"drop {K} most crash-like logs"), ["mj_major"]),
    ("mj_out", ["mj_out.py", "--mode", "major", "--logs", MAJ, "--seed", SEED, "--k", str(K), "--tl", "40"],
     rows(f"out_major_V2.5_dry300_m100_s{SEED}.json", f"output-space from 1 runs, remove {K}"), ["mj_major"]),
    ("mj_sign_ana", ["mj_sign_ana.py", "--seed", SEED], exists(f"signana_V2.5_dry300_m100_s{SEED}.json"),
     ["mj_major"]),
    ("mj_seeds", ["mj_seeds.py", "--logs", "logs_V2.5_dry100.npz", "--seed", SEED, "--rs", "1"],
     rows(f"seeds_V2.5_dry100_s{SEED}.json", "faulty", "add 1 runs, W=20 | retrain 1"), ["make_dry_logs"]),
    ("mj_out", ["mj_out.py", "--mode", "seeds", "--logs", "logs_V2.5_dry100.npz", "--seed", SEED, "--k", "10",
                "--tl", "10"], rows(f"out_seeds_V2.5_dry100_s{SEED}.json", "output-space from 1 runs, remove 10 | retrain 0"),
     ["make_dry_logs"]),
]

if __name__ == "__main__":
    run(os.path.join(ROOT, "code", "mujoco"), ".", copy_car, BOOT, STEPS, doc=__doc__)
