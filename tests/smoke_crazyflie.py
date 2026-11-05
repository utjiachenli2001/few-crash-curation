"""Smoke test of the Crazyflie scripts on synthetic data, without NanoBench.

Writes a small synthetic stand-in for cf100.npz (24 flights with the columns build_cf.py writes; thrust per PWM grows
with the battery voltage, and the flights' median voltages span 3.0-4.0 V, so most are above the 3.35 V threshold and
are the "bad" logs for a 3.1 V deployment), copies code/crazyflie/ into a temporary folder and runs, unchanged and with
CF_VOLTS=3.1 CF_EQ=1 as in the README:

  cf_table.py     the Table 9 rows: baselines, adding the crash dives, FACT, the oracle with the dives, L2RW
  cf_extra.py     other roll-ups, random removal
  cf_outfly.py    the gradient-free (output-space) score

Before each script a short bootstrap makes the work small: MPPI samples 32 rollouts over a 10-step horizon, every
evaluation set has 3 dives of at most 40 steps, and the plant keeps 20% of its thrust, so the drone cannot hover and
every dive crashes. The scripts themselves are not changed. The numbers mean nothing.

usage (from anywhere): python tests/smoke_crazyflie.py [--only cf_extra] [--keep DIR]
"""
import json
import os

import numpy as np

from smoke_common import ROOT, expect, rates_ok, run

SEED, K = "1", 6

BOOT = r"""
import torch
import cf
cf.K, cf.H = 32, 10                                    # MPPI rollouts and horizon
_tasks, _evaluate = cf.tasks, cf.evaluate
def tasks(n_each=4, *a, **kw):                         # 3 dives per evaluation set
    return _tasks(min(n_each, 3), *a, **kw)
def evaluate(model, *a, **kw):                         # at most 40 steps per evaluation dive
    kw["T"] = min(kw.get("T", 200), 40)
    return _evaluate(model, *a, **kw)
cf.tasks, cf.evaluate = tasks, evaluate
class Weak(torch.nn.Module):                           # 20% of the plant's thrust: cannot hover, every dive crashes
    def __init__(self, plant, keep):
        super().__init__(); self.plant, self.keep = plant, keep
    def forward(self, f):
        r = self.plant(f) * cf.RS
        c = f[..., 6] * cf.SD[6] + cf.MU[6]
        return torch.cat([r[..., :2], r[..., 2:] - (1 - self.keep) * c[..., None]], -1) / cf.RS
cf.PLANT = Weak(cf.PLANT, 0.2)
"""


def synthetic_cf100(folder, n_flights=24, rows=340, seed=0):
    """cf100.npz with the arrays build_cf.py writes, from a made-up Crazyflie: body force 6.8 * (vbat / 3.7) *
    sum(pwm^2) along the body z axis plus linear drag; each flight's battery sags by 0.2 V around its median."""
    rng = np.random.default_rng(seed)
    cols = ["vel x", "vel y", "vel z", "ang vel x", "ang vel y", "ang vel z", "quat x", "quat y", "quat z", "quat w",
            "pwm 1", "pwm 2", "pwm 3", "pwm 4", "acc x", "acc y", "acc z", "vbat", "pos x", "pos y", "pos z"]
    v_med = rng.permutation(np.linspace(3.0, 4.0, n_flights))
    t = np.arange(rows) * 0.01
    data, flight, seg = [], [], []
    for f in range(n_flights):
        ph = rng.uniform(0, 2 * np.pi, 7)
        v = 0.5 * np.stack([np.sin(0.7 * t + ph[0]), np.sin(0.5 * t + ph[1]), np.sin(0.9 * t + ph[2])], 1)
        w = 0.4 * np.stack([np.sin(1.3 * t + ph[3]), np.sin(1.1 * t + ph[4]), np.sin(0.8 * t + ph[5])], 1)
        q = rng.normal(0, 0.05, (rows, 4)); q[:, 3] = 1.0
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        pwm = 0.6 + 0.12 * np.sin(1.5 * t[:, None] + ph[6] + np.arange(4)) + rng.normal(0, 0.01, (rows, 4))
        vbat = v_med[f] + np.linspace(0.1, -0.1, rows)
        c = 6.8 * vbat / 3.7 * (pwm ** 2).sum(1)
        acc = np.stack([-0.35 * v[:, 0], -0.35 * v[:, 1], c - 0.25 * v[:, 2]], 1) + rng.normal(0, 0.2, (rows, 3))
        pos = np.c_[np.zeros(rows), np.zeros(rows), np.full(rows, 1.0)]
        data.append(np.c_[v, w, q, pwm, acc, vbat, pos])
        flight.append(np.full(rows, f)); seg.append(np.full(rows, f * 1000 + 1))
    flight = np.concatenate(flight)
    np.savez(os.path.join(folder, "cf100.npz"), data=np.concatenate(data), cols=np.array(cols), flight=flight,
             seg=np.concatenate(seg), flights=np.array([f"synthetic_{f:02d}" for f in range(n_flights)]),
             test=np.isin(flight, [0, 1]))


def rows(here):
    d = json.load(open(os.path.join(here, f"cf_3.1_eq_s{SEED}.json")))
    expect(d["n_crash_dives"] >= 1, "no crash dives to curate from")
    return {r["model"]: r for r in d["rows"]}


def check(names):
    def f(here):
        r = rows(here)
        for n in names:
            expect(n in r, f"no row {n!r}")
            expect(rates_ok(r[n]), f"bad crash/reach rates in row {n!r}: {r[n]}")
        return f"{len(r)} rows in the table file"
    return f


TABLE = ["faulty (all flights)", "oracle (flights below 3.35 V)", "huber loss", "l1 loss", f"remove {K} worst-fit flights",
         f"remove {K} most crash-like flights (most descent)", "add 1 crash dives, W=20", "add 3 crash dives, W=20",
         "ours from 1 dives, sign rule", f"ours from 3 dives, remove {K}", "ours from 3 dives, sign rule, and add them (W=20)",
         "oracle, and add 3 crash dives (W=20)", "learning to reweight, 1 crash dives as clean set",
         "learning to reweight, 3 crash dives as clean set, and add them (W=20)"]
EXTRA = [f"ours from 1 dives (worst 2 s), remove {K}", f"remove {K} random flights (draw 3)"]
OUTFLY = ["output-space from 1 dives, sign rule", "output-space from 3 dives, sign rule, and add them (W=20)"]
ARGS = ["--seed", SEED, "--k", str(K)]
STEPS = [
    ("cf_table", ["cf_table.py", *ARGS], check(TABLE), []),
    ("cf_extra", ["cf_extra.py", *ARGS], check(EXTRA), ["cf_table"]),
    ("cf_outfly", ["cf_outfly.py", *ARGS], check(OUTFLY), ["cf_table"]),
]

if __name__ == "__main__":
    run(os.path.join(ROOT, "code", "crazyflie"), ".", synthetic_cf100, BOOT, STEPS, {"CF_VOLTS": "3.1", "CF_EQ": "1"},
        doc=__doc__)
