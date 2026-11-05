"""Smoke test of the Crazyflie scripts on synthetic data, without NanoBench.

Writes small synthetic flights in NanoBench's format (one CSV per flight with the columns build_cf.py reads; 24
flights whose thrust per PWM grows with the battery voltage, and whose median voltages span 3.0-4.0 V, so most are above
the 3.35 V threshold and are the "bad" logs for a 3.1 V deployment), copies code/crazyflie/ into a temporary folder and
runs, unchanged and with CF_VOLTS=3.1 CF_EQ=1 as in the README:

  build_cf.py     the flights -> cf100.npz
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

PATCH = """
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
# cf.py loads cf100.npz on import, so build_cf.py, which writes it, runs without the patch
BOOT = 'if os.path.basename(sys.argv[1]) != "build_cf.py":\n' + "".join("    " + x + "\n" for x in PATCH.strip().split("\n"))
N_FLIGHTS, ROWS = 24, 340
COLS = ["qx", "qy", "qz", "qw", "imu_acc_x", "imu_acc_y", "imu_acc_z", "vx", "vy", "vz", "wx_vicon", "wy_vicon",
        "wz_vicon", "motor_motor_m1", "motor_motor_m2", "motor_motor_m3", "motor_motor_m4", "pwr_pm_vbat", "px", "py",
        "pz"]


def synthetic_nanobench(here, seed=0):
    """NanoBench flights as build_cf.py reads them (nanobench/<flight>.csv), from a made-up Crazyflie: body force
    6.8 * (vbat / 3.7) * sum(pwm^2) along the body z axis plus linear drag (the IMU logs it in g); each flight's battery
    sags by 0.2 V around its median. Near-level attitude, airborne throughout."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(here, "nanobench"))
    v_med = rng.permutation(np.linspace(3.0, 4.0, N_FLIGHTS))
    t = np.arange(ROWS) * 0.01
    for f in range(N_FLIGHTS):
        ph = rng.uniform(0, 2 * np.pi, 7)
        v = 0.5 * np.stack([np.sin(0.7 * t + ph[0]), np.sin(0.5 * t + ph[1]), np.sin(0.9 * t + ph[2])], 1)
        w = 0.4 * np.stack([np.sin(1.3 * t + ph[3]), np.sin(1.1 * t + ph[4]), np.sin(0.8 * t + ph[5])], 1)
        q = rng.normal(0, 0.05, (ROWS, 4)); q[:, 3] = 1.0
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        pwm = 0.6 + 0.12 * np.sin(1.5 * t[:, None] + ph[6] + np.arange(4)) + rng.normal(0, 0.01, (ROWS, 4))
        vbat = v_med[f] + np.linspace(0.1, -0.1, ROWS)
        c = 6.8 * vbat / 3.7 * (pwm ** 2).sum(1)
        acc = np.stack([-0.35 * v[:, 0], -0.35 * v[:, 1], c - 0.25 * v[:, 2]], 1) + rng.normal(0, 0.2, (ROWS, 3))
        pos = np.c_[0.3 * np.sin(0.4 * t), 0.3 * np.cos(0.4 * t), 1.0 + 0.2 * np.sin(0.6 * t)]
        np.savetxt(os.path.join(here, "nanobench", f"flight_{f:02d}.csv"),
                   np.c_[q, acc / 9.81, v, w, pwm * 65535, vbat, pos], fmt="%.6g", delimiter=",",
                   header=",".join(COLS), comments="")


def check_build(here):
    z = np.load(os.path.join(here, "cf100.npz"))
    expect(len(z["flights"]) == N_FLIGHTS and len(z["data"]) == N_FLIGHTS * ROWS, f"{len(z['data'])} rows")
    v = z["data"][:, list(z["cols"]).index("vbat")]
    expect(2.85 < v.min() < 2.95 and 4.05 < v.max() < 4.15, f"voltages not carried over: {v.min():.2f}-{v.max():.2f}")
    return f"{len(z['data'])} rows of {len(z['flights'])} flights"


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
    ("build_cf", ["build_cf.py", "nanobench"], check_build, []),
    ("cf_table", ["cf_table.py", *ARGS], check(TABLE), ["build_cf"]),
    ("cf_extra", ["cf_extra.py", *ARGS], check(EXTRA), ["cf_table"]),
    ("cf_outfly", ["cf_outfly.py", *ARGS], check(OUTFLY), ["cf_table"]),
]

if __name__ == "__main__":
    run(os.path.join(ROOT, "code", "crazyflie"), ".", synthetic_nanobench, BOOT, STEPS,
        {"CF_VOLTS": "3.1", "CF_EQ": "1"}, doc=__doc__)
