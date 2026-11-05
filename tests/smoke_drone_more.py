"""Smoke test of the other quadrotor scripts in the README's table, on the synthetic flights of smoke_drone.py.

Runs, unchanged and with the README's arguments, the scripts behind Tables 2-7, Fig. 3 and Sections 4.2-4.3,
7.2-7.4 and 7.6-7.7 that smoke_drone.py does not: cupid_drone, sup_drone, ksens_drone, ckpt_drone, agg_check,
margin_drone (on a benchmark and on a sweep setting), sweep_drone and stress_ref, failrec_drone (both settings),
plant_panel for every plant, local_drone, diag_local and iter_local (flights, windows and --nofly), pipeline_drone for
every variant, fig_scores, reference_stats and the three timing scripts. The bootstrap of smoke_drone.py makes the
work small; where a README argument cannot apply to 48 synthetic flights it is shrunk (ksens_drone removes 8 and 16
flights, not 24-64; reference_stats uses seed 1 only). Each step checks that its result file is written. The plant
variants of plant_panel.py replace the weakened plant, so their dives need not crash. The numbers mean nothing.

usage (from anywhere): python tests/smoke_drone_more.py [--only plant_panel,timing_e2e] [--keep DIR]
"""
import glob
import json
import os

from smoke_common import ROOT, expect, run
from smoke_drone import BOOT, PREP, synthetic_neurobem

S = ["--seed", "1"]
P = ["--fault", "prop50j8", *S]
LOCAL = "sev50_n11_on25-50_s1"


def made(*patterns):
    """Each glob pattern matches a non-empty JSON file (or a text file) in work/."""
    def f(here):
        for pat in patterns:
            fs = glob.glob(os.path.join(here, pat))
            expect(fs, f"no file {pat}")
            for x in fs:
                if x.endswith(".json"):
                    expect(json.load(open(x)), f"{os.path.basename(x)} is empty")
        return ", ".join(os.path.basename(x) for pat in patterns for x in sorted(glob.glob(os.path.join(here, pat))))
    return f


STEPS = [
    PREP,
    ("nat_probe", ["nat_probe.py", "prop50j8", "1"], made("nat_prop50j8_s1.json"), ["prep"]),
    ("nat_probe_gain", ["nat_probe.py", "gain40j8", "1"], made("nat_gain40j8_s1.json"), ["prep"]),
    ("nat_probe_look", ["nat_probe.py", "prop50j8m", "1"], made("nat_prop50j8m_s1.json"), ["prep"]),
    ("cupid_drone", ["cupid_drone.py", "--fault", "prop50j8m", *S], made("cupid_prop50j8m_s1.json"), ["nat_probe_look"]),
    ("sup_drone", ["sup_drone.py", *P, "--fly"], made("sup_prop50j8_s1.json"), ["nat_probe"]),
    ("ksens_drone", ["ksens_drone.py", *P, "--fly", "8,16"], made("ksens_prop50j8_s1.json"), ["nat_probe"]),
    ("ckpt_drone", ["ckpt_drone.py", *P], made("ckpt_prop50j8_s1.json"), ["nat_probe"]),
    ("agg_check", ["agg_check.py", *P], made("agg_prop50j8_s1.json"), ["nat_probe"]),
    ("margin_drone", ["margin_drone.py", *P], made("margin_prop50j8_s1.json"), ["nat_probe"]),
    ("sweep_drone", ["sweep_drone.py", "--sev", "35", "--nper", "8", "--look", "4", *S, "--fly"],
     made("sweep_sev35_n8_l4_s1.json"), ["prep"]),
    ("stress_ref", ["stress_ref.py", "--sev", "35", "--nper", "8", "--look", "4", *S], made("stressref_sev35_n8_l4_s1.json"),
     ["sweep_drone"]),
    ("margin_sweep", ["margin_drone.py", "--sweep", "35,8,4", *S], made("margin_sweep_sev35_n8_l4_s1.json"), ["sweep_drone"]),
    ("failrec_drone", ["failrec_drone.py", *S], made("failrec_prop50j8_s1.json"), ["nat_probe"]),
    ("failrec_sweep", ["failrec_drone.py", "--sweep", "35,8,4", *S], made("failrec_sweep_sev35_n8_l4_s1.json"),
     ["sweep_drone"]),
] + [
    ("plant_panel", ["plant_panel.py", "--plant", p, *S], made(f"plant_{p}_prop50j8_s1.json"), ["nat_probe"])
    for p in ("wide", "subset", "gbt", "mismatch", "heldout")
] + [
    ("local_drone", ["local_drone.py", *S], made(f"local_{LOCAL}.json"), ["prep"]),
    ("diag_local", ["diag_local.py", *S], made(f"localdiag_{LOCAL}.json"), ["local_drone"]),
    ("iter_local", ["iter_local.py", *S], made(f"localiter_{LOCAL}.json"), ["local_drone"]),
    ("iter_local", ["iter_local.py", *S, "--unit", "windows"], made(f"localiterw_{LOCAL}.json"), ["local_drone"]),
    ("iter_local", ["iter_local.py", *S, "--nofly"], made(f"localiter_{LOCAL}_nofly.json"), ["local_drone"]),
] + [
    ("pipeline_drone", ["pipeline_drone.py", "--variant", v, *S, "--fly"], made(f"pipeline_{v}_prop50j8_s1.json"),
     ["nat_probe"])
    for v in ("filtered", "shifted", "differenced")
] + [
    ("fig_scores", ["fig_scores.py", "prop50j8m", "1"], made("fig_scores_prop50j8m_s1.json"), ["nat_probe_look"]),
    ("reference_stats", ["reference_stats.py", "--fault", "prop50j8", "--seeds", "1"],
     made("reference_failure_prop50j8.json"), ["nat_probe"]),
    ("timing_l2rw", ["timing_l2rw.py"], made("timing_l2rw.json"), ["nat_probe"]),
    ("timing_fact_fast", ["timing_fact_fast.py"], made("timing_l2rw.json"), ["timing_l2rw"]),
    ("timing_e2e", ["timing_e2e.py"], made("timing_e2e.json"), ["nat_probe", "nat_probe_gain", "nat_probe_look"]),
]

if __name__ == "__main__":
    run(os.path.join(ROOT, "code", "drone", "work"), "work", synthetic_neurobem, BOOT, STEPS, doc=__doc__)
