"""Rerun the scripts that write Tables 5-7 and the pooled files in results/drone/, on the per-seed result files in
results/drone/, and check that they reproduce paper/ras/main.tex and the committed pooled files exactly.

check_numbers.py recomputes the other tables and the text; this covers what the fill_* and pool_* scripts produce:
  Table tab:plants (5)  fill_robust.py, from pool_robust.json (pool_robust.py)
  Table tab:sweep  (6)  fill_sweep.py, which runs pool_sweep.py
  Table tab:local  (7)  fill_local.py
  pool_robust.json, pool_stress.json, pool_local_sev50_n11_on25-50.json and pooled_margin.txt (pool_margin.py)

The scripts run unchanged in a temporary copy (they read and write work/), so nothing in the repository is touched.
Needs NumPy only.

usage (from anywhere): python paper/ras/check_tables.py
"""
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results", "drone")
SCRIPTS = os.path.join(ROOT, "code", "drone", "work")
TEX = os.path.join(ROOT, "paper", "ras", "main.tex")

# (script and arguments, pooled file it writes in work/ or None, committed file in results/drone/ to compare with)
STEPS = [
    (["pool_robust.py"], "pool_robust.json", "pool_robust.json"),
    (["pool_stress.py"], "pool_stress.json", "pool_stress.json"),
    (["pool_local.py"], "pool_local_sev50_n11_on25-50.json", "pool_local_sev50_n11_on25-50.json"),
    (["pool_margin.py"], None, "pooled_margin.txt"),  # compared with the script's printout
    (["fill_robust.py", "main.tex"], None, None),
    (["fill_sweep.py", "main.tex"], None, None),
    (["fill_local.py", "main.tex"], None, None),
]


def show_diff(a, b, name_a, name_b, n=40):
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), name_a, name_b, lineterm=""))
    print("\n".join("      " + x for x in d[:n]) + ("\n      ..." if len(d) > n else ""))


def main():
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "work")
        os.mkdir(work)
        for f in os.listdir(RES):
            shutil.copy(os.path.join(RES, f), work)
        for f in os.listdir(SCRIPTS):
            if f.startswith(("pool_", "fill_")) and f.endswith(".py"):
                shutil.copy(os.path.join(SCRIPTS, f), work)
        shutil.copy(TEX, os.path.join(tmp, "main.tex"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for args, written, committed in STEPS:
            p = subprocess.run([sys.executable, os.path.join("work", args[0])] + args[1:], cwd=tmp, env=env,
                               capture_output=True, text=True)
            if p.returncode != 0:
                print(f"  FAIL {args[0]} exited with {p.returncode}:\n{p.stderr}")
                bad.append(args[0])
                continue
            if committed is None:
                print(f"  ran  {' '.join(args)}")
                continue
            ref_path = os.path.join(RES, committed)
            if written is None:  # the committed file is the script's printout
                same = open(ref_path).read() == p.stdout
                if not same:
                    show_diff(open(ref_path).read(), p.stdout, "results/drone/" + committed, args[0] + " output")
            else:
                same = json.load(open(ref_path)) == json.load(open(os.path.join(work, written)))
                if not same:
                    show_diff(json.dumps(json.load(open(ref_path)), indent=1),
                              json.dumps(json.load(open(os.path.join(work, written))), indent=1),
                              "results/drone/" + committed, "regenerated " + written)
            print(f"  {'ok  ' if same else 'MISMATCH'} results/drone/{committed} reproduced by {args[0]}")
            if not same:
                bad.append(committed)
        new = open(os.path.join(tmp, "main.tex")).read()
        old = open(TEX).read()
        if new == old:
            print("  ok   paper/ras/main.tex unchanged by fill_robust.py, fill_sweep.py and fill_local.py")
        else:
            print("  MISMATCH the fill scripts change paper/ras/main.tex:")
            show_diff(old, new, "paper/ras/main.tex", "after fill_*")
            bad.append("main.tex")
    print()
    if bad:
        print("mismatches:", ", ".join(bad))
        sys.exit(1)
    print("Tables 5-7 and the pooled files match the per-seed result files")


if __name__ == "__main__":
    main()
