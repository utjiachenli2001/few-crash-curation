"""Shared runner for the smoke tests: copy a platform's scripts into a temporary folder next to synthetic data and run
them there, unchanged, each in a fresh Python process that first executes a small bootstrap (which makes the work
small) and then the script as __main__."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The bootstrap gets the script path and its arguments in sys.argv; it ends by running the script.
RUN_SCRIPT = """
script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


def parse_args(doc):
    p = argparse.ArgumentParser(description=doc.split("\n")[0])
    p.add_argument("--keep", help="run in this folder and keep it (default: a temporary folder)")
    p.add_argument("--only", default="", help="comma-separated step names to run (default: all)")
    return p.parse_args()


def run(src, sub, make_data, boot, steps, env_extra=None, doc=""):
    """src: the platform's code folder; sub: the folder the scripts run in, relative to the run folder ("work" for
    the drone, where scripts run from code/drone/, or "." for the Crazyflie); make_data(folder) writes the synthetic
    data into the scripts' folder; steps: (name, [script, args...], check or None, needs) with check(folder) -> note
    and needs the names of earlier steps it depends on. In place of [script, args...] a step can give a function of
    the scripts' folder, run in this process (to rename a file, say), whose docstring labels it."""
    args = parse_args(doc)
    tmp = args.keep or tempfile.mkdtemp(prefix="smoke_")
    try:
        here = os.path.join(tmp, sub)
        os.makedirs(here, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith(".py"):
                shutil.copy(os.path.join(src, f), here)
        make_data(here)
        failed = run_steps(steps, args, tmp, here, sub, boot, env_extra)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failed else 0)


def run_steps(steps, args, tmp, here, sub, boot, env_extra):
    only = set(filter(None, args.only.split(",")))
    want = set()
    for name, _, _, needs in reversed(steps):  # a step and everything it needs
        if not only or name in only or name in want:
            want |= {name, *needs}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", NT="1", **(env_extra or {}))
    code = "import os, runpy, sys\nsys.path.insert(0, os.path.abspath(%r))\n" % sub + boot + RUN_SCRIPT
    t0, failed, ok = time.time(), [], set()
    for name, cmd, check, needs in steps:
        if name not in want:
            continue
        label = cmd.__doc__ if callable(cmd) else " ".join(cmd)
        if not set(needs) <= ok:
            print(f"  skip {label} (needs {', '.join(sorted(set(needs) - ok))})", flush=True)
            failed.append(label)
            continue
        t1 = time.time()
        if callable(cmd):
            cmd(here)
        else:
            r = subprocess.run([sys.executable, "-c", code, os.path.join(sub, cmd[0])] + cmd[1:], cwd=tmp,
                               env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAIL {label}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}", flush=True)
                failed.append(label)
                continue
        try:
            note = check(here) if check else ""
        except AssertionError as e:
            print(f"  FAIL {label}: {e}", flush=True)
            failed.append(label)
            continue
        ok.add(name)
        print(f"  ok   {label} ({time.time() - t1:.0f}s){': ' + note if note else ''}", flush=True)
    print(f"\n{'FAILED: ' + '; '.join(failed) if failed else 'all smoke steps passed'} ({time.time() - t0:.0f}s)")
    return failed


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def rates_ok(row):
    """Crash and reach rates of the three dive sets, each in [0, 1]."""
    keys = [t + r for t in ("test dives", "steep dives", "shallow dives") for r in ("", " reach")]
    return all(k in row and 0.0 <= row[k] <= 1.0 for k in keys)
