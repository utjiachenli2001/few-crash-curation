"""Redraw Figs. 2 and 3 with make_figs.py in a temporary copy and check that the files are byte-identical to
paper/figs/. The repository is not touched.

Needs NumPy and Matplotlib at the versions in requirements.txt: another Matplotlib (or its bundled FreeType on another
platform) can rasterise slightly differently, which shows up here as a mismatch without any change in the numbers;
check_numbers.py checks the numbers behind Fig. 2 separately.

usage (from anywhere): python paper/check_figs.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGS = os.path.join(ROOT, "paper", "figs")


def main():
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "paper", "figs"))
        shutil.copy(os.path.join(ROOT, "paper", "make_figs.py"), os.path.join(tmp, "paper"))
        shutil.copytree(os.path.join(ROOT, "results"), os.path.join(tmp, "results"))
        p = subprocess.run([sys.executable, os.path.join("paper", "make_figs.py")], cwd=tmp, capture_output=True,
                           text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if p.returncode != 0:
            print(p.stdout, p.stderr)
            sys.exit(1)
        new = sorted(os.listdir(os.path.join(tmp, "paper", "figs")))
        for f in sorted(set(new) | set(os.listdir(FIGS))):
            a, b = os.path.join(FIGS, f), os.path.join(tmp, "paper", "figs", f)
            if not os.path.exists(b):
                print(f"  MISSING   paper/figs/{f} is not drawn by make_figs.py"); bad.append(f); continue
            if not os.path.exists(a):
                print(f"  NEW       make_figs.py draws {f}, which is not in paper/figs/"); bad.append(f); continue
            same = open(a, "rb").read() == open(b, "rb").read()
            print(f"  {'ok  ' if same else 'MISMATCH'} paper/figs/{f}")
            if not same:
                bad.append(f)
    print()
    if bad:
        print("differ:", ", ".join(bad))
        sys.exit(1)
    print("make_figs.py redraws every figure in paper/figs/ byte for byte")


if __name__ == "__main__":
    main()
