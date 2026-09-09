#!/usr/bin/env python3
"""Guard for the autoresearch loop (READ-ONLY): must pass on every iteration.

1. pipeline.py may not reach for the truth: no reference to the 10x labels, the landmark
   labels, the eval set, the gate markers or the data files.
2. The harness files are untouched (their hashes are pinned here at loop start).
3. The RNA-only path stays byte-identical (tests/test_protein_regression.py).
Exit 0 = pass.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORBIDDEN = re.compile(
    r"tenx|truth|landmark|eval_cell|cell_groups|GATE_MARKERS|autoresearch_data|\.h5ad", re.I
)
PINNED = {
    "eval.py": None,
    "prepare.py": None,
    "verify.py": None,
    "eval.sbatch": None,
    "guard.py": None,
}
PIN_FILE = HERE / "out" / "harness.sha256"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    bad = [
        f"{i}: {l.rstrip()}"
        for i, l in enumerate((HERE / "pipeline.py").read_text().splitlines(), 1)
        if FORBIDDEN.search(l) and not l.strip().startswith("#")
    ]
    if bad:
        print("GUARD FAIL: pipeline.py touches the truth or the data:\n  " + "\n  ".join(bad))
        return 1
    if PIN_FILE.exists():
        pins = dict(l.split() for l in PIN_FILE.read_text().splitlines() if l.strip())
        changed = [f for f in PINNED if pins.get(f) and pins[f] != digest(HERE / f)]
        if changed:
            print(f"GUARD FAIL: harness files modified: {changed}")
            return 1
    else:
        PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        PIN_FILE.write_text("".join(f"{f} {digest(HERE / f)}\n" for f in PINNED))
        print("guard: pinned harness hashes")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(HERE.parent.parent / "tests" / "test_protein_regression.py"),
        ],
        text=True,
        capture_output=True,
        cwd=HERE.parent.parent,
    )
    if r.returncode != 0:
        print("GUARD FAIL: test_protein_regression\n" + r.stdout[-2000:])
        return 1
    print("GUARD PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
