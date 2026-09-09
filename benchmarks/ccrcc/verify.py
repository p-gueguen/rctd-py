#!/usr/bin/env python3
"""Verify command for the autoresearch loop (READ-ONLY). Submits eval.sbatch on the GPU node,
waits, and prints the metric (one number) - or a non-numeric line on failure so the loop records
a metric-error instead of a phantom value.

    uv run python benchmarks/ccrcc/verify.py [--split tune] [--tag T] [--shuffle-protein-seed S]

Budget: one 40k-cell joint fit is ~3-4 min on the L40S; the Bash tool caps a call at 10 min,
so a busy node is reported, not waited for.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NODE = "fgcz-r-023"


def sh(args):
    return subprocess.run(args, text=True, capture_output=True)  # list args, no shell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shuffle-protein-seed", default=None)
    a = ap.parse_args()
    sha = sh(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"]).stdout.strip() or "nogit"
    tag = a.tag or f"{sha}_{int(time.time())}"
    state = sh(["sinfo", "-N", "-n", NODE, "-h", "-o", "%t"]).stdout.split()
    if not state or state[0] not in ("idle", "mix"):
        print(f"VERIFY-ERROR: {NODE} state {state} - not idle, refusing to queue", file=sys.stderr)
        print("nan")
        return 2
    extra = (
        f"--shuffle-protein-seed {int(a.shuffle_protein_seed)}"
        if a.shuffle_protein_seed is not None
        else ""
    )
    export = f"--export=ALL,AR_SPLIT={a.split},AR_TAG={tag},AR_EXTRA={extra}"
    for attempt in (1, 2):
        r = sh(["sbatch", "--wait", "--parsable", export, str(HERE / "eval.sbatch")])
        if r.returncode == 0:
            break
        print(f"VERIFY-WARN: sbatch attempt {attempt} failed: {r.stderr.strip()}", file=sys.stderr)
        time.sleep(20)
    else:
        print("nan")
        return 2
    js = HERE / "out" / f"eval_{tag}.json"
    if not js.exists():
        log = sorted((HERE / "out" / "logs").glob(f"eval_{tag}*.err"))
        tail = log[-1].read_text()[-1500:] if log else "(no log)"
        print(f"VERIFY-ERROR: no eval json for {tag}\n{tail}", file=sys.stderr)
        print("nan")
        return 2
    rec = json.loads(js.read_text())
    print(
        json.dumps(
            {
                k: rec[k]
                for k in [
                    "heldout_lineage_f1",
                    "tenx_fine_agreement",
                    "reject_frac",
                    "tenx_lineage_agreement",
                    "ca9_pos_frac_of_tumour_calls",
                    "elapsed_s",
                ]
            }
        ),
        file=sys.stderr,
    )
    print(f"{rec['metric']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
