"""Re-export the marimo tutorial as an executed notebook for the docs.

Read the Docs does not execute notebooks, so the docs ship them with outputs. Run on a GPU host
from the repository root, with marimo installed:

    RCTD_Q_MATRICES=/path/to/q_matrices.npz python docs/tutorials/export.py
"""

import json
import subprocess

SRC, OUT = "examples/tutorial.py", "docs/tutorials/slideseq_cerebellum.ipynb"

subprocess.run(["marimo", "export", "ipynb", SRC, "-o", OUT, "--include-outputs", "-f"], check=True)
nb = json.load(open(OUT))
# marimo's own import cell: the notebook does not use it, and it would fail without marimo installed
nb["cells"] = [c for c in nb["cells"] if "".join(c["source"]).strip() != "import marimo as mo"]
# marimo records each figure's display size; MyST-NB turns it into a fixed width and height, and
# the theme's max-width then narrows the image without its height, squashing it horizontally
for c in nb["cells"]:
    for o in c.get("outputs", []):
        o.get("metadata", {}).pop("image/png", None)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
