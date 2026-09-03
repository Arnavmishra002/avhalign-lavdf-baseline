#!/usr/bin/env python3
"""Assemble avhalign_fulltest.ipynb — the full-test-split variant.

The variant reuses three cells of the main notebook verbatim (imports, helpers,
environment setup) and replaces the rest with the cells in `_fulltest.py`, so
the two notebooks can never drift apart in the parts they share.

    python3 build_fulltest.py

Validation is the point of this script as much as assembly: every cell must
compile, the concatenation must have no undefined names, and the shared cells
must be byte-identical to the ones in avhalign_cells.ipynb.
"""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
MAIN = json.loads((HERE / "avhalign_cells.ipynb").read_text())
MAIN_CELLS = ["".join(c["source"]) for c in MAIN["cells"]]

# indices in avhalign_cells.ipynb: CELL 1 imports, CELL 3 helpers, CELL 5 setup
IMPORTS, HELPERS, SETUP = MAIN_CELLS[0], MAIN_CELLS[2], MAIN_CELLS[4]

frag = (HERE / "_fulltest.py").read_text()
# the file's own header explains what it is; it is not part of any cell
frag = frag.split("# CELL 2 -", 1)[1]
parts = ("# CELL 2 -" + frag).split("\n\n\n# CELL ")
new_cells = [parts[0]] + ["# CELL " + p for p in parts[1:]]

cells = [IMPORTS, new_cells[0], HELPERS, new_cells[1], SETUP] + new_cells[2:]

for i, c in enumerate(cells, 1):
    try:
        ast.parse(c)
    except SyntaxError as e:
        sys.exit(f"cell {i} does not compile: {e}")

concat = "\n\n".join(cells)
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
    fh.write(concat)
    tmp = fh.name
lint = subprocess.run([sys.executable, "-m", "pyflakes", tmp],
                      capture_output=True, text=True).stdout
undefined = [l for l in lint.splitlines() if "undefined name" in l]
if undefined:
    sys.exit("undefined names in the assembled notebook:\n" + "\n".join(undefined))

nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": c.splitlines(keepends=True)} for c in cells],
      "metadata": MAIN["metadata"], "nbformat": 4, "nbformat_minor": 5}
out = HERE / "avhalign_fulltest.ipynb"
out.write_text(json.dumps(nb, indent=1))

print(f"wrote {out} ({out.stat().st_size} bytes), {len(cells)} cells")
print(f"shared with avhalign_cells.ipynb: imports, helpers, setup "
      f"({len(IMPORTS.splitlines()) + len(HELPERS.splitlines()) + len(SETUP.splitlines())} lines)")
print("all cells compile; no undefined names in the concatenation")
