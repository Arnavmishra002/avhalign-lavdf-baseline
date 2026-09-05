#!/usr/bin/env python3
"""Build avhalign_shared1000.ipynb = avhalign_cells.ipynb with CELL 2 set to the shared 1,000-clip protocol.
Usage: python3 build_shared1000.py [--train-all]   (--train-all: AVH-Align head fitted on all 600 train clips)"""
import json, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
subprocess.run([sys.executable, str(HERE / "build_cells.py")], check=True)
train_all = "--train-all" in sys.argv
nb = json.load(open(HERE / "avhalign_cells.ipynb"))
reps = [
    ('name="AVH-Align_LAVDF",', 'name="AVH-Align_LAVDF_shared1000%s",' % ("_trainall" if train_all else "")),
    ('max_train=200 if SMOKE else 3000,    # real training clips (lavdf protocol; the paper uses 45000)',
     'max_train=600,            # NOT used under shared1000 (disk estimate only): train = 300 real + 300 fake'),
    ('max_val=40 if SMOKE else 300,        # real validation clips (lavdf protocol)',
     'max_val=200,              # NOT used under shared1000: val = 100 real + 100 fake'),
    ('max_test=100 if SMOKE else 1000,     # balanced test clips (lavdf protocol)',
     'max_test=200,             # NOT used under shared1000: test = 100 real + 100 fake'),
    ('resume_data=True,         #', 'resume_data=False,         #'),
    ('budget_hours=2.0 if SMOKE else 11.0,', 'budget_hours=2.0 if SMOKE else 8.0,'),
    ('protocol="lavdf",         #', 'protocol="shared1000",         #'),
    ('avh_train_all=False,      #', 'avh_train_all=%s,      #' % ("True" if train_all else "False")),
]
c = next(c for c in nb["cells"] if c["cell_type"] == "code" and "# CELL 2 -" in "".join(c["source"]))
src = "".join(c["source"])
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
c["source"] = src.splitlines(keepends=True)
out = HERE / "avhalign_shared1000.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "train_all =", train_all)
