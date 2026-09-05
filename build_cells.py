"""Build a multi-cell Kaggle notebook from the proven single-file runner.

Every stage/helper function is sliced out of _inner.py with `ast` so the code
bodies stay byte-identical to the version that is currently running on Kaggle.
Only the glue changes: argparse -> a CONFIG cell, the main() loop -> a small
run_stage() helper called once per stage cell, and no __main__ auto-run.
"""
import ast
import json
import py_compile
import re
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "_inner.py").read_text()
LINES = SRC.splitlines(keepends=True)
TREE = ast.parse(SRC)

FUNCS = {}
for node in TREE.body:
    if isinstance(node, ast.FunctionDef):
        FUNCS[node.name] = "".join(LINES[node.lineno - 1:node.end_lineno])


def fn(*names):
    out = []
    for n in names:
        if n not in FUNCS:
            raise SystemExit(f"function {n} not found in _inner.py")
        out.append(FUNCS[n].rstrip("\n"))
    return "\n\n\n".join(out)


CELLS = []


def cell(text):
    CELLS.append(textwrap.dedent(text).strip("\n") + "\n")


# --------------------------------------------------------------------------
cell('''
# CELL 1 - Imports
#
# Only the Python standard library is imported into the notebook kernel.
# Every heavy step (dlib face landmarks, AV-HuBERT feature extraction, training,
# evaluation) runs as a *subprocess* using the upstream repositories, so the
# kernel itself never imports torch or numpy. That matters because CELL 5
# installs numpy<2 and pins omegaconf/hydra for the 2021-era fairseq code: the
# subprocesses pick those pins up cleanly while the kernel stays untouched.

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

print("All libraries imported successfully")
''')

# --------------------------------------------------------------------------
cell('''
# CELL 2 - Configuration
#
# Everything that controls the run lives here. All working paths are under
# /kaggle/working, so every intermediate artifact (mouth ROIs, features,
# checkpoints, stage markers) is saved in the notebook output and survives a
# crash or a session timeout -- CELL 4 shows how a later run reuses them.
#
# SMOKE = True  -> tiny end-to-end pilot (~2 h) to validate the pipeline.
# SMOKE = False -> the real run: 3000 real train clips, 300 real val clips,
#                  1000 balanced (500 real / 500 fake) test clips.

WORK = Path("/kaggle/working")
REPO = WORK / "AVH-Align"                    # bit-ml/AVH-Align (CVPR 2025)
AVHUBERT = WORK / "av_hubert" / "avhubert"   # facebookresearch/av_hubert (feature backbone)

LINKS = WORK / "lavdf_root"    # symlink farm pointing at the LAV-DF mp4 files
META = WORK / "lavdf_meta"     # train / val / test metadata CSVs
PRE = WORK / "lavdf_pre"       # preprocessed 96x96 mouth-ROI clips (mp4 + wav)
FEATS = WORK / "lavdf_feats"   # AV-HuBERT features (.npz per clip)
CKPT = WORK / "checkpoints"    # trained alignment model
STATE = WORK / "state"         # <stage>.done markers used for resume
SCORES = WORK / "scores"       # per-clip scores written by CELL 10

ALL_STAGES = ["setup", "metadata", "preprocess", "extract", "train", "eval", "probe"]

SMOKE = False

CFG = SimpleNamespace(
    lavdf_root="/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df",
    stages="all",             # comma list, or "all"; completed stages are skipped
    force="",                 # comma list of stages to re-run even if marked done
    name="AVH-Align_LAVDF",
    max_train=200 if SMOKE else 3000,    # real training clips (the paper uses 45000)
    max_val=40 if SMOKE else 300,        # real validation clips (the paper uses 5000)
    max_test=100 if SMOKE else 1000,     # test clips drawn from the LAV-DF test split
    test_balanced=True,       # True  -> forced 50/50, the protocol behind the
                              #          reported AP 0.7872 / AUC 0.8263.
                              # False -> uniform sample of the test split, so AP
                              #          sits on LAV-DF's own class prior (73.5%
                              #          fake) and is comparable with published AP.
    data_splits="train,test", # which splits to preprocess+extract; use "test"
                              # alone when a trained model is restored from an
                              # attached input and only evaluation is needed.
    resume_data=True,         # reuse ROIs/features found in an attached input.
                              # Set False to force fresh data (e.g. a new test set)
                              # while still reusing a trained checkpoint.
    epochs=3 if SMOKE else 40,           # upper bound; early stopping usually ends sooner
    budget_hours=2.0 if SMOKE else 11.0, # Kaggle kills the session at 12 h
    workers=4,
    seed=42,
    protocol="lavdf",         # "lavdf" (audited run) or "shared1000": one seeded draw of
                              # n_pool clips from the whole dataset, split n_train/n_val/
                              # rest -- the reviewers' 1000-video 600/200/200 setup.
                              # AVH-Align still trains on the REAL clips of that split
                              # (it has no supervised loss); CELL 12 adds a supervised probe.
    split_file="",            # CSV path,split[,label] with the reviewers' exact clips
    n_pool=1000,
    n_train=600,
    n_val=200,
    skip_pip=False,
    purge_preprocessed=True,             # delete mouth ROIs once features exist
)

# The wall-clock budget starts now. Later stages check time_left() before
# committing to long work so the session can stop cleanly and save its state.
T0 = time.time()
DEADLINE = T0 + CFG.budget_hours * 3600
BYTES_PER_CLIP = None

print(f"test sampling: {'balanced 50/50' if CFG.test_balanced else 'natural prior'}"
      f" | splits processed: {CFG.data_splits} | resume_data={CFG.resume_data}")
print(f"SMOKE={SMOKE}  max_train={CFG.max_train}  max_val={CFG.max_val}  "
      f"max_test={CFG.max_test}  epochs={CFG.epochs}  budget={CFG.budget_hours}h")
print("Deadline:", time.strftime("%H:%M:%S", time.localtime(DEADLINE)))
''')

# --------------------------------------------------------------------------
cell('''
# CELL 3 - Helper Functions
#
# Small utilities shared by every stage:
#   log / run          timestamped logging and subprocess execution that aborts
#                      the run on a non-zero exit code
#   done / mark        stage completion markers in /kaggle/working/state
#   find_input_dir / fetch_repo
#                      git clone with retries, falling back to a checkout found
#                      in an attached input when GitHub is unreachable
#   budget_*, require_time, require_disk
#                      guards so a stage never starts work it cannot finish
#                      inside the session budget or the 20 GB output quota
#   run_stage          the stage runner: skips a stage whose marker exists
#                      (unless it is listed in CFG.force), otherwise runs it
#                      with timing

''' + fn("log", "run", "done", "mark", "disk_report", "find_input_dir", "fetch_repo",
         "est_feature_bytes",
         "time_left", "hms", "budget_report", "require_time", "free_bytes",
         "require_disk", "count_rows") + '''


ACTIVE_STAGES = ALL_STAGES if CFG.stages == "all" else [s.strip() for s in CFG.stages.split(",")]
FORCED = {s.strip() for s in CFG.force.split(",") if s.strip()}


def run_stage(name, stage_fn):
    if name not in ALL_STAGES:
        raise SystemExit(f"unknown stage {name!r}; valid: {ALL_STAGES}")
    if name not in ACTIVE_STAGES:
        log(f"--- {name} not selected in CFG.stages; skipping ---")
        return
    if done(name) and name not in FORCED:
        log(f"--- skipping {name} (already done; add it to CFG.force to re-run) ---")
        return
    log(f"=== stage: {name} (remaining budget {hms(time_left())}) ===")
    t0 = time.time()
    stage_fn(CFG)
    log(f"=== {name} finished in {(time.time()-t0)/60:.1f} min ===")


log(f"[budget] wall-clock budget {CFG.budget_hours:.1f}h "
    f"(Kaggle hard limit is 12h); deadline at "
    f"{time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")
est = est_feature_bytes(CFG.max_train + CFG.max_val + CFG.max_test)
log(f"[plan] max_train={CFG.max_train} max_val={CFG.max_val} max_test={CFG.max_test} "
    f"-> ~{est/2**30:.2f} GiB of features, plus ~4 GiB env+checkpoint")
''')

# --------------------------------------------------------------------------
cell('''
# CELL 4 - Resume From Persisted Checkpoints
#
# Preprocessing (~4.5 h) and feature extraction are the expensive stages, and
# their outputs are written to /kaggle/working so they are kept in the notebook
# output. To resume after a crash: add this notebook's previous output as an
# input dataset and re-run. The function below finds the saved lavdf_pre /
# lavdf_feats folders, links them in, and marks those stages done -- but only
# when the data is really present. A bare .done marker is never trusted.

''' + fn("restore_checkpoint") + '''


restore_checkpoint()
''')

# --------------------------------------------------------------------------
cell('''
# CELL 5 - Environment Setup: Repositories, Dependencies, Compatibility Patches, Assets
#
# What this cell does, in order:
#   0. Checks the accelerator first. The pinned torch build drives sm_70 and
#      newer; on Kaggle's P100 (sm_60) every CUDA kernel raises and the upstream
#      extraction script swallows it, so 4300 clips yield zero features after
#      hours of work. That is a two-minute failure here instead.
#   1. Clones bit-ml/AVH-Align and facebookresearch/av_hubert (with its pinned
#      fairseq submodule, commit afc77bd from 2021). The clone is retried, and
#      if GitHub is unreachable from the worker the checkout persisted in the
#      attached V8-output dataset is copied instead (fetch_repo, CELL 3).
#   2. Installs the pinned dependencies (numpy<2, omegaconf 2.0.6, hydra 1.0.7,
#      dlib-bin, python_speech_features, ...). fairseq is put on sys.path via a
#      .pth file instead of pip, because its CUDA extension no longer builds
#      against current torch and is not needed here.
#   3. Forces torch.load(weights_only=False) for every subprocess so the 2021
#      AV-HuBERT checkpoint (pickled config objects) can be loaded on torch 2.x.
#   4. Patches the 2021 code for Python 3.12 / numpy >= 1.24: mutable dataclass
#      defaults -> field(default_factory=...), removed numpy aliases, moved
#      collections ABCs, and rewrites av_hubert's package-relative imports to
#      absolute ones (the extraction script runs them as top-level modules).
#      fairseq.hydra_init() is disabled because the hydra CLI is never used.
#   5. Runs a self-healing import check that loads the *exact* module chain the
#      feature-extraction script needs, so any remaining incompatibility fails
#      here in minutes instead of after hours of preprocessing.
#   6. Downloads the dlib 68-landmark model, the mean-face template and the
#      AV-HuBERT large checkpoint (self_large_vox_433h.pt, 5.4 GB), copies the
#      deepfake_* scripts into place, and removes their unused multimodal pass
#      (~33% less GPU time and feature disk).

# ffmpeg is hard-coded by the preprocessing script; make sure it is installed.
subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=False)


''' + fn("stage_setup") + '''


run_stage("setup", stage_setup)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 6 - LAV-DF Metadata and Train / Validation / Test Splits
#
# Reads metadata.json, keeps clips with at least 31 frames, and labels a clip
# REAL only when neither its video nor its audio was modified and n_fakes == 0.
# AVH-Align is trained on real clips only (it learns audio-visual alignment and
# flags deviations from it), so train and val contain real clips exclusively;
# the test set is balanced 50/50 real/fake. Selection is seeded (CFG.seed) so
# the same clips are chosen on every run. A symlink farm under lavdf_root and
# four CSVs under lavdf_meta are produced in the layout the upstream scripts
# expect.

''' + fn("find_lavdf_metadata", "stage_metadata") + '''


run_stage("metadata", stage_metadata)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 7 - Preprocess: Mouth-ROI Extraction
#
# For every clip, dlib detects 68 facial landmarks, the mouth region is aligned
# to the mean face and cropped to a 96x96 grayscale video, and the audio track
# is exported as 16 kHz wav. This is the input format AV-HuBERT expects. It is
# CPU-bound (4 workers) and takes about 4.5 h for 4300 clips. Clips in which no
# face is detected fall back to the whole frame ("passing whole video" in the
# log) -- an expected upstream behaviour affecting roughly 1% of clips.

''' + fn("stage_preprocess") + '''


run_stage("preprocess", stage_preprocess)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 8 - AV-HuBERT Feature Extraction
#
# The frozen AV-HuBERT large model (self_large_vox_433h) turns each mouth-ROI
# clip into per-frame visual and audio feature sequences, saved as one .npz per
# clip -- 6.12 GiB for 4300 clips, 1.46 MiB per clip, measured. The multimodal
# pass was removed in CELL 5 because the alignment model never reads it. Each
# split's mouth ROIs are deleted once that split's features exist, to stay
# inside the 20 GB output quota; a split that produced none keeps its ROIs so
# the run can be retried without redoing 4.5 h of preprocessing.

''' + fn("stage_extract") + '''


run_stage("extract", stage_extract)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 9 - Train the Alignment Model
#
# Trains AVH-Align's small fusion model (~1M parameters) on the frozen features
# of REAL clips only, using the upstream train.py with the paper's settings
# (tau=15, batch 1024, lr 1e-5, early-stopping patience 10). Training is wrapped
# in `timeout` so it can never run past the session budget; 15 minutes are
# reserved for evaluation. If the cap is hit, the best checkpoint saved so far
# is used and the epoch count should be reported.

''' + fn("stage_train") + '''


run_stage("train", stage_train)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 10 - Evaluation: AP, AUC and the full operating-point metric suite
#
# Runs the upstream eval.py (AUC / AP) on the test set for every checkpoint that
# is available, then re-runs the identical arithmetic keeping one score per clip,
# because AP and AUC alone cannot give recall, F1, an operating point, or a
# significance test against another model. The per-clip scores are written to
# /kaggle/working/scores/test_scores.csv and summarised right here, so a single
# run of this notebook produces every number the write-up needs.
#
# Checkpoints evaluated:
#   - "retrained on LAV-DF real subset": the model trained in CELL 9
#   - "official AV1M checkpoint, zero-shot": the authors' released weights, if
#     the repository ships them
#
# Reporting note: these numbers come from a 3k-clip subset trained inside one
# Kaggle session. Label them "retrained-3k-subset" / "official-checkpoint-
# zero-shot"; they must not be presented as a reproduction of the published
# LAV-DF results.

''' + fn("stage_eval") + '''


run_stage("eval", stage_eval)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 12 - Supervised Probe (shared 600/200/200 protocol only)
#
# AVH-Align is self-supervised and real-only, so it cannot "train on fakes".
# To give the comparison a row that uses both classes exactly like the
# supervised detectors, this cell fits a logistic-regression probe on the FROZEN
# AV-HuBERT clip features of every train clip (real and fake), picks C on val,
# and scores the same test clips. Report it as "AV-HuBERT features + linear
# probe", never as AVH-Align. Skipped automatically under the lavdf protocol.

''' + fn("stage_probe") + '''


run_stage("probe", stage_probe)
''')

# --------------------------------------------------------------------------
cell('''
# CELL 11 - Final Summary

budget_report("ALL DONE")

print("=================================")
print("AVH-Align on LAV-DF (single-session run)")
print("=================================")
print()
print("Completed stages:", sorted(p.stem for p in STATE.glob("*.done")) if STATE.exists() else [])
print("Checkpoints:", sorted(p.name for p in CKPT.glob("*.pt")) if CKPT.exists() else [])
n_tr = len(list((FEATS / "train").glob("*.npz"))) if (FEATS / "train").exists() else 0
n_te = len(list((FEATS / "val").glob("*.npz"))) if (FEATS / "val").exists() else 0
print(f"Features on disk: {n_tr} train+val clips, {n_te} test clips")
disk_report()
print()
print("Metrics are printed in CELL 10: AP, AUC and EER, plus accuracy,")
print("precision, recall, F1 and specificity at three operating points, from")
print("per-clip scores saved to /kaggle/working/scores/test_scores.csv.")
print("Label them 'retrained-3k-subset' / 'official-checkpoint-zero-shot';")
print("this is NOT a reproduction of the published LAV-DF numbers.")
print("=================================")
''')

# --------------------------------------------------------------------------
# Assemble the .ipynb
nb = {
    "cells": [
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": c}
        for c in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT = HERE / "avhalign_cells.ipynb"
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes), {len(CELLS)} cells")

# --------------------------------------------------------------------------
# Validation 1: every cell compiles on its own; the concatenation compiles.
for i, c in enumerate(CELLS, 1):
    p = HERE / f"_cell{i}.py"
    p.write_text(c)
    py_compile.compile(str(p), doraise=True)
concat = "\n".join(CELLS)
(HERE / "_concat.py").write_text(concat)
py_compile.compile(str(HERE / "_concat.py"), doraise=True)
print("all cells compile; concatenation compiles")

# Validation 2: every function in the source (except main) is present verbatim.
missing = [n for n in FUNCS if n != "main" and FUNCS[n].rstrip("\n") not in concat]
print("functions missing from notebook:", missing or "NONE")

# Validation 3: no argparse / main / __main__ remnants; every args.<x> has a
# matching CFG field.
bad = [k for k in ("argparse", "def main(", "__main__", "parse_args") if k in concat]
print("leftover CLI remnants:", bad or "NONE")
cfg_keys = set(re.findall(r"^\s+(\w+)=", CELLS[1], re.M))
used = set(re.findall(r"\bargs\.(\w+)", concat))
print("args.<field> without CFG entry:", sorted(used - cfg_keys) or "NONE")

# Validation 4: pyflakes for undefined names, if available.
try:
    import pyflakes.api, pyflakes.reporter
    import io
    buf = io.StringIO()
    rep = pyflakes.reporter.Reporter(buf, buf)
    pyflakes.api.check(concat, "_concat.py", rep)
    out = [l for l in buf.getvalue().splitlines() if "undefined name" in l]
    print("pyflakes undefined names:", out or "NONE")
except ImportError:
    print("pyflakes not installed -- skipped")

# Validation 5: cell order = definition order (each name used in a cell must
# be defined in that cell or an earlier one). Static approximation via ast.
defined = set(dir(__builtins__))
problems = []
for i, c in enumerate(CELLS, 1):
    t = ast.parse(c)
    names_defined = set()
    for node in ast.walk(t):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names_defined.add(node.name)
        elif isinstance(node, ast.Import):
            names_defined.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names_defined.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for tg in node.targets:
                for n in ast.walk(tg):
                    if isinstance(n, ast.Name):
                        names_defined.add(n.id)
    top_level_loads = set()
    for node in t.body:  # only module-level statements, not inside defs
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                top_level_loads.add(n.id)
    undefined = sorted(n for n in top_level_loads
                       if n not in defined and n not in names_defined)
    if undefined:
        problems.append((i, undefined))
    defined |= names_defined
print("top-level names used before definition:", problems or "NONE")
