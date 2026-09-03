# CELL 2 - Configuration: full LAV-DF test split
#
# This notebook scores AVH-Align on the COMPLETE LAV-DF test split (26,100
# clips, 6,906 real / 19,194 fake) instead of a subsample, so AP sits on the
# dataset's own class prior and is directly comparable with published numbers.
#
# It cannot run in one Kaggle session: 26,100 clips need ~28 h of mouth-ROI
# preprocessing and 37 GiB of features, against a 12 h / 20 GB session. The work
# is therefore split by MODE, and each mode resumes from the previous session's
# output:
#
#   MODE = "preprocess"  CPU only. Cuts mouth ROIs for as many not-yet-done
#                        clips as the session budget and disk allow (~10,000),
#                        and leaves them in the output. No GPU quota is spent.
#                        Run this 3x, attaching the previous outputs each time.
#
#   MODE = "score"       GPU. Attaches every ROI output, then walks the split in
#                        chunks: extract features -> score both checkpoints ->
#                        write per-clip scores -> delete the features. Peak disk
#                        stays ~17 GiB and the whole split costs ~2.5 h of GPU.
#                        The last session prints AP / AUC with bootstrap CIs.

MODE = "preprocess"        # "preprocess" (CPU sessions) then "score" (GPU session)

WORK = Path("/kaggle/working")
REPO = WORK / "AVH-Align"
AVHUBERT = WORK / "av_hubert" / "avhubert"

LINKS = WORK / "lavdf_root"
META = WORK / "lavdf_meta"
PRE = WORK / "lavdf_pre"
FEATS = WORK / "lavdf_feats"
CKPT = WORK / "checkpoints"
SCORES = WORK / "scores"          # per-chunk score CSVs; tiny, always persisted
STATE = WORK / "state"

ALL_STAGES = ["setup", "metadata", "preprocess", "extract", "train", "eval"]

CFG = SimpleNamespace(
    lavdf_root="/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df",
    stages="all",
    force="",
    name="AVH-Align_LAVDF",          # our retrained checkpoint
    official="AVH-Align_AV1M",       # the authors' released checkpoint
    max_train=0, max_val=0, max_test=0,   # unused here; the whole split is scored
    epochs=0,
    budget_hours=11.0,
    workers=4,
    seed=42,
    skip_pip=False,
    chunk=4000,                      # clips per extract+score chunk
    prep_batch=2000,                 # clips per preprocess subprocess call
)

T0 = time.time()
DEADLINE = T0 + CFG.budget_hours * 3600
BYTES_PER_CLIP = None

# Measured on this pipeline: 3.9 s/clip preprocessing on 4 workers, 0.25 s/clip
# extraction on a T4, 0.28 MiB of mouth ROI and 1.46 MiB of features per clip.
# The whole test split's ROIs are only ~7 GiB, so preprocessing is bounded by
# session time, not disk; the features are what force chunked scoring.
SEC_PREP, SEC_EXTRACT = 3.9, 0.25
BYTES_ROI, BYTES_FEAT = 400_000, 1_600_000

print(f"MODE={MODE}  chunk={CFG.chunk}  budget={CFG.budget_hours}h")
print("Deadline:", time.strftime("%H:%M:%S", time.localtime(DEADLINE)))


# CELL 4 - Resume: checkpoints, mouth ROIs and scores from earlier sessions
#
# Three kinds of prior work can be attached as inputs, and each is picked up
# independently:
#   checkpoints/<name>.pt   the trained alignment head (from the training run)
#   lavdf_pre/val/*_roi.mp4 mouth ROIs cut by earlier "preprocess" sessions
#   scores/chunk_*.csv      per-clip scores already computed
# Nothing is recomputed that is already present, so a session always makes
# forward progress even if an earlier one was cut short.

def restore_inputs():
    inp = Path("/kaggle/input")
    if not inp.exists():
        log("[resume] no inputs attached")
        return

    ck = find_input_dir("checkpoints", (f"{CFG.name}.pt",))
    if ck is not None and not (CKPT / f"{CFG.name}.pt").exists():
        CKPT.mkdir(parents=True, exist_ok=True)
        shutil.copy(ck / f"{CFG.name}.pt", CKPT / f"{CFG.name}.pt")
        log(f"[resume] trained checkpoint <- {ck}")

    # Scores are small: copy them in so this session can skip those chunks and
    # still write one complete score set to its own output.
    SCORES.mkdir(parents=True, exist_ok=True)
    n_sc = 0
    for cur, dirs, files in os.walk(inp, topdown=True):
        if Path(cur).name == "scores":
            for f in files:
                if f.startswith("chunk_") and f.endswith(".csv"):
                    dst = SCORES / f
                    if not dst.exists():
                        shutil.copy(Path(cur) / f, dst)
                        n_sc += 1
            dirs[:] = []
        elif len(dirs) + len(files) > 2000 or cur[len(str(inp)):].count(os.sep) >= 6:
            dirs[:] = []
    if n_sc:
        log(f"[resume] {n_sc} score chunks copied from inputs")

    # ROI directories stay where they are (read-only mounts); a symlink farm in
    # CELL 6 gives the extraction script one directory that spans all of them.
    global ROI_MOUNTS
    ROI_MOUNTS = []
    for cur, dirs, files in os.walk(inp, topdown=True):
        p = Path(cur)
        if p.name == "val" and p.parent.name == "lavdf_pre":
            ROI_MOUNTS.append(p)
            dirs[:] = []
        elif len(dirs) + len(files) > 2000 or cur[len(str(inp)):].count(os.sep) >= 7:
            dirs[:] = []
    if ROI_MOUNTS:
        log(f"[resume] mouth-ROI mounts: {[str(m) for m in ROI_MOUNTS]}")
    else:
        log("[resume] no mouth ROIs attached")


ROI_MOUNTS = []
restore_inputs()


# CELL 6 - The complete LAV-DF test split
#
# Every clip of the official test split, in a fixed order (sorted by file path),
# so chunk boundaries are identical in every session and on every machine. No
# sampling and no class balancing: the label mix is the dataset's own, which is
# what makes AP comparable with published numbers.

def build_full_test(args):
    lavdf = Path(args.lavdf_root)
    META.mkdir(parents=True, exist_ok=True)
    meta_path = find_lavdf_metadata(lavdf)
    log(f"LAV-DF metadata: {meta_path}")
    video_root = meta_path.parent

    with open(meta_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("clips", list(data.values()))

    recs = []
    for r in data:
        if r.get("split") != "test":
            continue
        rel = r.get("file") or r.get("path") or r.get("filename")
        nf = r.get("video_frames") or r.get("n_frames") or r.get("num_frames")
        if rel is None or not nf or int(nf) < 31:
            continue
        mv, ma = bool(r.get("modify_video")), bool(r.get("modify_audio"))
        nfakes = r.get("n_fakes", len(r.get("fake_periods") or []))
        recs.append({"rel": str(rel), "num_frames": int(nf),
                     "label": 0 if (not mv and not ma and nfakes == 0) else 1})

    recs.sort(key=lambda r: r["rel"])          # fixed order == stable chunks
    seen = {}
    for rc in recs:
        base = Path(rc["rel"]).name
        if base in seen and seen[base] != rc["rel"]:
            base = Path(rc["rel"]).parent.name + "_" + base
        seen[base] = rc["rel"]
        rc["flat"] = base

    n_real = sum(1 for r in recs if r["label"] == 0)
    log(f"[test] full LAV-DF test split: {len(recs)} clips, {n_real} real / "
        f"{len(recs) - n_real} fake (fake prior {1 - n_real / len(recs):.3f})")

    with open(META / "full_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "num_frames", "rel"])
        w.writeheader()
        for rc in recs:
            w.writerow({"path": rc["flat"], "label": rc["label"],
                        "num_frames": rc["num_frames"], "rel": rc["rel"]})
    log(f"wrote {META / 'full_test.csv'}  rows={len(recs)}")
    return recs, video_root, lavdf


def link_clips(recs, video_root, lavdf):
    """Symlink farm in the layout deepfake_preprocess.py expects (test -> val)."""
    (LINKS / "val").mkdir(parents=True, exist_ok=True)
    n = 0
    for rc in recs:
        src = video_root / rc["rel"]
        if not src.exists():
            src = lavdf / rc["rel"]
        dst = LINKS / "val" / rc["flat"]
        if dst.exists() or dst.is_symlink() or not src.exists():
            continue
        os.symlink(src, dst)
        n += 1
    log(f"symlinked {n} clips -> {LINKS / 'val'}")


def roi_done(recs):
    """Names whose mouth ROI already exists in an attached mount or locally."""
    have = set()
    for m in ROI_MOUNTS + [PRE / "val"]:
        if m.exists():
            have |= {p.name[:-8] for p in m.glob("*_roi.mp4")}   # strip _roi.mp4
    return {rc["flat"] for rc in recs if rc["flat"][:-4] in have}


TEST_RECS, VIDEO_ROOT, LAVDF_ROOT = build_full_test(CFG)


# CELL 7 - Mode "preprocess": cut mouth ROIs, CPU only
#
# Takes the next clips that have no ROI yet and preprocesses as many as the
# session's time and disk allow, in batches of CFG.prep_batch so an interrupted
# session still leaves every completed batch in the output. Nothing here needs a
# GPU, so run these sessions with the accelerator set to None and keep the GPU
# quota for the scoring session.

def stage_prep(args):
    done = roi_done(TEST_RECS)
    todo = [rc for rc in TEST_RECS if rc["flat"] not in done]
    log(f"[prep] {len(done)}/{len(TEST_RECS)} clips already have ROIs; {len(todo)} left")
    if not todo:
        log("[prep] nothing to do -- switch MODE to 'score'")
        return

    by_time = int(max(0, time_left() - 20 * 60) / SEC_PREP)
    by_disk = int(free_bytes(WORK) * 0.85 / BYTES_ROI)
    n = max(0, min(len(todo), by_time, by_disk))
    log(f"[prep] budget allows {by_time} clips, disk allows {by_disk} -> doing {n}")
    if n == 0:
        raise SystemExit("no room for even one clip; attach fewer inputs or a longer session")

    batch = todo[:n]
    link_clips(batch, VIDEO_ROOT, LAVDF_ROOT)
    PRE.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(batch), args.prep_batch):
        part = batch[i:i + args.prep_batch]
        if time_left() < len(part) * SEC_PREP + 10 * 60:
            log(f"[prep] stopping before batch {i // args.prep_batch}: not enough budget left")
            break
        csv_path = META / f"prep_{i // args.prep_batch}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label"])
            w.writeheader()
            for rc in part:
                w.writerow({"path": rc["flat"], "label": rc["label"]})
        run([sys.executable, "deepfake_preprocess.py",
             "--dataset", "AV1M", "--split", "test",
             "--metadata", csv_path,
             "--data_path", LINKS,
             "--save_path", PRE,
             "--max_workers", str(args.workers)], cwd=AVHUBERT)
        have = len(list((PRE / "val").glob("*_roi.mp4"))) if (PRE / "val").exists() else 0
        log(f"[prep] {have} ROIs in this session's output")
        disk_report()
        budget_report(f"after batch {i // args.prep_batch}")

    total = len(roi_done(TEST_RECS))
    log(f"[prep] {total}/{len(TEST_RECS)} clips now have ROIs "
        f"({len(TEST_RECS) - total} still to do)")
    if total >= len(TEST_RECS):
        log("[prep] the split is fully preprocessed -- next session: MODE = 'score'")


# CELL 8 - Mode "score": extract features and score both checkpoints
#
# For each chunk of CFG.chunk clips: run AV-HuBERT over the chunk's mouth ROIs,
# score every clip with both checkpoints in a single pass over the features, save
# the per-clip scores, then delete the features. Peak disk stays around 17 GiB
# and only the scores survive, so the whole 26,100-clip split fits in one GPU
# session at roughly 2.5 h.
#
# The scoring code is byte-for-byte the pipeline in the authors' eval.py --
# L2-normalise both streams, run the fusion model, take logsumexp(-output) -- so
# the numbers are theirs, only the loop is ours.

SCORER = r'''
import csv, os, sys, numpy as np, torch
from model import FusionModel

meta, feats, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
ckpts = dict(p.split("=", 1) for p in sys.argv[4:])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
for name, path in ckpts.items():
    m = FusionModel().to(device)
    m.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    m.eval()
    models[name] = m

rows = list(csv.DictReader(open(meta)))
names = list(models)
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "label"] + [f"score_{n}" for n in names])
    kept = 0
    for r in rows:
        f = os.path.join(feats, r["path"].replace(".mp4", ".npz"))
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        v = torch.from_numpy(d["visual"]).to(device)
        a = torch.from_numpy(d["audio"]).to(device)
        v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)
        a = a / torch.linalg.norm(a, ord=2, dim=-1, keepdim=True)
        scores = []
        with torch.no_grad():
            for n in names:
                o = models[n](v, a)
                scores.append(float(torch.logsumexp(-o, dim=0).detach().cpu().squeeze()))
        w.writerow([r["path"], r["label"]] + scores)
        kept += 1
print(f"scored {kept}/{len(rows)} clips -> {out_csv}")
'''

AGGREGATE = r'''
import csv, glob, sys, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/chunk_*.csv")):
    rows += list(csv.DictReader(open(f)))
if not rows:
    sys.exit("no scores found")

y = np.array([int(r["label"]) for r in rows])
names = [k[6:] for k in rows[0] if k.startswith("score_")]
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]

print(f"clips scored: {len(y)}  ({int((y == 0).sum())} real / {int(y.sum())} fake, "
      f"fake prior {y.mean():.3f})")
for n in names:
    s = np.array([float(r["score_" + n]) for r in rows])
    ap, auc = average_precision_score(y, s), roc_auc_score(y, s)
    aps = [average_precision_score(y[i], s[i]) for i in idx]
    aucs = [roc_auc_score(y[i], s[i]) for i in idx]
    lo_ap, hi_ap = np.percentile(aps, [2.5, 97.5])
    lo_au, hi_au = np.percentile(aucs, [2.5, 97.5])
    print(f"{n}:  AP {ap:.4f} [{lo_ap:.4f}, {hi_ap:.4f}]   "
          f"AUC {auc:.4f} [{lo_au:.4f}, {hi_au:.4f}]")
'''


def stage_score(args):
    if not ROI_MOUNTS and not (PRE / "val").exists():
        raise SystemExit("no mouth ROIs available -- run MODE='preprocess' sessions first")

    # One directory spanning every attached ROI mount, so the extraction script
    # sees a single --data_path. Symlinks, so nothing is copied.
    merged = PRE / "val"
    merged.mkdir(parents=True, exist_ok=True)
    n_link = 0
    for m in ROI_MOUNTS:
        for p in m.iterdir():
            dst = merged / p.name
            if not dst.exists() and not dst.is_symlink():
                os.symlink(p, dst)
                n_link += 1
    log(f"[score] {n_link} ROI symlinks merged into {merged}")

    have = {p.name[:-8] for p in merged.glob("*_roi.mp4")}
    ready = [rc for rc in TEST_RECS if rc["flat"][:-4] in have]
    log(f"[score] {len(ready)}/{len(TEST_RECS)} clips have ROIs")

    SCORES.mkdir(parents=True, exist_ok=True)
    ck = {}
    if (CKPT / f"{args.name}.pt").exists():
        ck[args.name] = CKPT / f"{args.name}.pt"
    off = REPO / "checkpoints" / f"{args.official}.pt"
    if off.exists():
        ck[args.official] = off
    if not ck:
        raise SystemExit("no checkpoint to score with")
    log(f"[score] checkpoints: {list(ck)}")
    (REPO / "score_clips.py").write_text(SCORER)

    chunks = [(i, ready[i:i + args.chunk]) for i in range(0, len(ready), args.chunk)]
    for start, part in chunks:
        tag = f"chunk_{start:06d}"
        out = SCORES / f"{tag}.csv"
        if out.exists():
            log(f"[score] {tag} already scored; skipping")
            continue
        need = len(part) * SEC_EXTRACT + 180
        if time_left() < need + 10 * 60:
            log(f"[score] stopping before {tag}: needs ~{hms(need)}, "
                f"{hms(time_left())} left. Re-run with this output attached.")
            break
        require_disk(len(part) * BYTES_FEAT, f"features for {tag}")

        meta_csv = META / f"{tag}_meta.csv"
        with open(meta_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label"])
            w.writeheader()
            for rc in part:
                w.writerow({"path": rc["flat"], "label": rc["label"]})

        run([sys.executable, "deepfake_feature_extraction.py",
             "--dataset", "AV1M", "--split", "test",
             "--metadata", meta_csv,
             "--ckpt_path", "self_large_vox_433h.pt",
             "--data_path", PRE,
             "--save_path", FEATS], cwd=AVHUBERT)

        made = len(list((FEATS / "val").glob("*.npz"))) if (FEATS / "val").exists() else 0
        log(f"[score] {tag}: {made} feature files present")
        if made == 0:
            raise SystemExit(f"{tag} produced no features -- check the accelerator")

        run([sys.executable, "score_clips.py", meta_csv, FEATS / "val", out]
            + [f"{n}={p}" for n, p in ck.items()], cwd=REPO)
        shutil.rmtree(FEATS / "val", ignore_errors=True)
        disk_report()
        budget_report(f"after {tag}")

    scored = sorted(SCORES.glob("chunk_*.csv"))
    n_done = sum(sum(1 for _ in csv.DictReader(open(f))) for f in scored)
    log(f"[score] {n_done}/{len(TEST_RECS)} clips scored across {len(scored)} chunks")

    if n_done >= len(ready) and ready:
        (REPO / "aggregate_scores.py").write_text(AGGREGATE)
        log("=== FULL TEST SPLIT RESULTS ===")
        run([sys.executable, "aggregate_scores.py", SCORES], cwd=REPO, check=False)
    else:
        log("[score] more chunks to go -- attach this output to the next session")


if MODE == "preprocess":
    run_stage("preprocess", stage_prep)
elif MODE == "score":
    run_stage("eval", stage_score)
else:
    raise SystemExit(f"unknown MODE {MODE!r}; use 'preprocess' or 'score'")


# CELL 9 - Session summary
#
# Prints what this session added and what the next one should do, so a multi-
# session run never depends on remembering where it left off.

n_roi = len(roi_done(TEST_RECS))
n_scored = sum(sum(1 for _ in csv.DictReader(open(f))) for f in sorted(SCORES.glob("chunk_*.csv"))) \
    if SCORES.exists() else 0

print("=================================")
print("AVH-Align on the full LAV-DF test split")
print("=================================")
print(f"MODE this session      : {MODE}")
print(f"clips in the split     : {len(TEST_RECS)}")
print(f"mouth ROIs available   : {n_roi}")
print(f"clips scored           : {n_scored}")
disk_report()
budget_report("session end")
print()
if n_roi < len(TEST_RECS):
    print(f"NEXT: another MODE='preprocess' session (accelerator None), with this")
    print(f"      output and every earlier ROI output attached. {len(TEST_RECS) - n_roi} clips to go.")
elif n_scored < len(TEST_RECS):
    print("NEXT: MODE='score' on GPU T4 x2, with every ROI output attached")
    print("      plus this output if any chunks were already scored.")
else:
    print("DONE: AP / AUC with bootstrap CIs are printed above, on all")
    print(f"      {len(TEST_RECS)} test clips at the dataset's own class prior.")
print("=================================")
