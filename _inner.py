
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK = Path("/kaggle/working")
REPO = WORK / "AVH-Align"
AVHUBERT = WORK / "av_hubert" / "avhubert"

LINKS = WORK / "lavdf_root"
META = WORK / "lavdf_meta"
PRE = WORK / "lavdf_pre"   # persisted in output so preprocess survives a crash
FEATS = WORK / "lavdf_feats"
CKPT = WORK / "checkpoints"
STATE = WORK / "state"
SCORES = WORK / "scores"

ALL_STAGES = ["setup", "metadata", "preprocess", "extract", "train", "eval", "probe"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd=None, check=True):
    log(f"$ {' '.join(str(c) for c in cmd)}" + (f"   (cwd={cwd})" if cwd else ""))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed with code {r.returncode}")
    return r.returncode


def done(stage):
    return (STATE / f"{stage}.done").exists()


def mark(stage):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{stage}.done").write_text(time.ctime())


def disk_report():
    total, used, free = shutil.disk_usage(WORK)
    log(f"disk /kaggle/working: {used/2**30:.1f} GiB used, {free/2**30:.1f} GiB free")


def find_input_dir(name, must_have=(), root=Path("/kaggle/input")):
    """Locate a directory called `name` inside any attached input (dataset or
    notebook output), 1-4 levels deep, that also contains every relative path in
    `must_have`. Lets a run reuse a repo checkout persisted by an earlier run
    when GitHub is unreachable from the Kaggle worker."""
    for pat in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}", f"*/*/*/*/{name}"):
        for c in sorted(root.glob(pat)):
            if c.is_dir() and all((c / m).exists() for m in must_have):
                return c
    return None


def fetch_repo(dst, url, must_have=(), submodules=False, attempts=3):
    """git clone `url` into `dst`, retrying on transient network failures. If
    GitHub stays unreachable, fall back to a copy of the same checkout found in
    an attached input (the V8-output dataset carries AVH-Align and av_hubert).
    Every later source patch is idempotent, so a pre-patched copy is fine."""
    dst = Path(dst)
    if dst.exists():
        return
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for i in range(attempts):
        cmd = ["git", "clone"] + ([] if submodules else ["--depth", "1"]) + [url, str(dst)]
        log(f"$ {' '.join(cmd)}   (attempt {i + 1}/{attempts})")
        r = subprocess.run(cmd, env=env)
        if r.returncode == 0:
            if submodules:
                run(["git", "submodule", "init"], cwd=dst)
                run(["git", "submodule", "update"], cwd=dst)
            return
        shutil.rmtree(dst, ignore_errors=True)
        if i + 1 < attempts:
            time.sleep(20)
    src = find_input_dir(dst.name, must_have)
    if src is None:
        raise SystemExit(f"git clone of {url} failed {attempts}x and no copy of "
                         f"{dst.name}/ with {list(must_have)} is attached as an input")
    log(f"GitHub unreachable -> copying {dst.name} from attached input {src}")
    shutil.copytree(src, dst, symlinks=True)
    log(f"copied {dst.name}: {sum(1 for _ in dst.rglob('*'))} entries")


def restore_checkpoint():
    """Resume across sessions. If a prior run's output is attached as an input
    dataset, reuse its expensive DATA stages (preprocessed mouth ROIs, extracted
    AV-HuBERT features) instead of redoing hours of work. setup/metadata always
    re-run (cheap; keeps every source patch current). A stage is marked done ONLY
    when its data is actually present -- a bare .done marker is never trusted
    (a marker without data is exactly what a crash after preprocess leaves behind
    when the data lived on ephemeral scratch)."""
    # Kaggle mounts inputs at /kaggle/input/<kind>/<owner>/<slug>/ (datasets,
    # notebook outputs), older mounts at /kaggle/input/<slug>/ -- so look 1-4
    # levels deep for a directory that holds lavdf_pre / lavdf_feats. Bounded
    # depth keeps this instant even with the 136k-file LAV-DF dataset mounted.
    inp = Path("/kaggle/input")
    # Notebook-output mounts can sit 5+ levels deep (e.g. .../<slug>/versions/8/),
    # so walk instead of globbing a fixed depth. Prune hard: never descend into
    # the raw LAV-DF video tree or any directory with >2000 entries, and never
    # into a hit itself -- the walk stays at a few hundred stat calls.
    bases, seen = [], set()
    root = str(inp)
    for cur, dirs, files in os.walk(root, topdown=True):
        depth = cur[len(root):].count(os.sep)
        hit = False
        for name in ("lavdf_pre", "lavdf_feats"):
            if name in dirs and (Path(cur) / name).is_dir():
                if cur not in seen:
                    bases.append(Path(cur)); seen.add(cur)
                hit = True
        if hit or depth >= 6 or len(dirs) + len(files) > 2000:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in ("LAV-DF", "lavdf_pre", "lavdf_feats")]
    # A trained alignment model is tiny and always worth reusing: with it in
    # place the train stage is skipped and a session can spend its whole budget
    # on a larger evaluation set.
    ck = find_input_dir("checkpoints", (f"{CFG.name}.pt",))
    if ck is not None and not (CKPT / f"{CFG.name}.pt").exists():
        CKPT.mkdir(parents=True, exist_ok=True)
        shutil.copy(ck / f"{CFG.name}.pt", CKPT / f"{CFG.name}.pt")
        mark("train")
        log(f"[resume] trained checkpoint copied from {ck} -> train stage will be skipped")

    if not CFG.resume_data:
        # Reusing ROIs/features would also pin the clip selection they were built
        # from. When the point of the run is a *different* selection (a larger
        # test set, say), the data stages must run fresh -- but the trained
        # checkpoint above is still reused, so no GPU-hours are wasted.
        log("[resume] resume_data=False -> ignoring any ROIs/features under /kaggle/input")
        return

    if not bases:
        log("[resume] no lavdf_pre / lavdf_feats found under /kaggle/input -> full run")
        # Show what IS mounted so a miss is diagnosable from the log alone.
        try:
            for d in sorted(inp.glob("*")):
                sub = sorted(x.name for x in d.iterdir())[:8] if d.is_dir() else []
                log(f"[resume] /kaggle/input/{d.name}/ -> {sub}")
                for e in sub[:4]:
                    ee = d / e
                    if ee.is_dir():
                        log(f"[resume]   {e}/ -> {sorted(x.name for x in ee.iterdir())[:8]}")
        except Exception as ex:
            log(f"[resume] listing failed: {ex}")
        return
    log(f"[resume] candidate dirs: {[str(b) for b in bases]}")
    for base in bases:
        for name, dst in (("lavdf_pre", PRE), ("lavdf_feats", FEATS)):
            src = base / name
            if src.is_dir() and any(src.iterdir()) and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)
                log(f"[resume] linked {name} <- {src}")
        # Reuse the split CSVs that lavdf_pre was built from, so the metadata
        # stage cannot drift from the preprocessed clips (it is deterministic
        # anyway; this removes the assumption).
        src_meta = base / "lavdf_meta"
        need = ("train_metadata.csv", "val_metadata.csv",
                "test_metadata.csv", "trainval_metadata.csv")
        if (src_meta.is_dir() and all((src_meta / n).exists() for n in need)
                and PRE.exists() and any(PRE.iterdir()) and not done("metadata")):
            META.mkdir(parents=True, exist_ok=True)
            for n in need:
                shutil.copy(src_meta / n, META / n)
            mark("metadata")
            log(f"[resume] split CSVs copied from {src_meta} -> metadata stage will be skipped")
    if PRE.exists() and any(PRE.iterdir()):
        mark("preprocess")
        log("[resume] preprocess data present -> stage will be skipped")
    if FEATS.exists() and any(FEATS.rglob("*.npz")):
        mark("extract")
        log("[resume] feature data present -> stage will be skipped")


T0 = time.time()
DEADLINE = None
BYTES_PER_CLIP = None


def est_feature_bytes(n_clips, frames=190):
    # 190 frames/clip is calibrated from a measured run (6.12 GiB of .npz for
    # 4300 clips = 1.46 MiB/clip); the old default of 100 underestimated the
    # feature footprint by 2x and made the disk guard useless.
    return n_clips * frames * 2 * 1024 * 4


def time_left():
    return DEADLINE - time.time() if DEADLINE else float("inf")


def hms(sec):
    sec = max(0, int(sec))
    return f"{sec//3600}h{(sec%3600)//60:02d}m"


def budget_report(label=""):
    log(f"[budget] elapsed {hms(time.time()-T0)} | remaining {hms(time_left())} {label}")


def require_time(need_sec, what):

    if time_left() < need_sec:
        raise SystemExit(
            f"STOPPING BEFORE {what}: needs ~{hms(need_sec)} but only {hms(time_left())} "
            f"left in the budget.\nCompleted stages are marked -- rerun the same command "
            f"in a fresh session to resume.")


def free_bytes(path=WORK):
    return shutil.disk_usage(path).free


def require_disk(need_bytes, what, path=WORK):
    free = free_bytes(path)
    log(f"[disk] {what} needs ~{need_bytes/2**30:.1f} GiB, {free/2**30:.1f} GiB free at {path}")
    if free < need_bytes * 1.25:
        raise SystemExit(
            f"NOT ENOUGH DISK for {what}: needs ~{need_bytes/2**30:.1f} GiB "
            f"(+25% margin), only {free/2**30:.1f} GiB free.\n"
            f"Lower --max_train / --max_test and rerun.")


def stage_setup(args):

    WORK.mkdir(parents=True, exist_ok=True)

    # The torch build in this image supports sm_70 and newer (T4, V100, A100,
    # L4). Kaggle's P100 is sm_60: .cuda() succeeds but every kernel launch
    # raises, and the upstream extraction script swallows that in a bare
    # `except:` -- so all 4300 clips report "Unprocessed" and produce nothing
    # after hours of work (exactly how the 2026-09-03 run died). Check it here,
    # where the run has cost two minutes instead of five hours.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch\n"
         "if not torch.cuda.is_available():\n"
         "    print('GPU none')\n"
         "else:\n"
         "    print('GPU', torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0))"],
        capture_output=True, text=True)
    line = (probe.stdout or "").strip().splitlines()[-1] if probe.stdout.strip() else "GPU unknown"
    log(f"[gpu] {line}")
    parts = line.split()
    if len(parts) >= 3 and parts[-2].isdigit():
        major = int(parts[-2])
        if major < 7:
            raise SystemExit(
                f"UNUSABLE ACCELERATOR: {' '.join(parts[1:-2])} is compute capability "
                f"{parts[-2]}.{parts[-1]}; this torch build supports 7.0 and newer.\n"
                "Feature extraction would silently produce zero features.\n"
                "Fix: Settings -> Accelerator -> GPU T4 x2, then re-run.")
    elif line == "GPU none":
        raise SystemExit("no CUDA device visible -- enable a GPU accelerator and re-run")

    # Clone with retries; if GitHub is unreachable from the worker (V4 died on
    # "could not read Username for 'https://github.com'"), reuse the checkout
    # persisted in the attached V8-output dataset instead.
    fetch_repo(REPO, "https://github.com/bit-ml/AVH-Align.git",
               must_have=("train.py", "eval.py", "deepfake_preprocess.py"))
    fetch_repo(WORK / "av_hubert", "https://github.com/facebookresearch/av_hubert.git",
               must_have=("avhubert/hubert.py", "fairseq/fairseq/checkpoint_utils.py"),
               submodules=True)

    HEAL = None
    if not args.skip_pip:


        pins = [
            "numpy<2", "omegaconf==2.0.6", "hydra-core==1.0.7",
            "sacrebleu<2.0", "bitarray", "editdistance",
            "python_speech_features", "scikit-video", "librosa",
            "dlib-bin", "cython<3", "wheel",
            "setuptools<70",
        ]
        run([sys.executable, "-m", "pip", "install", "-q", "pip<24.1"])
        run([sys.executable, "-m", "pip", "install", "-q", *pins])
        fsq = WORK / "av_hubert" / "fairseq"
        import site, re as _re
        sp = next(pp for pp in site.getsitepackages() if "packages" in pp)
        (Path(sp) / "fairseq_local.pth").write_text(str(fsq) + "\n")
        log("fairseq installed by .pth path (pip C build skipped: libnat_cuda "
            "needs THC/THC.h, removed from torch -- unbuildable and unneeded)")

        # torch >= 2.6 defaults weights_only=True, which breaks loading the 2021
        # AV-HuBERT checkpoint (pickled fairseq Dictionary / argparse objects).
        # V8 lesson: a sitecustomize.py in site-packages is NEVER imported here --
        # Debian ships /usr/lib/python3.x/sitecustomize.py earlier on sys.path and
        # Python loads only the first one. Three independent layers instead:
        #   1. env var TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in this process, inherited
        #      by every subprocess (torch honours it whenever weights_only is not
        #      passed explicitly);
        #   2. a .pth "import" line, executed at site init by every interpreter that
        #      uses this site-packages, setting the same env var;
        #   3. every single-line torch.load(...) in fairseq + avhubert rewritten to
        #      pass weights_only=False explicitly, so the fix does not depend on the
        #      environment at all.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        (Path(sp) / "sitecustomize.py").unlink(missing_ok=True)
        (Path(sp) / "torch_weights_only_off.pth").write_text(
            "import os; os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')\n")
        nwo = 0
        for f in list(fsq.rglob("*.py")) + list(AVHUBERT.glob("*.py")):
            try:
                t = f.read_text()
            except Exception:
                continue
            t2 = _re.sub(r"torch\.load\(((?:[^()]|\([^()]*\))*)\)",
                         lambda m: m.group(0) if "weights_only" in m.group(1)
                         else f"torch.load({m.group(1)}, weights_only=False)", t)
            if t2 != t:
                f.write_text(t2)
                nwo += 1
        log(f"torch.load weights_only=False: env var + .pth hook set, "
            f"{nwo} fairseq/avhubert files patched explicitly")

        # ---- self-healing dataclass patch ---------------------------------
        # fairseq afc77bd + hydra 1.0.7 + omegaconf 2.0.6 predate py3.11's strict
        # dataclass rule (mutable default `x: T = T()` must use default_factory).
        # Rather than guess every offending module, loop: try `import fairseq`,
        # and whenever it dies on a mutable-default error, read the exact file from
        # the traceback, wrap that pattern, and retry. This walks fairseq -> hydra
        # -> omegaconf -> whatever is next, automatically.
        DC_PAT = _re.compile(r"(\w+): ([A-Za-z_][\w.]*) = \2\(\)")
        FROM_DC = _re.compile(r"from dataclasses import (?![^\n]*\bfield\b)")
        NP_ALIAS = {"float": "float", "int": "int", "bool": "bool",
                    "object": "object", "str": "str", "complex": "complex",
                    "long": "int", "unicode": "str"}
        ABC_NAMES = ("Mapping", "MutableMapping", "Sequence", "MutableSequence",
                     "Set", "MutableSet", "Iterable", "Callable", "Hashable",
                     "Container", "Sized")

        def _patch_file(fp):
            # Apply every known 2021-code -> modern-runtime fix and report if the
            # file changed. Covers: py3.12 mutable-default dataclasses, numpy>=1.24
            # removed aliases (np.float/int/bool/object/str/...), and py3.10 moved
            # collections ABCs. Idempotent: re-running makes no further change.
            try:
                t = fp.read_text()
            except Exception:
                return False
            o = t
            t = DC_PAT.sub(r"\1: \2 = field(default_factory=\2)", t)
            if t != o and "field(default_factory=" in t:
                if "from dataclasses import" in t and FROM_DC.search(t):
                    t = FROM_DC.sub("from dataclasses import field, ", t, count=1)
                elif "from dataclasses import" not in t:
                    t = "from dataclasses import field\n" + t
            for pfx in ("np", "numpy"):
                for k, v in NP_ALIAS.items():
                    t = _re.sub(r"\b" + pfx + r"\." + k + r"\b", v, t)
            for nm in ABC_NAMES:
                t = _re.sub(r"\bcollections\." + nm + r"\b",
                            "collections.abc." + nm, t)
            if t == o:
                return False
            fp.write_text(t)
            return True

        n0 = 0
        for root in (fsq / "fairseq", WORK / "av_hubert" / "avhubert",
                     Path(sp) / "hydra", Path(sp) / "omegaconf"):
            for f in root.rglob("*.py"):
                if _patch_file(f):
                    n0 += 1
        log(f"default_factory pre-patch: {n0} files")

        # av_hubert's avhubert/*.py use package-relative imports (e.g.
        # "from .hubert_dataset import AVHubertDataset"). The deepfake_* entry
        # scripts run as top-level scripts (cwd=avhubert) and do absolute
        # "import hubert_pretraining", so those relative imports raise
        # "attempted relative import with no known parent package". avhubert is
        # on sys.path via cwd, so rewrite relative -> absolute.
        nrel = 0
        for f in AVHUBERT.glob("*.py"):
            t = o2 = f.read_text()
            t = _re.sub(r'(?m)^(\s*)from \. import ', r'\1import ', t)
            t = _re.sub(r'(?m)^(\s*)from \.([A-Za-z_])', r'\1from \2', t)
            if t != o2:
                f.write_text(t)
                nrel += 1
        log(f"rel-import fix: {nrel} avhubert modules -> absolute imports")

        # THE CORE CONTRADICTION (verified): py3.12 forbids `x: T = T()` mutable
        # defaults -> we rewrite them to field(default_factory=T). But omegaconf
        # 2.0.6's OmegaConf.structured() reads field.default, which for a
        # default_factory field is dataclasses.MISSING -> "_MISSING_TYPE" error.
        # These cannot both be satisfied by patching. structured() is only called
        # from fairseq.hydra_init(), which registers configs for the hydra CLI we
        # never use -- feature extraction calls fairseq APIs directly and reads the
        # checkpoint's cfg as DATA. So we neuter hydra_init: the schemas still
        # define (default_factory), structured() is never invoked on them.
        finit = fsq / "fairseq" / "__init__.py"
        it = finit.read_text()
        if "hydra_init()" in it and "FAIRSEQ_HYDRA_INIT" not in it:
            it = it.replace(
                "hydra_init()",
                "import os as _os\n"
                "if _os.environ.get('FAIRSEQ_HYDRA_INIT') == '1':\n"
                "    hydra_init()", 1)
            finit.write_text(it)
            log("neutered fairseq.hydra_init() (avoids omegaconf structured()/MISSING)")

        # Exclude the *raising* machinery so the picker targets the caller
        # (the file that actually uses the bad construct), not the module
        # that raised (numpy/__init__.py __getattr__, dataclasses.py, ...).
        _STDLIB = ("/dataclasses.py", "/enum.py", "/typing.py",
                   "/functools.py", "/numpy/__init__.py", "/abc.py",
                   "/collections/__init__.py", "/re/__init__.py")
        # Deeper than `import fairseq`: exercise the actual extraction path so a
        # green light really means the pipeline can load models, not just import.
        # Also import the avhubert model modules the extraction script needs,
        # with cwd=avhubert exactly like the real run -- so a broken import
        # chain fails here in setup (minutes), not after hours of preprocess.
        # The torch.save/torch.load round trip of a fairseq Dictionary is exactly
        # what checkpoint_utils.load_checkpoint_to_cpu does on the 5.4 GB AV-HuBERT
        # file; V8 only discovered it was broken 5 hours in. Now it is a 6-minute
        # setup failure instead.
        CHECK = ("import fairseq; "
                 "from fairseq import checkpoint_utils, tasks, utils; "
                 "import omegaconf, hydra; "
                 "import hubert_pretraining, hubert, hubert_asr; "
                 "import torch, tempfile, os; "
                 "from fairseq.data.dictionary import Dictionary; "
                 "p = tempfile.mktemp(suffix='.pt'); "
                 "torch.save({'d': Dictionary(), 'cfg': {'x': 1}}, p); "
                 "torch.load(p, map_location='cpu'); os.remove(p); "
                 "print('imports ok, torch.load default-args ok:', omegaconf.__version__)")
        def heal(check_code, label):
            for attempt in range(30):
                r = subprocess.run([sys.executable, "-c", check_code],
                                   capture_output=True, text=True, cwd=str(AVHUBERT))
                if r.returncode == 0:
                    out = r.stdout.strip().splitlines()
                    log(out[-1] if out else f"{label} ok")
                    return
                err = r.stderr
                paths = _re.findall(r'File "([^"]+\.py)"', err)
                target = None
                for pp in reversed(paths):
                    if any(sfx in pp for sfx in _STDLIB):
                        continue
                    target = pp
                    break
                # With cwd=avhubert a traceback may name a module by a relative
                # path; resolve it against AVHUBERT before patching.
                tp = Path(target) if target else None
                if tp is not None and not tp.is_absolute():
                    tp = AVHUBERT / tp
                if tp is None or not tp.exists() or not _patch_file(tp):
                    log(f"cannot auto-fix {label} (attempt {attempt}); last error:")
                    print(err[-3000:], flush=True)
                    break
                log(f"patched {tp} (attempt {attempt}); retrying {label}")
            raise SystemExit(f"{label} could not be healed -- see errors above")

        heal(CHECK, "fairseq/avhubert import check")
        HEAL = heal


    misc = AVHUBERT / "content" / "data" / "misc"
    misc.mkdir(parents=True, exist_ok=True)

    def stage_asset(src_glob, dest, url, rel, is_bz2=False):


        if dest.exists() and dest.stat().st_size > 1024:
            log(f"already present: {dest.name} ({dest.stat().st_size/2**20:.1f} MiB)")
            return
        # Look for a copy in an attached input before downloading. Both lookups
        # are bounded-depth: `*/**/name` would recursively walk the 136k-file
        # LAV-DF mount and can take minutes per asset for nothing.
        base = find_input_dir("av_hubert", (rel,))
        if base is not None:
            hits = [base / rel]
        else:
            name = src_glob.rsplit("/", 1)[-1]
            hits = [h for pat in (f"*/{name}", f"*/*/{name}", f"*/*/*/{name}",
                                  f"*/*/*/*/{name}", f"*/*/*/*/*/{name}")
                    for h in sorted(Path("/kaggle/input").glob(pat)) if h.is_file()]
        if hits:
            log(f"staging {hits[0]} -> {dest}")
            shutil.copy(hits[0], dest)
            return
        log(f"downloading {url}")
        if is_bz2:
            tmp = Path("/tmp/_asset.bz2")
            run(["wget", "-q", url, "-O", str(tmp)])
            import bz2 as _bz2
            dest.write_bytes(_bz2.decompress(tmp.read_bytes()))
            tmp.unlink()
        else:
            run(["wget", "-q", url, "-O", str(dest)])
        if not dest.exists() or dest.stat().st_size < 1024:
            raise SystemExit(f"download failed or truncated: {dest}")
        log(f"got {dest.name} ({dest.stat().st_size/2**20:.1f} MiB)")

    stage_asset("*/**/shape_predictor_68_face_landmarks.dat",
                misc / "shape_predictor_68_face_landmarks.dat",
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                rel="avhubert/content/data/misc/shape_predictor_68_face_landmarks.dat",
                is_bz2=True)
    stage_asset("*/**/20words_mean_face.npy", misc / "20words_mean_face.npy",
                "https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks"
                "/raw/master/preprocessing/20words_mean_face.npy",
                rel="avhubert/content/data/misc/20words_mean_face.npy")
    stage_asset("*/**/self_large_vox_433h.pt", AVHUBERT / "self_large_vox_433h.pt",
                "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/"
                "self_large_vox_433h.pt",
                rel="avhubert/self_large_vox_433h.pt")

    # Load the real 5.4 GB checkpoint through the exact fairseq call the
    # extraction script makes (checkpoint_utils.load_model_ensemble_and_task).
    # Extraction has never run end-to-end in this environment (V8 died before
    # it), so its riskiest step must fail here, in setup, not hours later.
    if HEAL is not None:
        HEAL("from fairseq import checkpoint_utils; "
             "import hubert_pretraining, hubert, hubert_asr; "
             "m, _, t = checkpoint_utils.load_model_ensemble_and_task(['self_large_vox_433h.pt']); "
             "print('checkpoint load ok:', type(m[0]).__name__, 'crop', t.cfg.image_crop_size)",
             "AV-HuBERT checkpoint load check")


    for f in ("deepfake_preprocess.py", "deepfake_feature_extraction.py"):
        shutil.copy(REPO / f, AVHUBERT / f)


    fe = AVHUBERT / "deepfake_feature_extraction.py"
    src_fe = fe.read_text()
    subs = [
        ('f_mm, _ = model.extract_finetune({"video": frames, "audio": audio}, None, None)',
         'f_mm = None  # [patched] never read downstream'),
        ('f_mm.squeeze(0).cpu().numpy()', 'None'),
        ('"multimodal": feature_multimodal,', '# [patched] multimodal dropped'),
    ]
    applied = 0
    for old, new in subs:
        if old in src_fe:
            src_fe = src_fe.replace(old, new)
            applied += 1
    fe.write_text(src_fe)
    if applied == len(subs):
        log("patched feature extraction: multimodal pass removed "
            "(~33% less GPU time and feature disk)")
    else:
        log(f"WARNING: multimodal patch applied {applied}/{len(subs)} substitutions -- "
            f"upstream may have changed. Continuing with the unpatched behaviour; "
            f"expect ~50% more feature disk than estimated.")


    if not Path("/usr/bin/ffmpeg").exists():
        found = shutil.which("ffmpeg")
        if not found:
            raise SystemExit("ffmpeg not found. Run: !apt-get -qq install -y ffmpeg")
        log(f"symlinking {found} -> /usr/bin/ffmpeg (hardcoded in the repo)")
        os.symlink(found, "/usr/bin/ffmpeg")

    disk_report()
    mark("setup")


def find_lavdf_metadata(root: Path) -> Path:
    # First honour an explicit --lavdf_root if it actually holds the metadata.
    for name in ("metadata.json", "metadata.min.json"):
        for c in [root / name, *root.glob(f"*/{name}"), *root.glob(f"*/*/{name}")]:
            if c.is_file():
                return c
    # Otherwise discover it anywhere under /kaggle/input, regardless of the
    # dataset's mount slug (which varies by owner). Bounded-depth glob = fast;
    # the runner script's own dir is skipped.
    inp = Path("/kaggle/input")
    if inp.exists():
        pats = []
        for name in ("metadata.json", "metadata.min.json"):
            pats += [f"*/{name}", f"*/*/{name}", f"*/*/*/{name}", f"*/*/*/*/{name}"]
        for pat in pats:
            for c in sorted(inp.glob(pat)):
                if "avhalign" in str(c).lower():
                    continue
                if c.is_file():
                    return c
    listing = []
    if inp.exists():
        for d in sorted(inp.iterdir()):
            try:
                sub = sorted(x.name for x in d.iterdir())[:8]
            except Exception:
                sub = []
            listing.append(f"{d.name} -> {sub}")
    raise SystemExit("No metadata.json/metadata.min.json found under /kaggle/input.\n"
                     + "\n".join(listing))


def stage_metadata(args):

    random.seed(args.seed)
    lavdf = Path(args.lavdf_root)
    META.mkdir(parents=True, exist_ok=True)

    meta_path = find_lavdf_metadata(lavdf)
    log(f"LAV-DF metadata: {meta_path}")
    video_root = meta_path.parent

    with open(meta_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("clips", list(data.values()))
    log(f"{len(data)} records; sample keys: {list(data[0])}")

    recs = []
    for r in data:
        rel = r.get("file") or r.get("path") or r.get("filename")
        if rel is None:
            raise SystemExit(f"record has no file key; keys={list(r)}")
        nf = r.get("video_frames") or r.get("n_frames") or r.get("num_frames")
        if not nf:
            continue
        mv = bool(r.get("modify_video", False))
        ma = bool(r.get("modify_audio", False))
        nfakes = r.get("n_fakes", len(r.get("fake_periods") or []))
        if int(nf) < 31:
            continue
        recs.append({
            "rel": str(rel),
            "num_frames": int(nf),
            "split": r.get("split", ""),
            "label": 0 if (not mv and not ma and nfakes == 0) else 1,
        })
    log(f"{len(recs)} usable records after filtering")


    seen = {}
    for rc in recs:
        base = Path(rc["rel"]).name
        if base in seen and seen[base] != rc["rel"]:
            base = Path(rc["rel"]).parent.name + "_" + base
        seen[base] = rc["rel"]
        rc["flat"] = base

    def pick(pool, n):
        pool = list(pool)
        random.shuffle(pool)
        return pool if (n <= 0 or n >= len(pool)) else pool[:n]

    probe_rows = []          # train+val clips WITH labels (both classes) for the supervised probe
    if getattr(args, "protocol", "lavdf") == "shared1000":
        # Reviewers' protocol: one seeded draw of n_pool clips from the WHOLE dataset
        # (all official splits, both classes), split n_train / n_val / rest. If
        # split_file is given (path,split[,label]) that exact list is used instead,
        # so the clips are identical to the other methods' -- the only fair setup.
        by_name = {}
        for rc in recs:
            by_name.setdefault(Path(rc["rel"]).name, rc)
            by_name.setdefault(rc["rel"], rc)
        if getattr(args, "split_file", ""):
            chosen = []
            with open(args.split_file) as f:
                for row in csv.DictReader(f):
                    key = row.get("path") or row.get("file") or row.get("filename")
                    rc = by_name.get(key) or by_name.get(Path(key).name)
                    if rc is None:
                        log(f"[split_file] not in usable LAV-DF records (<31 frames?): {key}")
                        continue
                    rc = dict(rc); rc["proto"] = row["split"].strip().lower()
                    chosen.append(rc)
            log(f"[protocol] split_file {args.split_file}: {len(chosen)} clips matched")
        elif getattr(args, "balanced_splits", True):
            # Class-balanced draw: every split is 50/50 real/fake (train 300+300,
            # val 100+100, test 100+100 by default). LAV-DF is 73% fake, so an
            # unbalanced draw would leave AVH-Align only ~170 real training clips.
            chosen = []
            for label in (0, 1):
                pool = sorted((r for r in recs if r["label"] == label), key=lambda r: r["rel"])
                random.shuffle(pool)              # seeded above (args.seed)
                n_tr, n_va = args.n_train // 2, args.n_val // 2
                n_te = args.n_pool // 2 - n_tr - n_va
                take = [dict(rc) for rc in pool[:n_tr + n_va + n_te]]
                for i, rc in enumerate(take):
                    rc["proto"] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
                chosen += take
            log(f"[protocol] shared1000 balanced: seed {args.seed}, {len(chosen)} clips, each split "
                f"50/50 real/fake (NOT the reviewers' exact clips -- pass split_file for that)")
        else:
            pool = sorted(recs, key=lambda r: r["rel"])
            random.shuffle(pool)                  # seeded above (args.seed)
            chosen = [dict(rc) for rc in pool[:args.n_pool]]
            for i, rc in enumerate(chosen):
                rc["proto"] = ("train" if i < args.n_train else
                               "val" if i < args.n_train + args.n_val else "test")
            log(f"[protocol] shared1000: seed {args.seed} draw of {len(chosen)} clips from "
                f"{len(pool)} usable records (NOT necessarily the reviewers' exact clips -- "
                f"pass split_file for that)")
        tr_all = [r for r in chosen if r["proto"] == "train"]
        va_all = [r for r in chosen if r["proto"] == "val"]
        test = [r for r in chosen if r["proto"] == "test"]
        train_all = bool(getattr(args, "avh_train_all", False))
        if train_all:
            # Reviewers' request (2026-09-05): the alignment head sees every train clip, real AND fake,
            # and early-stops on every val clip. Fake clips enter the alignment objective as if aligned.
            train, val = list(tr_all), list(va_all)
        else:
            train = [r for r in tr_all if r["label"] == 0]
            val = [r for r in va_all if r["label"] == 0]
        probe_rows = tr_all + va_all
        for name_, grp in (("train", tr_all), ("val", va_all), ("test", test)):
            nf = sum(r["label"] for r in grp)
            log(f"[protocol] {name_}: {len(grp)} clips = {len(grp) - nf} real / {nf} fake")
        if train_all:
            log(f"[protocol] avh_train_all=True: AVH-Align trains on ALL {len(train)} train clips "
                f"({len(train) - sum(r['label'] for r in train)} real / {sum(r['label'] for r in train)} fake), "
                f"validates on all {len(val)} val clips; the supervised probe uses all {len(probe_rows)}")
        else:
            log(f"[protocol] AVH-Align trains on the {len(train)} real clips of train, validates on "
                f"the {len(val)} real clips of val; the supervised probe uses all {len(probe_rows)}")
        with open(META / "shared1000_split.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "split", "label"])
            for r in chosen:
                w.writerow([r["rel"], r["proto"], r["label"]])
        random.shuffle(test)
        assert train_all or all(r["label"] == 0 for r in train + val)
        for sub in ("train", "val"):
            (LINKS / sub).mkdir(parents=True, exist_ok=True)
    else:
        train = val = test = None
    real = [r for r in recs if r["label"] == 0]
    tr_pool = [r for r in real if r["split"] in ("train", "")]
    va_pool = [r for r in real if r["split"] in ("dev", "val")]
    if not va_pool:
        log("no dev/val split found -- carving 10% out of train")
        cut = max(1, len(tr_pool) // 10)
        random.shuffle(tr_pool)
        va_pool, tr_pool = tr_pool[:cut], tr_pool[cut:]

    if train is None:
        train = pick(tr_pool, args.max_train)
        val = pick(va_pool, args.max_val)

    te_pool = [r for r in recs if r["split"] == "test"]
    if test is not None:
        te_pool = []                      # protocol test set already fixed above
    elif not te_pool:
        chosen = {r["rel"] for r in train + val}
        te_pool = [r for r in recs if r["rel"] not in chosen]
    if test is not None:
        pass
    elif args.test_balanced:
        # 50/50 real/fake: convenient, but AP then sits on a 0.5 base rate and
        # cannot be compared with numbers published on the full test split.
        half = max(1, args.max_test // 2)
        test = (pick([r for r in te_pool if r["label"] == 0], half)
                + pick([r for r in te_pool if r["label"] == 1], half))
    elif te_pool:
        # Uniform sample of the test split, so the class prior is the dataset's
        # own. AP is then an unbiased estimate of full-split AP; AUC is prior-
        # independent either way.
        n_real = sum(1 for r in te_pool if r["label"] == 0)
        log(f"[test] LAV-DF test split: {len(te_pool)} clips, "
            f"{n_real} real / {len(te_pool) - n_real} fake "
            f"(fake prior {1 - n_real / max(1, len(te_pool)):.3f})")
        test = pick(te_pool, args.max_test)
    random.shuffle(test)

    assert getattr(args, "avh_train_all", False) or all(r["label"] == 0 for r in train + val),\
        "train/val contain fakes -- AVH-Align's objective requires real only (unless avh_train_all)"


    for sub in ("train", "val"):
        (LINKS / sub).mkdir(parents=True, exist_ok=True)

    def link(group, sub):
        n = 0
        for rc in group:
            src = video_root / rc["rel"]
            if not src.exists():
                src = lavdf / rc["rel"]
            dst = LINKS / sub / rc["flat"]
            if dst.exists() or dst.is_symlink():
                continue
            if not src.exists():
                continue
            os.symlink(src, dst)
            n += 1
        return n

    extract_rows = probe_rows if probe_rows else train + val
    log(f"symlinked {link(extract_rows, 'train')} clips -> {LINKS/'train'}")
    log(f"symlinked {link(test, 'val')} clips -> {LINKS/'val'}")


    def write_csv(path, rows, cols):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (r["flat"] if c == "path" else r[c]) for c in cols})
        log(f"wrote {path}  rows={len(rows)}")


    write_csv(META / "train_metadata.csv", train, ["path", "num_frames"])
    write_csv(META / "val_metadata.csv", val, ["path", "num_frames"])

    write_csv(META / "test_metadata.csv", test, ["path", "label"])

    write_csv(META / "trainval_metadata.csv", extract_rows, ["path", "num_frames"])
    if probe_rows:
        write_csv(META / "probe_metadata.csv", [dict(r, proto=r["proto"]) for r in probe_rows], ["path", "proto", "label"])

    n_fake = sum(r["label"] for r in test)
    n_tf, n_vf = sum(r["label"] for r in train), sum(r["label"] for r in val)
    log(f"train={len(train)} ({len(train)-n_tf} real / {n_tf} fake) | val={len(val)} ({len(val)-n_vf} real / {n_vf} fake) | "
        f"test={len(test)} ({len(test)-n_fake} real / {n_fake} fake)")
    log(f"total frames to featurise (train+val): {sum(r['num_frames'] for r in train+val):,}")
    mark("metadata")


def count_rows(csv_path):
    with open(csv_path) as f:
        return sum(1 for _ in csv.DictReader(f))


def stage_preprocess(args):

    todo = [(sp, mt) for sp, mt in (("train", "trainval_metadata.csv"),
                                    ("test", "test_metadata.csv"))
            if sp in args.data_splits]
    n_clips = sum(count_rows(META / mt) for _, mt in todo)
    log(f"[plan] preprocessing splits {[sp for sp, _ in todo]}: {n_clips} clips")


    PRE.mkdir(parents=True, exist_ok=True)
    # 0.28 MiB per clip, read off the disk deltas of a from-scratch run (1.2 GiB
    # of ROIs for 4300 clips; 0.25 MiB/clip on train, 0.31 on test). 400 kB per
    # clip leaves headroom without demanding space the stage cannot use.
    require_disk(n_clips * 400_000, "preprocessed mouth ROIs", PRE)
    # dlib runs at roughly 4 s/clip on 4 workers; refuse to start a pass that
    # cannot finish rather than burn hours and die halfway.
    require_time(n_clips * 4.0, "preprocess")

    budget_report("before preprocess")
    for split, meta in todo:
        run([sys.executable, "deepfake_preprocess.py",
             "--dataset", "AV1M",
             "--split", split,
             "--metadata", META / meta,
             "--data_path", LINKS,
             "--save_path", PRE,
             "--max_workers", str(args.workers)], cwd=AVHUBERT)
    disk_report()
    budget_report("after preprocess")
    mark("preprocess")


def stage_extract(args):

    todo = [(sp, mt) for sp, mt in (("train", "trainval_metadata.csv"),
                                    ("test", "test_metadata.csv"))
            if sp in args.data_splits]
    n_clips = sum(count_rows(META / mt) for _, mt in todo)
    need = est_feature_bytes(n_clips)
    log(f"[plan] {n_clips} clips -> ~{need/2**30:.2f} GiB of features "
        f"(visual+audio only; ~{need*1.5/2**30:.2f} GiB if the multimodal patch failed)")
    # The guard is applied per split, not for both at once: on a from-scratch
    # run the mouth ROIs of both splits are still on disk when extraction
    # starts, and a whole-run guard (+25% margin) would refuse to start a pass
    # that in fact fits, because each split's ROIs are freed as it completes.
    budget_report("before extract")

    # The extraction script writes the test split under the name "val".
    SPLIT_DIR = {"train": "train", "test": "val"}
    for split, meta in todo:

        require_disk(est_feature_bytes(count_rows(META / meta)),
                     f"AV-HuBERT features ({split})")
        run([sys.executable, "deepfake_feature_extraction.py",
             "--dataset", "AV1M",
             "--split", split,
             "--metadata", META / meta,
             "--ckpt_path", "self_large_vox_433h.pt",
             "--data_path", PRE,
             "--save_path", FEATS], cwd=AVHUBERT)

        # Count what this split actually produced BEFORE reclaiming anything.
        # The 2026-09-03 run purged on schedule while extraction was failing on
        # every clip, so 4h40m of mouth ROIs were deleted for zero features.
        # The ROIs are the expensive artifact: they are freed only once the
        # features that replace them exist.
        rows = count_rows(META / meta)
        made = (len(list((FEATS / SPLIT_DIR[split]).glob("*.npz")))
                if (FEATS / SPLIT_DIR[split]).exists() else 0)
        log(f"[extract] {split}: {made}/{rows} clips have features")
        if made == 0:
            raise SystemExit(
                f"extraction produced 0 features for split {split!r}. The mouth ROIs "
                f"are NOT deleted -- re-run with a working accelerator and the "
                f"preprocess stage will be skipped.")

        # Holding every ROI until both splits are done puts the peak (env + ROIs
        # + features) within ~1.5 GiB of the 20 GB output quota; freeing as we go
        # keeps a comfortable margin. Never touch a resumed, symlinked ROI dir.
        roi = PRE / SPLIT_DIR[split]
        if made < 0.9 * rows:
            log(f"keeping {roi}: only {made}/{rows} clips extracted, "
                f"the ROIs are still needed for a retry")
        elif args.purge_preprocessed and roi.is_dir() and not PRE.is_symlink():
            log(f"deleting {roi} to reclaim scratch")
            shutil.rmtree(roi)
            disk_report()

    n_tr = len(list((FEATS / "train").glob("*.npz"))) if (FEATS / "train").exists() else 0
    n_te = len(list((FEATS / "val").glob("*.npz"))) if (FEATS / "val").exists() else 0
    log(f"features: {n_tr} train+val, {n_te} test")
    for split, meta in (("train", "trainval_metadata.csv"), ("val", "test_metadata.csv")):
        with open(META / meta) as f:
            rows = list(csv.DictReader(f))
        miss = [r["path"] for r in rows
                if not (FEATS / split / r["path"].replace(".mp4", ".npz")).exists()]
        log(f"[extract] {split}: {len(rows) - len(miss)}/{len(rows)} clips have features"
            + (f"; first missing: {miss[:5]}" if miss else ""))
    if "train" in args.data_splits and n_tr == 0:
        raise SystemExit("no training features produced -- check the preprocess stage output")

    actual = sum(f.stat().st_size for f in FEATS.rglob("*.npz"))
    log(f"[disk] features on disk: {actual/2**30:.2f} GiB "
        f"({actual/max(1, n_tr+n_te)/2**20:.2f} MiB per clip)")

    # Never purge a resumed (symlinked, read-only input) ROI dir; only our own,
    # and only when both splits actually produced features.
    enough = all((n_tr if sp == "train" else n_te) > 0 for sp, _ in todo)
    if (args.purge_preprocessed and enough
            and PRE.exists() and not PRE.is_symlink()):
        log(f"deleting {PRE} to reclaim scratch")
        shutil.rmtree(PRE)
    disk_report()
    budget_report("after extract")
    mark("extract")


def stage_train(args):

    tp = REPO / "train.py"
    src = tp.read_text()

    if "num_workers = 32" in src:
        tp.write_text(src.replace("num_workers = 32", f"num_workers = {args.workers}"))
        log(f"patched train.py num_workers 32 -> {args.workers}")

    CKPT.mkdir(parents=True, exist_ok=True)


    reserve = 15 * 60
    cap = int(max(0, time_left() - reserve))
    require_time(10 * 60, "training")
    log(f"[budget] capping training at {hms(cap)} (reserving {hms(reserve)} for eval)")
    budget_report("before train")

    rc = run(["timeout", "-k", "60", str(cap), sys.executable, "train.py",
         "--name", args.name,
         "--data_root_path", FEATS,
         "--metadata_root_path", META,

         "--tau", "15",
         "--batch_size", "1024",
         "--learning_rate", "1e-5",
         "--early_stopping_patience", "10",
         "--scheduler_patience", "5",

         "--epochs", str(args.epochs),
         "--save_path", CKPT,
         "--use_tqdm"], cwd=REPO, check=False)
    if rc == 124:
        log("training hit the time cap -- using the best checkpoint saved so far. "
            "This is expected on a large --max_train; report the epoch count in the paper.")
    elif rc != 0:
        raise SystemExit(f"train.py failed with code {rc}")
    budget_report("after train")
    mark("train")



def stage_eval(args):

    # because they need torch, numpy and sklearn -- which the notebook kernel
    # deliberately never imports (CELL 1).

    SCORER_SRC = r"""
import csv, os, sys, numpy as np, torch
from model import FusionModel

meta, feats, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
ckpts = dict(p.split("=", 1) for p in sys.argv[4:])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
for name, path in ckpts.items():
    m = FusionModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    m.eval()
    models[name] = m

rows = list(csv.DictReader(open(meta)))
names = list(models)
kept = 0
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "label"] + [f"score_{n}" for n in names])
    for r in rows:
        f = os.path.join(feats, r["path"].replace(".mp4", ".npz"))
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        v = torch.from_numpy(d["visual"]).to(device)
        a = torch.from_numpy(d["audio"]).to(device)
        v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)
        a = a / torch.linalg.norm(a, ord=2, dim=-1, keepdim=True)
        with torch.no_grad():
            sc = [float(torch.logsumexp(-models[n](v, a), dim=0).cpu().squeeze()) for n in names]
        w.writerow([r["path"], r["label"]] + sc)
        kept += 1
print(f"scored {kept}/{len(rows)} clips -> {out_csv}")
"""

    METRICS_SRC = r"""
import csv, sys, numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: r["path"])
y = np.array([int(r["label"]) for r in rows])
names = [k[6:] for k in rows[0] if k.startswith("score_")]
S = {n: np.array([float(r["score_" + n]) for r in rows]) for n in names}

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
idx = [i for i in idx if 0 < y[i].sum() < len(i)]
print(f"clips {len(y)}  real {(y == 0).sum()}  fake {y.sum()}  fake prior {y.mean():.3f}")

for n in names:
    s = S[n]
    aps = np.array([average_precision_score(y[i], s[i]) for i in idx])
    aucs = np.array([roc_auc_score(y[i], s[i]) for i in idx])
    fpr, tpr, thr = roc_curve(y, s)
    e = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    pts = {"EER": thr[e], "Youden J": thr[int(np.argmax(tpr - fpr))]}
    pts["max F1"] = max(((precision_recall_fscore_support(
        y, (s >= t).astype(int), average="binary", zero_division=0)[2], t)
        for t in np.unique(s)))[1]
    print(f"\n=== {n} ===")
    print(f"AP  {average_precision_score(y, s):.4f} "
          f"[{np.percentile(aps, 2.5):.4f}, {np.percentile(aps, 97.5):.4f}]")
    print(f"AUC {roc_auc_score(y, s):.4f} "
          f"[{np.percentile(aucs, 2.5):.4f}, {np.percentile(aucs, 97.5):.4f}]")
    print(f"EER {(fpr[e] + (1 - tpr[e])) / 2:.4f}")
    print(f"{'operating point':<16}{'thr':>10}{'acc':>8}{'prec':>8}{'recall':>8}"
          f"{'F1':>8}{'spec':>8}   confusion")
    for lab, t in pts.items():
        pred = (s >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        print(f"{lab:<16}{t:>10.4f}{accuracy_score(y, pred):>8.4f}{p:>8.4f}{r:>8.4f}"
              f"{f:>8.4f}{tn / (tn + fp):>8.4f}   TN {tn} FP {fp} FN {fn} TP {tp}")

if len(names) == 2:
    a, b = names
    d = np.array([roc_auc_score(y[i], S[a][i]) - roc_auc_score(y[i], S[b][i]) for i in idx])
    obs = roc_auc_score(y, S[a]) - roc_auc_score(y, S[b])
    pv = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    print(f"\npaired AUC, {a} - {b}: {obs:+.4f} "
          f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
          f"p = {max(pv, 1 / len(d)):.4f}")
"""


    targets = []
    mine = CKPT / f"{args.name}.pt"
    if mine.exists():
        targets.append(("retrained on LAV-DF train split (real + fake)" if getattr(args, "avh_train_all", False)
                        else "retrained on LAV-DF real subset", mine))
    official = REPO / "checkpoints" / "AVH-Align_AV1M.pt"
    if official.exists():
        targets.append(("official AV1M checkpoint, zero-shot", official))

    if not targets:
        raise SystemExit("no checkpoint found to evaluate")

    # eval.py np.load()s one .npz per CSV row with no try/except: a single test
    # clip whose extraction failed would crash the evaluation after ~11 h of
    # work. Evaluate the rows whose features exist and log the rest.
    with open(META / "test_metadata.csv") as f:
        rows = list(csv.DictReader(f))
    avail = [r for r in rows
             if (FEATS / "val" / r["path"].replace(".mp4", ".npz")).exists()]
    n_fake = sum(int(r["label"]) for r in avail)
    log(f"test clips with features: {len(avail)}/{len(rows)} "
        f"({len(avail) - n_fake} real / {n_fake} fake); {len(rows) - len(avail)} skipped")
    if not avail or n_fake == 0 or n_fake == len(avail):
        raise SystemExit("cannot evaluate: need features for both real and fake test clips")
    eval_csv = META / "test_metadata_eval.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows({"path": r["path"], "label": r["label"]} for r in avail)

    for label, ck in targets:
        log(f"=== evaluating: {label} ===")
        run([sys.executable, "eval.py",
             "--checkpoint_path", ck,
             "--features_path", FEATS / "val",
             "--metadata", eval_csv,
             "--dataset", "LAV-DF"], cwd=REPO, check=False)

    # eval.py prints AP and AUC and nothing else, which is not enough to report
    # recall, F1 or an operating point, and not enough to test this model against
    # another one. So the same arithmetic is run again here, keeping one score per
    # clip: L2-normalise both streams, run the fusion model, take
    # logsumexp(-output). It reproduces eval.py's AP/AUC exactly, and the scores
    # it writes are what the metric suite below and compare_models.py consume.
    SCORES.mkdir(parents=True, exist_ok=True)
    (REPO / "score_clips.py").write_text(SCORER_SRC)
    (REPO / "metrics_report.py").write_text(METRICS_SRC)
    out_csv = SCORES / "test_scores.csv"
    run([sys.executable, "score_clips.py", eval_csv, FEATS / "val", out_csv]
        + [f"{'retrained' if ck == mine else 'official'}={ck}" for _, ck in targets],
        cwd=REPO, check=False)
    if out_csv.exists():
        log("=== metrics: AP / AUC / EER and accuracy, precision, recall, F1, "
            "specificity at three operating points ===")
        run([sys.executable, "metrics_report.py", out_csv], cwd=REPO, check=False)
        log(f"per-clip scores saved to {out_csv}")
    mark("eval")


def stage_probe(args):
    """Supervised row for the shared 600/200/200 protocol: a logistic-regression
    probe on FROZEN AV-HuBERT clip features, trained on all train clips (real AND
    fake), C chosen on val, scored on test. This is not AVH-Align (which has no
    supervised loss) -- it is reported as 'AV-HuBERT features + linear probe'."""
    if not (META / "probe_metadata.csv").exists():
        log("[probe] no probe_metadata.csv (protocol is not shared1000) -> skipped")
        mark("probe")
        return
    PROBE_SRC = r"""
import csv, os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
meta, feats_tr, test_meta, feats_te, out_csv = sys.argv[1:6]

def vec(path):
    d = np.load(path, allow_pickle=True); v = d["visual"].astype(np.float32); a = d["audio"].astype(np.float32)
    v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8; a /= np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    n = min(len(v), len(a)); v, a = v[:n], a[:n]
    return np.concatenate([v.mean(0), a.mean(0), np.abs(v - a).mean(0), (v * a).sum(-1).mean(keepdims=True), (v * a).sum(-1).std(keepdims=True)])

def load(meta_csv, feats, split_filter=None):
    X, y, P = [], [], []
    for r in csv.DictReader(open(meta_csv)):
        if split_filter and r.get("proto") != split_filter: continue
        f = os.path.join(feats, r["path"].replace(".mp4", ".npz"))
        if not os.path.exists(f): continue
        X.append(vec(f)); y.append(int(r["label"])); P.append(r["path"])
    return np.array(X), np.array(y), P

Xtr, ytr, _ = load(meta, feats_tr, "train"); Xva, yva, _ = load(meta, feats_tr, "val"); Xte, yte, Pte = load(test_meta, feats_te)
print(f"probe: train {len(ytr)} ({ytr.sum()} fake) val {len(yva)} ({yva.sum()} fake) test {len(yte)} ({yte.sum()} fake)")
sc = StandardScaler().fit(Xtr); best = None
for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    ap = average_precision_score(yva, clf.decision_function(sc.transform(Xva)))
    print(f"  C={C:<6} val AP {ap:.4f}")
    if best is None or ap > best[0]: best = (ap, C, clf)
ap, C, clf = best; s = clf.decision_function(sc.transform(Xte))
print(f"=== AV-HuBERT features + linear probe (C={C}, chosen on val) ===")
print(f"test AP {average_precision_score(yte, s):.4f}  AUC {roc_auc_score(yte, s):.4f}")
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "score_probe"])
    for p, l, v in zip(Pte, yte, s): w.writerow([p, l, v])
"""
    (REPO / "probe_clips.py").write_text(PROBE_SRC)
    SCORES.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "probe_clips.py", META / "probe_metadata.csv", FEATS / "train",
         META / "test_metadata.csv", FEATS / "val", SCORES / "probe_scores.csv"], cwd=REPO, check=False)
    mark("probe")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lavdf_root",
                   default="/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df")
    p.add_argument("--stages", default="all",
                   help="comma list, or 'all'. Completed stages are skipped.")
    p.add_argument("--force", default="", help="comma list of stages to re-run")
    p.add_argument("--name", default="AVH-Align_LAVDF")
    p.add_argument("--max_train", type=int, default=3000,
                   help="real training clips. Paper uses 45000. 3000 fits a 12h session\n                         with headroom; 6000 is close to the limit.")
    p.add_argument("--max_val", type=int, default=300,
                   help="real validation clips. Paper uses 5000.")
    p.add_argument("--max_test", type=int, default=1000, help="test clips, balanced")
    p.add_argument("--epochs", type=int, default=40,
                   help="upper bound; early stopping (patience 10) or the wall-clock\n                         cap usually ends it sooner")
    p.add_argument("--budget_hours", type=float, default=11.0,
                   help="wall-clock budget. Kaggle kills the session at 12h; this\n                         leaves margin to stop cleanly and save state.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocol", default="lavdf", choices=["lavdf", "shared1000"],
                   help="lavdf: real-only train/val from the official splits, balanced test "
                        "(the audited run). shared1000: one seeded draw of n_pool clips from "
                        "the whole dataset split n_train/n_val/rest, as in the reviewers' setup")
    p.add_argument("--split_file", default="", help="CSV path,split[,label] fixing the exact clips")
    p.add_argument("--avh_train_all", action=argparse.BooleanOptionalAction, default=False,
                   help="shared1000: fit the AVH-Align head on ALL train clips of the split (300 real + 300 fake)\n"
                        "                         and early-stop on all 200 val clips, instead of the real ones only.\n"
                        "                         Deviates from the paper's real-only objective; reviewers' request.")
    p.add_argument("--n_pool", type=int, default=1000)
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--balanced_splits", action=argparse.BooleanOptionalAction, default=True,
                   help="shared1000: draw each split 50/50 real/fake (300+300 / 100+100 / 100+100)")
    p.add_argument("--skip_pip", action="store_true",
                   help="skip installs if a previous session already built them")
    p.add_argument("--purge_preprocessed", action="store_true", default=True,
                   help="delete mouth-ROI mp4/wav after features exist (frees ~10GB)")
    p.add_argument("--official", default="AVH-Align_AV1M",
                   help="name of the authors' released checkpoint, if the repo ships it")
    p.add_argument("--test_balanced", action=argparse.BooleanOptionalAction, default=True,
                   help="balanced 50/50 test set; --no-test_balanced draws a uniform "
                        "sample instead, so AP sits on the dataset's own class prior")
    p.add_argument("--data_splits", default="train,test",
                   help="which splits preprocess and extract touch")
    p.add_argument("--resume_data", action=argparse.BooleanOptionalAction, default=True,
                   help="reuse mouth ROIs / features found in an attached input")
    args = p.parse_args()

    # The stage functions and restore_checkpoint() are shared verbatim with the
    # notebook, where the same settings live in a module-level CFG. Binding it
    # here keeps one copy of the code working in both contexts.
    global CFG
    CFG = args

    global DEADLINE
    DEADLINE = T0 + args.budget_hours * 3600
    log(f"[budget] wall-clock budget {args.budget_hours:.1f}h "
        f"(Kaggle hard limit is 12h); deadline at "
        f"{time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")
    est = est_feature_bytes(args.max_train + args.max_val + args.max_test)
    log(f"[plan] max_train={args.max_train} max_val={args.max_val} max_test={args.max_test} "
        f"-> ~{est/2**30:.2f} GiB of features, plus ~4 GiB env+checkpoint")

    restore_checkpoint()
    stages = ALL_STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    forced = {s.strip() for s in args.force.split(",") if s.strip()}

    fns = {"setup": stage_setup, "metadata": stage_metadata,
           "preprocess": stage_preprocess, "extract": stage_extract,
           "train": stage_train, "eval": stage_eval, "probe": stage_probe}

    for s in stages:
        if s not in fns:
            raise SystemExit(f"unknown stage {s!r}; valid: {ALL_STAGES}")
        if done(s) and s not in forced:
            log(f"--- skipping {s} (already done; --force {s} to re-run) ---")
            continue
        log(f"=== stage: {s} (remaining budget {hms(time_left())}) ===")
        t0 = time.time()
        fns[s](args)
        log(f"=== {s} finished in {(time.time()-t0)/60:.1f} min ===")

    budget_report("ALL DONE")


if __name__ == "__main__":
    main()
