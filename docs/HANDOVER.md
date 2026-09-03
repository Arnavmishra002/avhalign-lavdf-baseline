# AVH-Align on LAV-DF — complete handover

Everything needed to (a) write up this baseline and (b) compare it fairly with
other detectors trained on the same dataset.

---

## 1. Where everything lives

> Every Kaggle notebook and dataset below is public, so the logs, outputs and
> cached features can be inspected directly; the repository alone is also enough
> to reproduce the work from scratch.

### The run that produced the numbers

| item | location |
| --- | --- |
| notebook (run + outputs) | https://www.kaggle.com/code/vansika545/notebook15b67d4dda — **version 5**, status complete, 2 h 16 m, GPU T4 x2 |
| log | https://www.kaggle.com/code/vansika545/notebook15b67d4dda/log?scriptVersionId=346833076 |
| trained checkpoint | that run's output, `checkpoints/AVH-Align_LAVDF.pt` |
| clip lists actually used | that run's output, `lavdf_meta/*.csv` — copied into this repo under `splits/` |
| features + mouth ROIs | dataset `vansika545/avhalign-lavdf-v8-output` (`lavdf_feats/`, `lavdf_pre/`) |
| clean 11-cell notebook | https://www.kaggle.com/code/vansika545/avhalign-cells (same pipeline, audited) |

### Code in this repo

| file | what it is |
| --- | --- |
| `avhalign_cells.ipynb` | the complete pipeline as 11 documented cells, raw video to final metrics in one session — the artifact to ship with the paper |
| `_inner.py`, `build_cells.py` | source of truth; `build_cells.py` splits it into the notebook's cells |
| `_fulltest.py`, `build_fulltest.py`, `avhalign_fulltest.ipynb` | variant that scores the complete 26,100-clip test split across sessions. Assembled from the same imports / helpers / setup cells as the main notebook and statically verified (every cell compiles, no undefined names), but **not yet executed end to end** |
| `splits/*.csv` | **the exact clip lists V5 used** — 3,000 train, 300 val, 1,000 test (500/500) |
| `score_clips.py` | writes per-clip AVH-Align scores (eval.py prints only AP/AUC) |
| `compare_models.py` | joins several models' per-clip scores and emits the comparison table |
| `metrics_from_scores.py` | AP, AUC, EER plus accuracy / precision / recall / F1 / specificity and confusion matrices at three operating points |
| `scores/test_scores.csv` | per-clip scores of both checkpoints on the 1,000 test clips |
| `push_kernel.py` | pushes the notebook to Kaggle and starts a run |
| `docs/EXPLANATION.md` | plain-language walk-through of the whole project |
| `docs/PAPER_NOTES.md` | plain-language summary, methods paragraph, training curve, claim limits |
| `docs/PIPELINE.md` | protocols, cell map, reproduction modes |

### Upstream code

| component | source | pin |
| --- | --- | --- |
| AVH-Align (method, train/eval scripts) | https://github.com/bit-ml/AVH-Align | default branch, CVPR 2025 |
| AV-HuBERT (feature backbone) | https://github.com/facebookresearch/av_hubert | default branch |
| fairseq (AV-HuBERT dependency) | https://github.com/pytorch/fairseq | commit `afc77bd` (submodule pin) |
| AV-HuBERT weights | `https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt` | 5.4 GB |
| dlib landmarks | `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` | 95 MiB |
| mean-face template | mpc001/Lipreading_using_Temporal_Convolutional_Networks, `preprocessing/20words_mean_face.npy` | — |
| LAV-DF | Kaggle `elin75/localized-audio-visual-deepfake-dataset-lav-df` | 136,304 clips |

---

## 2. The technique, in the order the pipeline applies it

**1. Clip selection.** LAV-DF ships `metadata.json` with a `split` field. A clip
counts as real only when `modify_video` is false, `modify_audio` is false, and
`n_fakes == 0`. Clips shorter than 31 frames are dropped, because the model's
temporal window is 2τ+1 = 31 frames. Selection is seeded (42).

**2. Mouth-ROI preprocessing** (`deepfake_preprocess.py`, upstream). dlib finds
68 facial landmarks per frame, the mouth region is aligned to a 20-word mean
face, cropped to 96x96 and written as grayscale video; audio is exported at
16 kHz. When no face is found the upstream code keeps the whole frame and logs
`passing whole video` (~1% of clips). CPU-bound: 3.9 s per clip on 4 workers.

**3. Frozen feature extraction** (`deepfake_feature_extraction.py`, upstream).
AV-HuBERT Large (`self_large_vox_433h`, fine-tuned checkpoint) runs twice per
clip — once with video only, once with audio only — giving two per-frame
sequences of 1024-d features, saved as one `.npz` per clip. Nothing is trained
here. The model's third, multimodal pass is disabled in our runs because the
alignment head never reads it (~33% less GPU time and disk).

**4. The alignment model** (`model.py`, upstream). A small fusion network:
visual and audio each go through a 1024->512 linear projection, are
concatenated, and pass an MLP 1024->512->256->128->1 with LayerNorm and ReLU.
About 1M parameters.

**5. Training on real clips only** (`train.py`, upstream). For each frame, the
visual feature of that frame is paired with a window of 2τ+1 = 31 audio frames
centred on it. The model scores every offset in the window; a log-softmax across
the window is taken and the loss is the negative log-probability of the centre
offset. In words: on genuine video the audio that belongs with a frame is the
audio at the same instant, so the model learns to put its mass on the centre.
Fakes break that correspondence. This is why only real clips are needed — no
fake examples enter training.

**6. Scoring** (`eval.py`, upstream). Both feature streams are L2-normalised,
the model is run over the clip, and the clip score is
`logsumexp(-output, dim=0)`: high when alignment is poor, i.e. high = fake.
Metrics are `average_precision_score` and `roc_auc_score` from scikit-learn.

**Compatibility work (ours, not the method).** fairseq `afc77bd` is from 2021 and
does not import on Python 3.12: mutable dataclass defaults are rewritten to
`field(default_factory=...)`, removed numpy aliases and moved collections ABCs
are patched, av_hubert's relative imports are made absolute, `hydra_init()` is
disabled (the hydra CLI is unused), and `torch.load(weights_only=False)` is
forced so the 2021 checkpoint loads on torch 2.x. All of it is in CELL 5 and is
idempotent.

---

## 3. Exact configuration

```
train clips        3000 real      (LAV-DF train split, seed 42)
val clips           300 real      (LAV-DF dev split)
test clips         1000           (LAV-DF test split, 500 real / 500 fake)
tau                  15           window = 2*tau+1 = 31 frames
batch_size         1024
optimiser          Adam, lr 1e-5
scheduler          ReduceLROnPlateau, factor 0.1, patience 5
early stopping     patience 10, epoch cap 40
epochs run           37           best val loss 1.051858 at epoch 27
num_workers           4           (upstream default 32)
features           AV-HuBERT Large self_large_vox_433h, frozen
environment        Python 3.12, numpy 1.26.4, omegaconf 2.0.6, hydra-core 1.0.7,
                   torch 2.x (Kaggle image), fairseq afc77bd via .pth
hardware           1 session, NVIDIA T4 x2, 2 h 16 m wall clock
```

## 4. Results

Balanced 1,000-clip test subsample (500 real / 500 fake), both models scored by
upstream `eval.py` in the same session:

| model | AP | AUC |
| --- | ---: | ---: |
| AVH-Align retrained on 3k real LAV-DF clips | 0.7872 | 0.8263 |
| AVH-Align official AV1M checkpoint, zero-shot | 0.8272 | 0.8659 |


Threshold-dependent metrics, both checkpoints, same 1,000 clips (full tables and
confusion matrices in `docs/PAPER_NOTES.md`):

| model | EER | acc @EER | recall @EER | F1 @EER | recall @maxF1 | F1 @maxF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retrained on 3k real clips | 0.2520 | 0.7480 | 0.7460 | 0.7475 | 0.8800 | 0.7746 |
| official AV1M, zero-shot | 0.2140 | 0.7860 | 0.7860 | 0.7860 | 0.9280 | 0.8191 |

Paired bootstrap over the same clips: ΔAUC -0.0396 [-0.0604, -0.0188], p = 0.0005.

Per-clip scores: `scores/test_scores.csv`, from Kaggle notebook
`vansika545/avhalign-scores` (CPU only). It reproduces upstream `eval.py`'s AP
and AUC exactly, which is why its per-clip output is trusted for the rest.

Full LAV-DF test split for reference: 26,100 clips, 6,906 real / 19,194 fake
(73.5% fake). Because our subsample is balanced, **AP is on a 0.5 base rate and
AUC is the metric to compare across papers.**

---

## 5. Comparing with other models

Three rules make the comparison defensible, and this repo carries the pieces for
each.

**Rule 1 — identical clips.** Use `splits/test_metadata.csv`: the exact 1,000
clips (with labels) that produced the numbers above. Score every other model
on that list, no re-sampling. If a model cannot process a clip, drop that clip
from *every* model, not just its own.

**Rule 2 — per-clip scores, not just summary metrics.** Summary AP/AUC cannot
support a significance test. Emit one CSV per model with columns `path,score`
(higher = more likely fake). For AVH-Align, `score_clips.py` produces exactly
that from the features in the v8 dataset:

```bash
# inside the AVH-Align checkout, features mounted
python3 score_clips.py test_metadata.csv lavdf_feats/val avhalign_scores.csv \
    retrained=checkpoints/AVH-Align_LAVDF.pt official=checkpoints/AVH-Align_AV1M.pt
```

AVH-Align's own scores are already in `scores/test_scores.csv`, so this only
needs re-running for a different clip list.

**Rule 3 — paired statistics.** The models see the same clips, so differences
must be tested paired; an unpaired test discards that and overstates the
uncertainty. `compare_models.py` does the joining, the coverage check, bootstrap
CIs, and paired bootstrap differences:

```bash
python3 compare_models.py \
    --labels splits/test_metadata.csv \
    --scores 'avh-align=scores/test_scores.csv#score_retrained' \
             model-b=modelb_scores.csv model-c=modelc_scores.csv
```

It prints a markdown results table and a paired-difference table (ΔAUC, 95% CI,
two-sided p) ready to paste.

**What to say about training scale.** AVH-Align here saw 3,000 real clips; if
your other two models trained on the full split, say so in the table caption —
otherwise the comparison reads as a method comparison when it is partly a data
comparison. The fair-scale alternative is to also report the authors' released
checkpoint (row 2 above), which is the strongest published version of this
method.

---

## 6. Reproducing or extending

| goal | inputs to attach | settings | time |
| --- | --- | --- | --- |
| repeat V5 exactly | LAV-DF + `avhalign-lavdf-v8-output` | `resume_data=True` | ~2.5 h |
| full pipeline from raw video | LAV-DF | `data_splits="train,test"` | ~7 h |
| score the full 26,100-clip test split | see `_fulltest.py` | `MODE="preprocess"` x3 on CPU, then `MODE="score"` on GPU | ~28 h CPU + 2.5 h GPU |

Accelerator must be **GPU T4 x2** for anything that touches AV-HuBERT. On
Kaggle's P100 every CUDA kernel raises and the upstream extractor swallows the
error, producing zero features after hours; the notebook now refuses to start
there.
