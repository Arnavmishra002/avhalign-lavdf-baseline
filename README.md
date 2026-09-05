# AVH-Align on LAV-DF — reproducible baseline

A single-session, end-to-end re-implementation of the **AVH-Align** pipeline (bit-ml, CVPR 2025) on
the **LAV-DF** dataset, built as an 11-cell Kaggle notebook, together with the
exact clip lists, per-clip scoring, and a paired-statistics harness for comparing
it against other detectors on identical data.

## Results

Balanced 1,000-clip test subsample of the LAV-DF test split (500 real / 500 fake),
both models scored by the authors' own `eval.py` in the same session:

| model | AP | AUC |
| --- | ---: | ---: |
| AVH-Align retrained on 3,000 real LAV-DF clips | 0.7872 | 0.8263 |
| AVH-Align official AV1M checkpoint, zero-shot | 0.8272 | 0.8659 |


Threshold-dependent metrics on the same clips (EER operating point; full tables,
confusion matrices and the max-F1 / Youden-J points are in
[`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md)):

| model | EER | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| retrained on 3k real clips | 0.2520 | 0.7480 | 0.7490 | 0.7460 | 0.7475 |
| official AV1M, zero-shot | 0.2140 | 0.7860 | 0.7860 | 0.7860 | 0.7860 |

Paired bootstrap over identical clips: **ΔAUC -0.0396** [-0.0604, -0.0188],
p = 0.0005. Per-clip scores ship in `scores/test_scores.csv`.

Training converged by early stopping: best validation loss 1.051858 at epoch 27,
stopped after epoch 37, 116 minutes in a T4 x2 session. Test coverage
1000/1000, none skipped.

> The full LAV-DF test split is 26,100 clips at 6,906 real / 19,194 fake (73.5%
> fake). Because the subsample above is balanced, **AP sits on a 0.5 base rate —
> compare AUC across papers**, and label these numbers *retrained-3k-subset* and
> *official-checkpoint-zero-shot*. They are not a reproduction of the published
> training scale (45,000 clips).

Run of record: [`vansika545/notebook15b67d4dda` v5](https://www.kaggle.com/code/vansika545/notebook15b67d4dda) ·
clean notebook: [`vansika545/avhalign-cells`](https://www.kaggle.com/code/vansika545/avhalign-cells) ·
per-clip scoring: [`vansika545/avhalign-scores`](https://www.kaggle.com/code/vansika545/avhalign-scores) ·
cached features: [`vansika545/avhalign-lavdf-v8-output`](https://www.kaggle.com/datasets/vansika545/avhalign-lavdf-v8-output).
All are public, so the logs, outputs and features behind these numbers can be
inspected directly.

## One notebook, start to finish

`avhalign_cells.ipynb` runs the entire pipeline in a single Kaggle session with
nothing but the LAV-DF dataset attached: it clones the upstream code, patches the
2021 dependency stack, builds the seeded splits, cuts mouth ROIs, extracts
AV-HuBERT features, trains the alignment head, and evaluates — ending with AP,
AUC, EER and accuracy / precision / recall / F1 / specificity at three operating
points, plus per-clip scores written to `scores/test_scores.csv`. About 7 hours;
no second notebook and no manual step in between.

The other notebook, `avhalign_fulltest.ipynb`, exists only for the case that does
not fit one session: scoring all 26,100 test clips, whose 28 h of preprocessing
must be spread across sessions.

## All links

Everything behind these numbers is public and inspectable.

**This work**

| what | link |
| --- | --- |
| this repository | https://github.com/Arnavmishra002/avhalign-lavdf-baseline |
| run of record — the reported numbers, full log and outputs (v5) | https://www.kaggle.com/code/vansika545/notebook15b67d4dda |
| the clean 11-cell pipeline notebook | https://www.kaggle.com/code/vansika545/avhalign-cells |
| per-clip scoring run | https://www.kaggle.com/code/vansika545/avhalign-scores |
| cached mouth ROIs + AV-HuBERT features (skips 4.6 h of preprocessing) | https://www.kaggle.com/datasets/vansika545/avhalign-lavdf-v8-output |

**Upstream**

| what | link | pin |
| --- | --- | --- |
| AVH-Align — the method, `train.py`, `eval.py`, preprocessing and extraction | https://github.com/bit-ml/AVH-Align | default branch (CVPR 2025) |
| AV-HuBERT — the frozen feature backbone | https://github.com/facebookresearch/av_hubert | default branch |
| fairseq — AV-HuBERT's dependency | https://github.com/pytorch/fairseq | commit `afc77bd` |
| AV-HuBERT Large weights (5.4 GB) | https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt | `self_large_vox_433h` |
| dlib 68-point landmark model (95 MiB) | http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | — |
| 20-word mean-face template | https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks | `preprocessing/20words_mean_face.npy` |
| LAV-DF dataset | https://www.kaggle.com/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df | 136,304 clips |

## What's here

| path | purpose |
| --- | --- |
| `avhalign_cells.ipynb` | the pipeline as 11 documented cells: setup → metadata → mouth ROIs → features → training → evaluation |
| `_inner.py` + `build_cells.py` | source of truth; the builder splits it into the notebook's cells and re-checks it |
| `_fulltest.py` + `build_fulltest.py` → `avhalign_fulltest.ipynb` | variant that scores the complete 26,100-clip test split across sessions; shares imports / helpers / setup with the main notebook, statically verified but **not yet executed end to end** |
| `splits/` | **the exact clip lists used** — 3,000 train, 300 val, 1,000 test with labels |
| `score_clips.py` | per-clip scores (upstream `eval.py` prints only AP/AUC) |
| `compare_models.py` | joins several models' scores, checks coverage, reports bootstrap CIs and paired ΔAUC |
| `metrics_from_scores.py` | AP, AUC, EER + accuracy / precision / recall / F1 / specificity and confusion matrices at three operating points |
| `scores/` | per-clip scores of both checkpoints on the 1,000 test clips, written by CELL 10 of the run itself |
| `push_kernel.py` | pushes the notebook to Kaggle and starts a run |
| `docs/EXPLANATION.md` | plain-language walk-through of the whole project: method, engineering, results, claims |
| `docs/HANDOVER.md` | every artifact, link and pin, plus the method in pipeline order |
| `docs/PAPER_NOTES.md` | plain-language summary, drop-in methods paragraph, claim limits |
| `docs/PIPELINE.md` | protocols, cell map, reproduction modes, quota notes |

## Comparing against other detectors

Three rules keep a multi-model comparison defensible; this repo carries the
pieces for each.

1. **Identical clips.** Score every model on `splits/test_metadata.csv`. If a
   model cannot process a clip, drop it from *all* models.
2. **Per-clip scores**, not summary metrics — a summary AP cannot support a
   significance test.
   ```bash
   python3 score_clips.py test_metadata.csv lavdf_feats/val avh.csv \
       avhalign=checkpoints/AVH-Align_LAVDF.pt official=checkpoints/AVH-Align_AV1M.pt
   ```
3. **Paired statistics** — the models see the same clips.
   ```bash
   python3 compare_models.py --labels splits/test_metadata.csv \
       --scores avh-align=scores/test_scores.csv model-b=b.csv model-c=c.csv
   ```
   Prints a markdown results table with bootstrap CIs and a paired-difference
   table (ΔAUC, 95% CI, two-sided p). For a single model's full operating-point
   metrics — accuracy, precision, recall, F1, specificity, confusion matrix at
   the EER, max-F1 and Youden-J thresholds — use `metrics_from_scores.py`.

## Running the pipeline

Kaggle, accelerator **GPU T4 ×2**, internet on:

```bash
export KAGGLE_API_TOKEN=KGAT_...   # kaggle.com/settings/api
python3 push_kernel.py             # pushes the notebook, pins T4 x2, starts a run
```

| goal | inputs to attach | settings | time |
| --- | --- | --- | --- |
| repeat the run above | LAV-DF + `vansika545/avhalign-lavdf-v8-output` | `resume_data=True` | ~2.5 h |
| full pipeline from raw video | LAV-DF | `data_splits="train,test"` | ~7 h |
| full 26,100-clip test split | see `_fulltest.py` | `MODE="preprocess"` ×3 (CPU), then `MODE="score"` (GPU) | ~28 h CPU + ~2 h GPU |

The accelerator matters: the pinned torch build supports compute capability 7.0
and newer. On Kaggle's P100 (6.0) every CUDA kernel raises while the upstream
extractor swallows the error in a bare `except:`, so a run produces **zero
features after hours**. The notebook checks the device and aborts in ~2 minutes
instead.

## Method in one paragraph

Mouth regions are cropped with dlib 68-point landmarks aligned to a mean face
(96×96 grey) and audio exported at 16 kHz. A frozen AV-HuBERT Large model
(`self_large_vox_433h`) produces per-frame visual and audio features. A ~1M
parameter fusion head is trained **on real clips only**: each frame's visual
feature is matched against a 31-frame audio window (τ = 15) and the loss is the
negative log-probability of the centre offset — genuine speech has its audio at
the same instant, and manipulation breaks that. A clip is scored
`logsumexp(-output)`, high meaning poor alignment, i.e. fake.

Full detail, including every compatibility patch applied to the 2021 fairseq
dependency, is in [`docs/HANDOVER.md`](docs/HANDOVER.md).

## Attribution

This repository contains glue code, notebooks and documentation. It does **not**
redistribute the dataset or any model weights.

- AVH-Align — https://github.com/bit-ml/AVH-Align (CVPR 2025)
- AV-HuBERT — https://github.com/facebookresearch/av_hubert, checkpoint `self_large_vox_433h.pt`
- fairseq — https://github.com/pytorch/fairseq, pinned at `afc77bd`
- LAV-DF — Kaggle mirror `elin75/localized-audio-visual-deepfake-dataset-lav-df`

Please cite the AVH-Align and LAV-DF papers for the method and the data.


## Shared 1,000-clip protocol (600 / 200 / 200)

`avhalign_shared1000.ipynb` runs the reviewers' protocol: a class-balanced seeded draw
(train 300 real + 300 fake, val 100 + 100, test 100 + 100), AVH-Align trained on the real
training clips, plus a supervised "AV-HuBERT features + linear probe" row that uses both
classes (CELL 12). See `docs/PAPER_NOTES.md`, section "Shared 1,000-clip protocol".
Pass the reviewers' exact clip list via `CFG.split_file` for identical test clips.
