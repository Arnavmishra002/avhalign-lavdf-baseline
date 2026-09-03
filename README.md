# AVH-Align on LAV-DF — reproducible baseline

A single-session, end-to-end reproduction of **AVH-Align** (bit-ml, CVPR 2025) on
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
stopped after epoch 37, 116 minutes on one T4. Test coverage 1000/1000, none
skipped.

> The full LAV-DF test split is 26,100 clips at 6,906 real / 19,194 fake (73.5%
> fake). Because the subsample above is balanced, **AP sits on a 0.5 base rate —
> compare AUC across papers**, and label these numbers *retrained-3k-subset* and
> *official-checkpoint-zero-shot*. They are not a reproduction of the published
> training scale (45,000 clips).

Run of record: [`vansika545/notebook15b67d4dda` v5](https://www.kaggle.com/code/vansika545/notebook15b67d4dda) ·
clean notebook: [`vansika545/avhalign-cells`](https://www.kaggle.com/code/vansika545/avhalign-cells)

## What's here

| path | purpose |
| --- | --- |
| `avhalign_cells.ipynb` | the pipeline as 11 documented cells: setup → metadata → mouth ROIs → features → training → evaluation |
| `_inner.py` + `build_cells.py` | source of truth; the builder splits it into the notebook's cells and re-checks it |
| `_fulltest.py` | multi-session variant that scores the complete 26,100-clip test split |
| `splits/` | **the exact clip lists used** — 3,000 train, 300 val, 1,000 test with labels |
| `score_clips.py` | per-clip scores (upstream `eval.py` prints only AP/AUC) |
| `compare_models.py` | joins several models' scores, checks coverage, reports bootstrap CIs and paired ΔAUC |
| `metrics_from_scores.py` | AP, AUC, EER + accuracy / precision / recall / F1 / specificity and confusion matrices at three operating points |
| `scores/` | per-clip scores of both checkpoints on the 1,000 test clips |
| `push_kernel.py` | pushes the notebook to Kaggle and starts a run |
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
