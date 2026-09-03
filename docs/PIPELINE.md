# AVH-Align on LAV-DF — single-session Kaggle reproduction

Runs the AVH-Align pipeline (bit-ml, CVPR 2025) end to end on a LAV-DF subset
inside one Kaggle session: mouth-ROI preprocessing, frozen AV-HuBERT feature
extraction, alignment-model training, and evaluation.

## Results

Measured on `vansika545/notebook15b67d4dda` version 5 (complete, 2 h 16 m,
GPU T4 x2). Test set: a **balanced 1000-clip subsample** of the LAV-DF test
split (500 real / 500 fake), 1000/1000 clips with features, none skipped.

| checkpoint | AP | AUC |
| --- | --- | --- |
| retrained on a 3000-clip real-only LAV-DF subset | 0.7872 | 0.8263 |
| official AV1M checkpoint, zero-shot | 0.8272 | 0.8659 |

Training: 3000 real train clips, 300 real val clips, tau 15, batch 1024,
lr 1e-5, early stopping patience 10. Best validation loss 1.051858 at epoch 27;
training stopped after epoch 37 of a 40-epoch cap. Features: 6.12 GiB for
4300 clips (1.46 MiB per clip).

### How these numbers may be described

They are **retrained-3k-subset** and **official-checkpoint-zero-shot** numbers.
They are not a reproduction of the published LAV-DF results: the paper trains on
45000 clips and evaluates on the full test split at its natural class prior,
whereas this test set is balanced 50/50, which shifts AP (AUC is prior-
independent but still measured on a subsample). About 1% of clips hit dlib's
"passing whole video" fallback and carry full frames instead of mouth crops.

The official checkpoint scoring above the retrained one is the expected
direction — it saw 45000 AV1M clips against this run's 3000 — and is worth
reporting as a sanity anchor.

## Evaluation protocol

Two test protocols exist in this repo, and they are not interchangeable.

| protocol | selection | what AP means |
| --- | --- | --- |
| balanced — the shipped default, `test_balanced=True` | 500 real + 500 fake | base rate 0.5; not comparable with published AP |
| natural prior, `test_balanced=False` | uniform sample of the LAV-DF test split | unbiased estimate of full-split AP |

AUC is prior-independent and comparable under both. Sample size is bounded by
the 20 GB Kaggle output quota, and the binding term is the features at
1.46 MiB/clip; the mouth ROIs are only 0.28 MiB/clip.

A natural-prior run reuses the trained model instead of retraining: attach the
output of a completed run as a kernel input, and CELL 4 copies `<name>.pt` into
place and marks the train stage done. `resume_data=False` makes it ignore that
input's ROIs and features, which would otherwise pin the old clip selection.

## Notebook

`avhalign_cells.ipynb` — 11 code cells, ~1160 lines, generated from `_inner.py`
by `build_cells.py`. Published as `vansika545/avhalign-cells`.

| cell | stage |
| --- | --- |
| 1-3 | imports, configuration, helpers (budget and disk guards, stage runner) |
| 4 | resume: reuse preprocessed ROIs / features from an attached input |
| 5 | accelerator check, repos, pinned deps, 2021-code compatibility patches, assets |
| 6 | LAV-DF metadata and seeded train / val / test splits |
| 7 | mouth-ROI preprocessing (dlib, 4 workers, ~4.6 h for 4300 clips) |
| 8 | AV-HuBERT feature extraction (~18 min on T4 x2) |
| 9 | training (upstream `train.py`, wrapped in `timeout` against the budget) |
| 10 | evaluation (upstream `eval.py`: AP / AUC) |
| 11 | summary |

## Running it

Accelerator must be **GPU T4 x2**. The pinned torch build supports compute
capability 7.0 and newer; Kaggle's P100 is 6.0, where `.cuda()` succeeds but
every kernel launch raises — the upstream extraction script swallows that in a
bare `except:` and silently produces zero features. CELL 5 now refuses to start
on such a device.

Fast path (~2.5 h) — attach `vansika545/avhalign-lavdf-v8-output` alongside the
LAV-DF dataset. CELL 4 symlinks its `lavdf_pre` / `lavdf_feats`, copies the
split CSVs, and marks metadata/preprocess/extract done only because the data is
actually there, so the run is setup -> train -> eval.

Full path (~7 h) — attach only `elin75/localized-audio-visual-deepfake-dataset-lav-df`.
CELL 4 logs `no lavdf_pre / lavdf_feats found -> full run`.

```bash
export KAGGLE_API_TOKEN=KGAT_...      # kaggle.com/settings/api
python3 push_kernel.py                 # pushes the notebook, sets T4 x2, starts a run
```

`push_kernel.py` talks to the REST API directly because the installed kaggle CLI
(1.7.4.5) predates KGAT tokens. `machineShape: "NvidiaTeslaT4"` is what pins
T4 x2; `enableGpu` alone gets a P100.

