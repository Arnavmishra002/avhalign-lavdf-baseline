# Paper material — AVH-Align baseline on LAV-DF

Everything needed to cite this baseline in a comparison paper: what was run,
what the numbers mean, and what may not be claimed.

## Plain summary

What we did, in order:

1. Took LAV-DF (136,304 clips). Picked 3,000 real clips to train on, 300 real
   clips to validate on, and clips from the test split to score on. The picking
   is seeded, so it repeats exactly.
2. Cut each clip down to the speaker's mouth (96x96, grayscale) and pulled the
   audio out at 16 kHz. This is the input AVH-Align expects.
3. Ran every clip through a frozen AV-HuBERT Large model to get one feature file
   per clip. Nothing is learned here; it is a fixed feature extractor.
4. Trained the small AVH-Align alignment head on the real clips only. It learns
   what genuine lip-audio alignment looks like; a fake scores as a deviation.
5. Scored two models on the same test clips with the authors' own eval script:
   our retrained head, and the checkpoint the authors released.

What we got (balanced 1,000-clip test set, 500 real / 500 fake):

- our retrained model: AUC 0.826, AP 0.787
- the authors' released model, never trained by us: AUC 0.866, AP 0.827

What it means: the released model wins by about 4 AUC points, which is expected
because it saw 45,000 training clips and ours saw 3,000. Our run is a working,
verified baseline of the method, not a reproduction of the paper's training
scale.

The one number to be careful with is AP. Our test set was forced to 50/50, so AP
starts from a 0.5 base rate. The real LAV-DF test split is 26,100 clips at
6,906 real / 19,194 fake, i.e. 73.5% fake, so published AP is on a different
base rate. AUC does not have this problem and is the safe number to compare.

## Methods paragraph (drop-in)

We evaluate AVH-Align [bit-ml, CVPR 2025] on LAV-DF. Mouth regions of interest
are extracted with the authors' preprocessing (dlib 68-point landmarks aligned
to the 20-word mean face, 96x96 grayscale, 16 kHz audio), and per-frame visual
and audio representations are taken from a frozen AV-HuBERT Large model
(`self_large_vox_433h`); the model's multimodal pass is unused by the alignment
head and is disabled. The alignment model is the authors' fusion network trained
on real clips only, with the released hyper-parameters (tau = 15, batch 1024,
Adam lr 1e-5, ReduceLROnPlateau patience 5, early stopping patience 10, 40-epoch
cap). We train on a 3,000-clip real-only sample of the LAV-DF train split with
300 real validation clips from the dev split, selected with a fixed seed, and
evaluate on a sample of the LAV-DF test split. All stages run in a single
12-hour Kaggle session on one NVIDIA T4.

## Results

### Balanced 1,000-clip test subsample (500 real / 500 fake)

| model | AP | AUC |
| --- | --- | --- |
| AVH-Align, retrained on 3k real LAV-DF clips | 0.7872 | 0.8263 |
| AVH-Align, official AV1M checkpoint (zero-shot) | 0.8272 | 0.8659 |

Source: Kaggle `vansika545/notebook15b67d4dda` v5, 2 h 16 m, T4 x2. Training
converged by early stopping: best validation loss 1.051858 at epoch 27, stopped
after epoch 37. Test coverage 1000/1000 clips, none skipped.

### Natural-prior 4,000-clip test sample

Run in progress (`vansika545/avhalign-cells` v4). Fill in on completion:

| model | AP | AUC | n | fake prior |
| --- | --- | --- | --- | --- |
| AVH-Align, retrained on 3k real LAV-DF clips | _pending_ | _pending_ | 4000 | _pending_ |
| AVH-Align, official AV1M checkpoint (zero-shot) | _pending_ | _pending_ | 4000 | _pending_ |

The retrained model here is the same checkpoint as above, reused rather than
retrained, so the two rows differ only in test protocol.

## Detection metrics

AP and AUC need no threshold and are the headline numbers. Accuracy, precision,
recall, F1 and specificity do need an operating point, so three defensible ones
are reported rather than one arbitrary cut. Intervals are 2,000-resample
bootstraps over clips (seed 0). Test set: the 1,000 clips in
`splits/test_metadata.csv`, 500 real / 500 fake.

Per-clip scores are in `scores/test_scores.csv`, produced by Kaggle notebook
`vansika545/avhalign-scores` (CPU only). That scorer reproduces the upstream
`eval.py` numbers exactly — AP 0.7872 / AUC 0.8263 and AP 0.8272 / AUC 0.8659 —
which is what licenses using its per-clip output for the metrics below.

### AVH-Align retrained on 3,000 real LAV-DF clips

AP **0.7872** [0.7456, 0.8296] · AUC **0.8263** [0.7999, 0.8508] · EER **0.2520**

| operating point | threshold | accuracy | precision | recall | F1 | specificity | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EER | 5.6800 | 0.7480 | 0.7490 | 0.7460 | 0.7475 | 0.7500 | 375 | 125 | 127 | 373 |
| Youden J | 5.5647 | 0.7580 | 0.7416 | 0.7920 | 0.7660 | 0.7240 | 362 | 138 | 104 | 396 |
| max F1 | 5.1311 | 0.7440 | 0.6918 | 0.8800 | 0.7746 | 0.6080 | 304 | 196 | 60 | 440 |

### AVH-Align official AV1M checkpoint, zero-shot

AP **0.8272** [0.7882, 0.8673] · AUC **0.8659** [0.8434, 0.8878] · EER **0.2140**

| operating point | threshold | accuracy | precision | recall | F1 | specificity | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EER | -3.1790 | 0.7860 | 0.7860 | 0.7860 | 0.7860 | 0.7860 | 393 | 107 | 107 | 393 |
| Youden J | -3.7439 | 0.7990 | 0.7488 | 0.9000 | 0.8174 | 0.6980 | 349 | 151 | 50 | 450 |
| max F1 | -3.9645 | 0.7950 | 0.7330 | 0.9280 | 0.8191 | 0.6620 | 331 | 169 | 36 | 464 |

### The two checkpoints, tested against each other

Paired bootstrap on identical clips: **ΔAUC = -0.0396** [-0.0604, -0.0188],
**p = 0.0005** (retrained minus official). The released checkpoint is better by a
margin that is not attributable to sampling noise, which is the expected
direction at 45,000 training clips against 3,000.

Thresholds are on the raw `logsumexp(-output)` scale and are not comparable
between checkpoints; only the metrics are.

## Training curve

Loss is the negative log-probability of the true (centre) audio offset within the
31-frame window, averaged over frames. Lower is better; there is no accuracy to
report during training because no labels are used — the model only ever sees real
clips.

| epoch | train loss | val loss |
| ---: | ---: | ---: |
| 1 | 2.502812 | 1.713182 |
| 2 | 1.585880 | 1.487583 |
| 3 | 1.441480 | 1.385080 |
| 4 | 1.353225 | 1.318291 |
| 5 | 1.290571 | 1.272802 |
| 6 | 1.242882 | 1.238791 |
| 7 | 1.203762 | 1.211809 |
| 8 | 1.170115 | 1.189316 |
| 9 | 1.140276 | 1.170103 |
| 10 | 1.113355 | 1.153155 |
| 11 | 1.088636 | 1.138458 |
| 12 | 1.065733 | 1.125316 |
| 13 | 1.044328 | 1.113968 |
| 14 | 1.024200 | 1.103676 |
| 15 | 1.005161 | 1.094504 |
| 16 | 0.987115 | 1.086833 |
| 17 | 0.969884 | 1.079751 |
| 18 | 0.953440 | 1.073404 |
| 19 | 0.937653 | 1.068268 |
| 20 | 0.922455 | 1.063809 |
| 21 | 0.907826 | 1.060088 |
| 22 | 0.893709 | 1.057060 |
| 23 | 0.880018 | 1.054715 |
| 24 | 0.866757 | 1.053166 |
| 25 | 0.853895 | 1.052050 |
| 26 | 0.841343 | 1.051876 |
| 27 | 0.829172 | 1.051858 |  ← best, evaluated checkpoint
| 28 | 0.817321 | 1.051954 |
| 29 | 0.805744 | 1.052968 |
| 30 | 0.794480 | 1.054575 |
| 31 | 0.783507 | 1.056855 |
| 32 | 0.772745 | 1.059354 |  lr 1e-5 → 1e-6
| 33 | 0.757291 | 1.060631 |
| 34 | 0.753887 | 1.060556 |
| 35 | 0.751965 | 1.060721 |
| 36 | 0.750303 | 1.060971 |
| 37 | 0.748751 | 1.061300 |

Best validation loss **1.051858 at epoch 27**. Training continued for the full
early-stopping patience of 10 epochs and stopped after epoch 37; the evaluated
checkpoint is epoch 27's. Train loss keeps falling after that while validation
loss rises slightly — mild overfitting, caught by early stopping, not by the
session budget.

## What may and may not be claimed

- Report these as **retrained-3k-subset** and **official-checkpoint-zero-shot**.
  They are not a reproduction of the published LAV-DF results: the paper trains
  on 45,000 clips.
- AP from the balanced subsample sits on a 0.5 base rate and must not be
  compared with AP published on the full test split. Use the natural-prior
  numbers for that comparison, or compare AUC, which is prior-independent.
- Roughly 1% of clips fall back to full frames when dlib finds no face
  ("passing whole video" in the logs); this is upstream behaviour, not a
  modification.
- The official checkpoint outperforming the retrained one is expected given
  45,000 AV1M clips against 3,000, and is worth reporting as a sanity anchor
  rather than omitting.

## Reproduction

Notebook: `avhalign_cells.ipynb` (11 cells) — published as
`vansika545/avhalign-cells`. Accelerator must be GPU T4 x2; see README for why a
P100 silently produces zero features.

| goal | inputs to attach | CFG | time |
| --- | --- | --- | --- |
| full pipeline from raw video | LAV-DF | defaults, `data_splits="train,test"` | ~7 h |
| train only, reusing ROIs/features | LAV-DF + `vansika545/avhalign-lavdf-v8-output` | `resume_data=True` | ~2.5 h |
| evaluate a new test set with a trained model | LAV-DF + a finished run's output | `resume_data=False`, `data_splits="test"` | ~5 h |

## Cost of scaling up (measured, not estimated)

Rates from our own runs: preprocessing 3.9 s per clip on 4 CPU workers, feature
extraction 0.25 s per clip on a T4, training ~3.1 min per epoch per 3,000 train
clips, features 1.46 MiB per clip, mouth ROIs 0.28 MiB per clip (read off the
disk deltas of a from-scratch run). Kaggle gives 12 h per session and 20 GB of
output per session.

| target | clips | preprocess | extract | train | sessions |
| --- | --- | --- | --- | --- | --- |
| full LAV-DF test split | 26,100 | ~28 h | ~1.8 h | — | ~5 |
| all real train clips | ~25,000 | ~27 h | ~1.7 h | ~14 h | ~5-7 |

Neither fits one session. Disk is only a problem for the features — 26,100 clips
of features are 37 GiB against a 20 GB quota, while their mouth ROIs are just
7 GiB and fit comfortably. So preprocessing is limited by session *time* and can
simply be split across sessions, whereas scoring must work in chunks that delete
each chunk's features once its scores are written. Full-scale training would
additionally need resume-from-checkpoint across sessions.

## Upstream

- AVH-Align: https://github.com/bit-ml/AVH-Align (CVPR 2025)
- AV-HuBERT: https://github.com/facebookresearch/av_hubert, checkpoint
  `self_large_vox_433h.pt`, fairseq pinned at `afc77bd`
- LAV-DF (Kaggle mirror): `elin75/localized-audio-visual-deepfake-dataset-lav-df`
