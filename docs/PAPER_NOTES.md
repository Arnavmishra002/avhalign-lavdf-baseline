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
| full pipeline from raw video | LAV-DF | defaults, `data_splits="train,test"` | ~9 h |
| train only, reusing ROIs/features | LAV-DF + `vansika545/avhalign-lavdf-v8-output` | `resume_data=True` | ~2.5 h |
| evaluate a new test set with a trained model | LAV-DF + a finished run's output | `resume_data=False`, `data_splits="test"` | ~5 h |

## Cost of scaling up (measured, not estimated)

Rates from our own runs: preprocessing 3.9 s per clip on 4 CPU workers, feature
extraction 0.25 s per clip on a T4, training ~3.1 min per epoch per 3,000 train
clips, features 1.46 MiB per clip, mouth ROIs 1.45 MiB per clip. Kaggle gives
12 h per session and 20 GB of output per session.

| target | clips | preprocess | extract | train | sessions |
| --- | --- | --- | --- | --- | --- |
| full LAV-DF test split | 26,100 | ~28 h | ~1.8 h | — | ~5 |
| all real train clips | ~25,000 | ~27 h | ~1.7 h | ~14 h | ~5-7 |

Neither fits one session, and neither fits the 20 GB output quota in one piece
(26,100 clips of features alone are 37 GiB). Both need the same two additions:
process the split in shards and carry only what the next stage needs, and resume
training across sessions from a saved optimizer state.

## Upstream

- AVH-Align: https://github.com/bit-ml/AVH-Align (CVPR 2025)
- AV-HuBERT: https://github.com/facebookresearch/av_hubert, checkpoint
  `self_large_vox_433h.pt`, fairseq pinned at `afc77bd`
- LAV-DF (Kaggle mirror): `elin75/localized-audio-visual-deepfake-dataset-lav-df`
