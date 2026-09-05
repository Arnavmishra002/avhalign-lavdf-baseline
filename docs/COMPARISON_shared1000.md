# Comparison on the shared 1,000-clip protocol (2026-09-05)

All three notebooks use the same split SHAPE: LAV-DF, 1,000 clips, seed 42, **train 300 real + 300 fake,
validation 100 + 100, test 100 + 100**, no overlap. Numbers for AVoiD-DF and AuViRe are read from the reviewers'
notebooks (`avoid-df-2-lav-1000.ipynb`, `auvire-lavdf--1000.ipynb`, Section 16 "Final evaluation on 200-video
test split"); AVH-Align numbers are from `scores/shared1000/test_scores.csv` (this repository).

| method | trained on | epochs | test acc | precision | recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AVoiD-DF (reviewers' notebook, from scratch) | 300 real + 300 fake | 5 | 61.5 % | 62.9 % | 56.0 % | 59.3 % | 0.682 | — |
| AuViRe-inspired (reviewers' notebook, from scratch) | 300 real + 300 fake | 5 | 69.5 % | 75.3 % | 58.0 % | 65.5 % | 0.750 | — |
| **AVH-Align, retrained head (ours)** | 300 real (self-supervised) | ≤ 40, best 30 | 82.5 % (Youden) / 81.5 % (EER) | 81.0 % / 81.8 % | 85.0 % / 81.0 % | 82.9 % / 81.4 % | **0.875** [0.823, 0.923] | 0.837 [0.757, 0.913] |
| AVH-Align, official AV1M checkpoint, zero-shot (ours) | AV1M (45k real) | — | 83.5 % (Youden) | 80.2 % | 89.0 % | 84.4 % | **0.894** [0.847, 0.935] | 0.878 [0.809, 0.936] |
| AV-HuBERT features + linear probe (ours, supervised reference) | 300 real + 300 fake | LR | — | — | — | — | **0.987** | 0.988 |

Notes that must accompany the table:
1. **Clips are not identical across the three notebooks.** The reviewers select clips with `random.sample` over
   the glob order of 136,304 mp4 paths (order is filesystem-dependent) and label by `n_fakes == 0`; this
   repository draws a seeded, class-balanced sample from the sorted metadata and labels by
   `modify_video / modify_audio / fake_periods`. Same shape, different 1,000 clips. For an identical-clip
   comparison, drop the reviewers' `AVoiD_DF_1000_video_split.json` into `CFG.split_file` and re-run
   (no code change); `compare_models.py` then gives paired bootstrap differences on the same 200 clips.
2. The reviewers' models are trained **from scratch** on 600 clips for 5 epochs (AVoiD-DF averages 16
   face-cropped frames into one image; the audio is a mel spectrogram treated as an image). Their AUCs
   (0.68 / 0.75) are under-trained re-implementations, not the published numbers of those papers.
   AVH-Align uses a frozen pretrained AV-HuBERT backbone by design — that asymmetry must be stated.
3. AVH-Align's accuracy / precision / recall / F1 need a threshold (it outputs an alignment score);
   two operating points are given (Youden J, EER) so the row can be read next to argmax-based numbers.
   AUC and AP are threshold-free and are the numbers to compare.
4. The probe row is a supervised method that uses the fake training clips; report it separately from AVH-Align.
5. Retrained vs official AVH-Align on the same 200 clips: paired ΔAUC −0.019 [−0.064, +0.026], p = 0.43.
