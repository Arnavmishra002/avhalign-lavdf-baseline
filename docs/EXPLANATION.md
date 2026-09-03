# What this project is, in plain language

A walk-through of the whole piece of work: what was being reproduced, how each
step works, what had to be engineered around, what came out, and what any of it
may be claimed to mean. No prior knowledge of the codebase assumed.

---

## 1. The goal

**AVH-Align** (bit-ml, CVPR 2025) detects deepfake video by checking whether the
mouth and the voice still belong together. We wanted a working, verified copy of
that method on the **LAV-DF** dataset so it can serve as a baseline in a
comparison paper alongside other detectors trained on the same data.

Two numbers were wanted, not one:

- how the method performs when **we train it ourselves** on LAV-DF, and
- how the **authors' released model** performs on the same clips without any
  training from us.

The second is the fairer comparison point for a paper, because it is the method
at the strength its authors published.

## 2. Why the method works the way it does

Most detectors learn "what a fake looks like" from labelled fakes. AVH-Align
does the opposite: it learns **what genuine speech looks like** and flags
deviation. Concretely, in a real recording the audio that belongs with a video
frame is the audio at that same instant. The model is shown a frame's visual
features and a window of audio features around it, and is trained to point at the
centre of that window. On a manipulated clip that correspondence is disturbed and
the model's confidence in the centre drops.

Two consequences follow, and both shaped this project:

- **Training uses real clips only.** No fake examples are needed, and the fake
  half of the data is used purely for evaluation.
- **The score is a mismatch measure.** A clip's score is
  `logsumexp(-model output)`: high means poor alignment, i.e. more likely fake.

## 3. The pipeline, step by step

**Step 1 — choose clips.** LAV-DF ships a `metadata.json` describing 136,304
clips with a `split` field (train / dev / test). A clip counts as real only when
its video was not modified, its audio was not modified, and it has zero fake
segments. Clips shorter than 31 frames are dropped because the model's temporal
window is 31 frames wide. Choices are seeded, so the same clips come out every
run.

**Step 2 — cut the mouth out.** For every clip, dlib finds 68 facial landmarks,
the mouth region is aligned to a standard "mean face", cropped to a 96×96
greyscale video, and the audio is written out at 16 kHz. This is the input format
the feature model expects. It is the slowest part of the whole pipeline —
about 3.9 seconds per clip on four CPU workers — and it needs no GPU. Roughly 1%
of clips have no detectable face; the upstream code then keeps the whole frame
and logs `passing whole video`.

**Step 3 — turn clips into features.** A frozen **AV-HuBERT Large** model runs
over each clip twice, once seeing only the video and once only the audio,
producing two sequences of per-frame 1024-dimensional features saved as one
`.npz` file per clip. Nothing is learned here; this model is never updated. It is
the only step that really wants a GPU: about 0.25 s per clip on a T4.

**Step 4 — train the alignment head.** A small network (~1M parameters)
projects the visual and audio features to 512 dimensions each, concatenates them,
and passes them through an MLP down to a single number. Training pairs each
frame's visual feature with the 31-frame audio window around it and minimises the
negative log-probability of the centre offset. Settings are the authors' own:
τ = 15, batch 1024, Adam at 1e-5, learning-rate decay on plateau, early stopping
after 10 epochs without improvement.

**Step 5 — score and measure.** Each test clip gets one score. From the scores
and the true labels come AP and AUC (threshold-free), and — once a threshold is
picked — accuracy, precision, recall, F1 and specificity. All of this happens in
the last cell of the same notebook, which also writes the per-clip scores to
disk, so one run start to finish produces every number reported here.

## 4. What had to be engineered around

None of this changes the method; it is what made a 2021 research stack run on a
2026 machine inside a 12-hour session.

**The dependency stack is four years old.** AV-HuBERT pins fairseq at commit
`afc77bd` from 2021, which will not even import on Python 3.12: mutable dataclass
defaults are now illegal, several numpy aliases were removed, and the collections
ABCs moved. The setup cell patches each of these automatically and re-tries the
import until it succeeds, so a broken dependency fails in minutes rather than
after hours of preprocessing.

**Old checkpoints versus new PyTorch.** Torch 2.6 stopped unpickling arbitrary
objects by default, and the AV-HuBERT checkpoint contains exactly that. The fix
is applied three independent ways so it cannot be missed by a subprocess.

**Kaggle's limits shape the design.** A session gives 12 hours and 20 GB of
output. Features are 1.46 MiB per clip, so they are the binding constraint;
mouth ROIs are only 0.28 MiB per clip. The pipeline therefore checks time and
disk before starting each stage, deletes a split's mouth ROIs once its features
exist — but never before, so a failed extraction cannot destroy hours of
preprocessing — and can resume from a previous run's output.

**The accelerator is not interchangeable.** The pinned PyTorch build supports
compute capability 7.0 and newer. On Kaggle's P100 (6.0) every CUDA kernel
raises, and because the upstream extraction script catches all exceptions per
clip, the run *looks* healthy while producing zero features. The notebook now
checks the device up front and refuses to start on one it cannot use.

## 5. What came out

Test set: 1,000 clips from the LAV-DF test split, 500 real and 500 fake.

| model | AP | AUC | EER | accuracy @EER | recall @EER | F1 @EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trained here on 3,000 real clips | 0.7872 | 0.8263 | 0.2520 | 0.7480 | 0.7460 | 0.7475 |
| authors' released model, untouched | 0.8272 | 0.8659 | 0.2140 | 0.7860 | 0.7860 | 0.7860 |

Training converged on its own: the best validation loss (1.051858) came at epoch
27, and training stopped at epoch 37 after ten epochs without improvement — it
was not cut off by the session clock. Total run time was 2 h 16 m.

Tested against each other on identical clips, the released model is ahead by
**ΔAUC 0.0396** (95% CI 0.0188 to 0.0604, p = 0.0005). That is the expected
direction: it was trained on 45,000 clips, ours on 3,000.

## 6. What these numbers may be claimed to be

- They are a **scale-reduced baseline**, not a reproduction of the published
  training run.
- **AUC is the number to compare across papers.** Our test set was deliberately
  balanced 50/50, so AP starts from a 0.5 base rate, whereas the real LAV-DF test
  split is 26,100 clips at 73.5% fake and published AP sits on that.
- About 1% of clips carry full frames instead of mouth crops, because no face was
  detected. That is upstream behaviour, not a modification.

## 7. How to reuse it

- To **repeat the run**: attach LAV-DF plus the saved features dataset and run
  the notebook; about 2.5 hours.
- To **run everything from raw video**: attach LAV-DF only; about 7 hours.
- To **score another model against this one**: score it on
  `splits/test_metadata.csv` — the exact clips used here — write one row per clip
  with a score, then run `compare_models.py`, which checks that every model saw
  every clip and reports paired bootstrap differences. `metrics_from_scores.py`
  gives the full metric suite for a single model.
- To **evaluate on the whole test split** (26,100 clips at their natural class
  mix): `_fulltest.py` splits that work across sessions — CPU sessions for the
  preprocessing, one GPU session for scoring.

Everything above can be re-run from this repository alone, and the Kaggle
notebooks that produced the numbers are public, so their logs and outputs can be
read directly.

## 8. Glossary

| term | meaning here |
| --- | --- |
| mouth ROI | the cropped, aligned 96×96 greyscale mouth video |
| feature / `.npz` | frozen AV-HuBERT output for one clip: per-frame visual and audio vectors |
| alignment head | the small trained network that scores audio-visual correspondence |
| τ (tau) | half-width of the audio window; τ = 15 gives a 31-frame window |
| AP | average precision — area under precision-recall; depends on the fake/real mix |
| AUC | area under the ROC curve; independent of the fake/real mix |
| EER | the threshold where false accepts equal false rejects |
| zero-shot | applying a model without training it on this dataset |
