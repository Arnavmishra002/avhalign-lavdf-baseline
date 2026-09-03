#!/usr/bin/env python3
"""Full metric suite for one model's per-clip scores.

AP and AUC are threshold-free and are the headline numbers. Recall, precision,
F1, accuracy and specificity need an operating point, so three defensible ones
are reported instead of one arbitrary cut: the equal-error-rate threshold, the
threshold that maximises F1, and Youden's J. Intervals are bootstrap over clips.

    python3 metrics_from_scores.py scores.csv --score-col score_retrained
"""
import argparse
import csv

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


def operating_points(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[i] + (1 - tpr[i])) / 2)
    pts = {"EER": float(thr[i]), "Youden J": float(thr[int(np.argmax(tpr - fpr))])}
    best_f1, best_t = -1.0, thr[i]
    for t in np.unique(s):
        _, _, f, _ = precision_recall_fscore_support(
            y, (s >= t).astype(int), average="binary", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    pts["max F1"] = float(best_t)
    return eer, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.scores)))
    # Sort by clip id so the bootstrap draws the same resamples as
    # compare_models.py, which also sorts; otherwise the two tools report
    # slightly different intervals for the same data.
    rows.sort(key=lambda r: r["path"])
    cols = [a.score_col] if a.score_col else [k for k in rows[0] if k.startswith("score")]
    y = np.array([int(r["label"]) for r in rows])
    rng = np.random.default_rng(a.seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(a.boot)]
    idx = [i for i in idx if 0 < y[i].sum() < len(i)]

    print(f"clips {len(y)}   real {(y == 0).sum()}   fake {y.sum()}   "
          f"fake prior {y.mean():.3f}   bootstrap {len(idx)}\n")
    for col in cols:
        s = np.array([float(r[col]) for r in rows])
        aps = np.array([average_precision_score(y[i], s[i]) for i in idx])
        aucs = np.array([roc_auc_score(y[i], s[i]) for i in idx])
        eer, pts = operating_points(y, s)
        print(f"=== {col} ===")
        print(f"AP  {average_precision_score(y, s):.4f} "
              f"[{np.percentile(aps, 2.5):.4f}, {np.percentile(aps, 97.5):.4f}]")
        print(f"AUC {roc_auc_score(y, s):.4f} "
              f"[{np.percentile(aucs, 2.5):.4f}, {np.percentile(aucs, 97.5):.4f}]")
        print(f"EER {eer:.4f}")
        print(f"{'operating point':<16}{'thr':>10}{'acc':>8}{'prec':>8}"
              f"{'recall':>8}{'F1':>8}{'spec':>8}   confusion")
        for name, t in pts.items():
            pred = (s >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            print(f"{name:<16}{t:>10.4f}{accuracy_score(y, pred):>8.4f}{p:>8.4f}"
                  f"{r:>8.4f}{f:>8.4f}{tn / (tn + fp):>8.4f}   "
                  f"TN {tn} FP {fp} FN {fn} TP {tp}")
        print()


if __name__ == "__main__":
    main()
