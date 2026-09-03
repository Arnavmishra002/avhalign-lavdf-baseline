#!/usr/bin/env python3
"""Compare several detectors on one fixed clip list.

Every model must be scored on the SAME clips with the SAME labels, otherwise the
numbers are not comparable. This script enforces that: it joins each model's
per-clip scores onto the label file, refuses to proceed on partial coverage, and
reports AP / AUC with bootstrap confidence intervals plus paired bootstrap tests
for every model pair (paired, because the models see identical clips -- an
unpaired test would throw away that structure and overstate the uncertainty).

    python3 compare_models.py \
        --labels splits/test_metadata.csv \
        --scores avh-align=scores/avhalign.csv ours=scores/ours.csv third=scores/third.csv

Label file: CSV with columns `path` and `label` (1 = fake, 0 = real).
Score file: CSV with columns `path` and `score` (higher = more likely fake).
"""
import argparse
import csv
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_labels(path):
    rows = list(csv.DictReader(open(path)))
    return {r["path"]: int(r["label"]) for r in rows}


def read_scores(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path} is empty")
    col = "score" if "score" in rows[0] else next(
        (k for k in rows[0] if k.startswith("score")), None)
    if col is None:
        sys.exit(f"{path} has no score column (found {list(rows[0])})")
    return {r["path"]: float(r[col]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", nargs="+", required=True,
                    help="name=path.csv, one per model")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = read_labels(args.labels)
    models = {}
    for spec in args.scores:
        if "=" not in spec:
            sys.exit(f"--scores wants name=path.csv, got {spec!r}")
        name, path = spec.split("=", 1)
        models[name] = read_scores(path)

    clips = sorted(labels)
    for name, sc in models.items():
        missing = [c for c in clips if c not in sc]
        if missing:
            sys.exit(f"{name} is missing {len(missing)} of {len(clips)} clips, "
                     f"e.g. {missing[:3]} -- score the full list before comparing")

    y = np.array([labels[c] for c in clips])
    S = {n: np.array([models[n][c] for c in clips]) for n in models}

    rng = np.random.default_rng(args.seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.boot)]
    idx = [i for i in idx if 0 < y[i].sum() < len(i)]          # keep both classes

    print(f"clips: {len(y)}   real {int((y == 0).sum())}   fake {int(y.sum())}   "
          f"fake prior {y.mean():.3f}")
    print(f"bootstrap resamples: {len(idx)} (seed {args.seed})\n")

    print("| model | AP | AP 95% CI | AUC | AUC 95% CI |")
    print("| --- | ---: | :---: | ---: | :---: |")
    boot = {}
    for n, s in S.items():
        aps = np.array([average_precision_score(y[i], s[i]) for i in idx])
        aucs = np.array([roc_auc_score(y[i], s[i]) for i in idx])
        boot[n] = (aps, aucs)
        print(f"| {n} | {average_precision_score(y, s):.4f} | "
              f"[{np.percentile(aps, 2.5):.4f}, {np.percentile(aps, 97.5):.4f}] | "
              f"{roc_auc_score(y, s):.4f} | "
              f"[{np.percentile(aucs, 2.5):.4f}, {np.percentile(aucs, 97.5):.4f}] |")

    names = list(S)
    if len(names) > 1:
        print("\nPaired differences (row minus column), AUC:")
        print("| A vs B | ΔAUC | 95% CI | p (two-sided) |")
        print("| --- | ---: | :---: | ---: |")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = boot[a][1] - boot[b][1]
                obs = roc_auc_score(y, S[a]) - roc_auc_score(y, S[b])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                print(f"| {a} vs {b} | {obs:+.4f} | "
                      f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] | "
                      f"{max(p, 1 / len(d)):.4f} |")


if __name__ == "__main__":
    main()
