#!/usr/bin/env python3
"""Write per-clip AVH-Align scores for a list of clips.

`eval.py` prints only AP and AUC, which is not enough to compare models pairwise
or to plot ROC curves. This does the identical computation -- L2-normalise both
streams, run the fusion model, take logsumexp(-output) -- and writes one score
per clip instead. Run it from inside the AVH-Align checkout, with the extracted
features on hand:

    python3 score_clips.py test_metadata.csv lavdf_feats/val out.csv \
        avhalign=checkpoints/AVH-Align_LAVDF.pt official=checkpoints/AVH-Align_AV1M.pt
"""
import csv
import os
import sys

import numpy as np
import torch

from model import FusionModel

meta, feats, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
ckpts = dict(p.split("=", 1) for p in sys.argv[4:])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
for name, path in ckpts.items():
    m = FusionModel().to(device)
    m.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    m.eval()
    models[name] = m

rows = list(csv.DictReader(open(meta)))
names = list(models)
kept = 0
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "label"] + [f"score_{n}" for n in names])
    for r in rows:
        f = os.path.join(feats, r["path"].replace(".mp4", ".npz"))
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        v = torch.from_numpy(d["visual"]).to(device)
        a = torch.from_numpy(d["audio"]).to(device)
        v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)
        a = a / torch.linalg.norm(a, ord=2, dim=-1, keepdim=True)
        with torch.no_grad():
            scores = [float(torch.logsumexp(-models[n](v, a), dim=0).cpu().squeeze())
                      for n in names]
        w.writerow([r["path"], r.get("label", "")] + scores)
        kept += 1
print(f"scored {kept}/{len(rows)} clips -> {out_csv}")
