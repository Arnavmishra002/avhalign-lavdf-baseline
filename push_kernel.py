#!/usr/bin/env python3
"""Push avhalign_cells.ipynb to Kaggle as a new version (Save & Run All).

Auth: export KAGGLE_API_TOKEN=<KGAT_... token from kaggle.com/settings/api>.
The installed kaggle CLI (1.7.4.5) predates KGAT tokens, so this talks to the
REST API directly with a bearer header instead.

Pushing replaces the notebook source, applies the settings below, creates a new
version and runs it. Nothing else about the account is touched.
"""
import json, os, sys, urllib.request, urllib.error

TOKEN = os.environ.get("KAGGLE_API_TOKEN")
if not TOKEN:
    sys.exit("set KAGGLE_API_TOKEN first")

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = {
    "slug": "vansika545/avhalign-cells",
    "newTitle": "avhalign_cells",
    "text": open(os.path.join(HERE, "avhalign_cells.ipynb")).read(),
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableTpu": False,
    "enableInternet": True,
    "datasetDataSources": ["elin75/localized-audio-visual-deepfake-dataset-lav-df"],
    "competitionDataSources": [],
    "kernelDataSources": [],
    "modelDataSources": [],
    "categoryIds": [],
    "dockerImagePinningType": "original",
}

req = urllib.request.Request(
    "https://www.kaggle.com/api/v1/kernels/push",
    data=json.dumps(BODY).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN})
try:
    print(urllib.request.urlopen(req).read().decode())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode())
