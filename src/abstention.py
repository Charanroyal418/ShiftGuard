import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from load_data import load_all
from pipeline import FEATURES, make_logreg, make_rf
from metrics import expected_calibration_error
from sklearn.metrics import brier_score_loss, roc_auc_score

os.makedirs("results", exist_ok=True)

data = load_all()
sites = list(data["site"].unique())

BASE = {"LogReg": make_logreg, "RandomForest": make_rf}
VARIANTS = ["raw", "sigmoid", "isotonic"]   # raw = no calibration

def build(make, variant):
    if variant == "raw":
        return make()
    return CalibratedClassifierCV(estimator=make(), method=variant, cv=3)

rows = []
preds = []   # stores every test prediction for the abstention part

for name, make in BASE.items():
    for variant in VARIANTS:
        for tr in sites:
            train = data[data["site"] == tr]
            model = build(make, variant).fit(train[FEATURES], train["target"])
            for te in sites:
                if te == tr:
                    continue
                test = data[data["site"] == te]
                p = model.predict_proba(test[FEATURES])[:, 1]
                y = test["target"].values
                rows.append(dict(model=name, calib=variant, train=tr, test=te,
                                 auc=roc_auc_score(y, p),
                                 brier=brier_score_loss(y, p),
                                 ece=expected_calibration_error(y, p)))
                preds.append(pd.DataFrame(dict(model=name, calib=variant,
                                               y=y, p=p)))
    print("Done", name)

res = pd.DataFrame(rows)
res.to_csv("results/step4_calibration_pairs.csv", index=False)
summary = res.groupby(["model", "calib"])[["auc", "brier", "ece"]].mean().round(3)
summary.to_csv("results/step4_calibration_summary.csv")
print("\n=== CALIBRATION UNDER SHIFT (average over site pairs) ===")
print(summary)

# ---- Selective prediction (abstention) ----
allp = pd.concat(preds, ignore_index=True)
allp["conf"] = np.maximum(allp["p"], 1 - allp["p"])   # confidence in its own answer
allp["correct"] = ((allp["p"] >= 0.5).astype(int) == allp["y"]).astype(int)

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
sel_rows = []
for (m, c), g in allp.groupby(["model", "calib"]):
    for t in thresholds:
        kept = g[g["conf"] >= t]
        sel_rows.append(dict(model=m, calib=c, threshold=t,
                             coverage=round(len(kept) / len(g), 3),
                             accuracy=round(kept["correct"].mean(), 3)
                             if len(kept) else np.nan))
sel = pd.DataFrame(sel_rows)
sel.to_csv("results/step4_selective.csv", index=False)
print("\n=== ABSTENTION: coverage vs accuracy (all cross-site predictions) ===")
print(sel.to_string(index=False))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 5))
for (m, c), g in sel.groupby(["model", "calib"]):
    ax.plot(g["coverage"], g["accuracy"], marker="o", label=f"{m} / {c}")
ax.set_xlabel("Coverage (share of patients the model answers)")
ax.set_ylabel("Accuracy on answered patients")
ax.set_title("Accuracy vs coverage under site shift")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/step4_selective.png", dpi=150)
print("\nSaved plot to results/step4_selective.png")