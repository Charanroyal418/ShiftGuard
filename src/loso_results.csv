import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from load_data import load_all
from pipeline import FEATURES
from kernel_svm import make_weighted_svm

os.makedirs("results", exist_ok=True)
data = load_all()
sites = list(data["site"].unique())
MODES = ["uniform", "mi", "rf"]
rows = []

def score_row(model, test_df, **info):
    s = model.decision_function(test_df[FEATURES])
    y = test_df["target"].values
    return dict(auc=roc_auc_score(y, s),
                bal_acc=balanced_accuracy_score(y, (s > 0).astype(int)),
                **info)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for mode in MODES:
    print("Running", mode, "...")
    # A) pooled 5-fold CV
    for tr_idx, te_idx in skf.split(data, data["target"]):
        train, test = data.iloc[tr_idx], data.iloc[te_idx]
        m = make_weighted_svm(mode).fit(train[FEATURES], train["target"])
        rows.append(score_row(m, test, mode=mode, setting="pooled",
                              train="all", test="all"))
    # B) cross-site shift
    for tr in sites:
        train = data[data["site"] == tr]
        m = make_weighted_svm(mode).fit(train[FEATURES], train["target"])
        for te in sites:
            if te == tr:
                continue
            test = data[data["site"] == te]
            rows.append(score_row(m, test, mode=mode, setting="cross-site",
                                  train=tr, test=te))

res = pd.DataFrame(rows)
res.to_csv("results/step5_kernel_pairs.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== SUMMARY ===")
print(res.groupby(["setting", "mode"])[["auc", "bal_acc"]].mean().round(3))

print("\n=== AUC PER SITE PAIR ===")
cross = res[res["setting"] == "cross-site"]
print(cross.pivot_table(index=["train", "test"], columns="mode",
                        values="auc").round(3))

print("\n=== LEARNED FEATURE WEIGHTS (fit on all data) ===")
for mode in ["mi", "rf"]:
    m = make_weighted_svm(mode).fit(data[FEATURES], data["target"])
    w = m.named_steps["weights"].weights_
    print(mode, {k: round(v, 2) for k, v in sorted(w.items(), key=lambda kv: -kv[1])})