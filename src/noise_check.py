import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from load_data import load_all
from pipeline import FEATURES
from kernel_svm import make_weighted_svm

os.makedirs("results", exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]     # seeds for MI / RF weights
B = 1000                    # bootstrap resamples
MODES = ["uniform", "mi", "rf"]
rng = np.random.default_rng(0)

data = load_all()
sites = list(data["site"].unique())

# 1) Leave-one-site-out scores, averaged over seeds for mi / rf
scores, labels = {}, {}
for site in sites:
    train = data[data["site"] != site]
    test_df = data[data["site"] == site]
    labels[site] = test_df["target"].values
    for mode in MODES:
        seeds = [0] if mode == "uniform" else SEEDS
        runs = []
        for sd in seeds:
            m = make_weighted_svm(mode, seed=sd).fit(train[FEATURES], train["target"])
            runs.append(m.decision_function(test_df[FEATURES]))
        scores[(mode, site)] = np.mean(runs, axis=0)
    print("Done", site)

# 2) Paired stratified bootstrap (same resampled patients for every mode)
boot = {(m, s): np.zeros(B) for m in MODES for s in sites}
for site in sites:
    y = labels[site]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for mode in MODES:
            boot[(mode, site)][b] = roc_auc_score(y[idx], scores[(mode, site)][idx])

def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)

# 3) Per-site table
rows = []
for mode in MODES:
    for site in sites:
        auc = roc_auc_score(labels[site], scores[(mode, site)])
        lo, hi = ci(boot[(mode, site)])
        rows.append(dict(mode=mode, site=site, auc=round(auc, 3),
                         ci_low=round(lo, 3), ci_high=round(hi, 3)))
per_site = pd.DataFrame(rows)
per_site.to_csv("results/noise_per_site.csv", index=False)

# 4) Average over sites, with and without Switzerland, and difference vs uniform
def macro(mode, use):
    point = np.mean([roc_auc_score(labels[s], scores[(mode, s)]) for s in use])
    dist = np.mean([boot[(mode, s)] for s in use], axis=0)
    return point, dist

summary = []
for name, use in [("all 4 sites", sites),
                  ("without switzerland", [s for s in sites if s != "switzerland"])]:
    u_point, u_dist = macro("uniform", use)
    for mode in MODES:
        point, dist = macro(mode, use)
        lo, hi = ci(dist)
        d_lo, d_hi = ci(dist - u_dist)
        summary.append(dict(sites=name, mode=mode, auc=round(point, 3),
                            ci_low=round(lo, 3), ci_high=round(hi, 3),
                            diff_vs_uniform=round(point - u_point, 3),
                            diff_ci_low=round(d_lo, 3), diff_ci_high=round(d_hi, 3)))
summary = pd.DataFrame(summary)
summary.to_csv("results/noise_summary.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== PER SITE (AUC with 95% CI) ===")
print(per_site.to_string(index=False))
print("\n=== AVERAGE OVER SITES (AUC with 95% CI, difference vs uniform) ===")
print(summary.to_string(index=False))