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
B = 500
rng = np.random.default_rng(0)

# ---- LLM ratings (0-10), written directly in the script ----
# PLACEHOLDER values: replace with real ratings from a fresh LLM chat.
ratings = {"age": 6, "sex": 5, "cp": 9, "trestbps": 5, "chol": 5, "fbs": 3,
           "restecg": 4, "thalach": 7, "exang": 8, "oldpeak": 8, "slope": 6,
           "ca": 9, "thal": 8}

missing = [c for c in FEATURES if c not in ratings]
assert not missing, f"Missing features in ratings: {missing}"

r = np.array([float(ratings[c]) for c in FEATURES])
w_llm = {c: max(v / r.mean(), 0.05) for c, v in zip(FEATURES, r)}      # mean 1
w_soft = {c: 0.5 + 0.5 * w_llm[c] for c in FEATURES}                   # blend with uniform

CONFIGS = {
    "uniform":  ("uniform", None),
    "mi":       ("mi", None),
    "rf":       ("rf", None),
    "llm":      ("fixed", w_llm),
    "llm_soft": ("fixed", w_soft),
}

data = load_all()
sites = list(data["site"].unique())

# ---- Leave-one-site-out scores ----
scores, labels = {}, {}
for site in sites:
    train = data[data["site"] != site]
    test_df = data[data["site"] == site]
    labels[site] = test_df["target"].values
    for name, (mode, fixed) in CONFIGS.items():
        seeds = [1, 2, 3] if mode in ("mi", "rf") else [0]
        runs = []
        for sd in seeds:
            m = make_weighted_svm(mode, fixed=fixed, seed=sd)
            m.fit(train[FEATURES], train["target"])
            runs.append(m.decision_function(test_df[FEATURES]))
        scores[(name, site)] = np.mean(runs, axis=0)
    print("Done", site)

# ---- Paired stratified bootstrap ----
boot = {(n, s): np.zeros(B) for n in CONFIGS for s in sites}
for site in sites:
    y = labels[site]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for n in CONFIGS:
            boot[(n, site)][b] = roc_auc_score(y[idx], scores[(n, site)][idx])


def macro(name, use):
    point = np.mean([roc_auc_score(labels[s], scores[(name, s)]) for s in use])
    dist = np.mean([boot[(name, s)] for s in use], axis=0)
    return point, dist


def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


# ---- Tables ----
per_site = pd.DataFrame(
    [{"weights": n, **{s: round(roc_auc_score(labels[s], scores[(n, s)]), 3)
                       for s in sites}} for n in CONFIGS]).set_index("weights")
print("\n=== AUC BY HELD-OUT HOSPITAL ===")
print(per_site)

rows = []
for label, use in [("all 4 sites", sites),
                   ("without switzerland", [s for s in sites if s != "switzerland"])]:
    u_point, u_dist = macro("uniform", use)
    for n in CONFIGS:
        p, d = macro(n, use)
        lo, hi = ci(d)
        dlo, dhi = ci(d - u_dist)
        rows.append(dict(sites=label, weights=n, auc=round(p, 3),
                         ci_low=round(lo, 3), ci_high=round(hi, 3),
                         diff_vs_uniform=round(p - u_point, 3),
                         diff_ci_low=round(dlo, 3), diff_ci_high=round(dhi, 3)))
summary = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n=== AVERAGE AUC WITH 95% CI AND DIFFERENCE VS UNIFORM ===")
print(summary.to_string(index=False))

# ---- Do LLM weights agree with data-driven weights? ----
wt = {"llm": w_llm}
for mode in ["mi", "rf"]:
    mm = make_weighted_svm(mode).fit(data[FEATURES], data["target"])
    wt[mode] = mm.named_steps["weights"].weights_
wdf = pd.DataFrame(wt).loc[FEATURES]
print("\n=== FEATURE WEIGHTS (mean = 1) ===")
print(wdf.round(2).sort_values("llm", ascending=False))
print("\nSpearman correlation between weight sources:")
print(wdf.corr(method="spearman").round(2))

per_site.to_csv("results/step7_per_site.csv")
summary.to_csv("results/step7_summary.csv", index=False)
wdf.to_csv("results/step7_weights.csv")