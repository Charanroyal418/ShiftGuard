import pandas as pd
from sklearn.model_selection import StratifiedKFold
from load_data import load_all
from pipeline import FEATURES, MODELS
from metrics import all_metrics
import warnings
warnings.filterwarnings("ignore")
data = load_all()
sites = list(data["site"].unique())
rows = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, make in MODELS.items():
    print("Running", name, "...")

    # A) Pooled: all hospitals mixed, 5-fold CV
    for train_idx, test_idx in skf.split(data, data["target"]):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        model = make().fit(train[FEATURES], train["target"])
        proba = model.predict_proba(test[FEATURES])[:, 1]
        row = all_metrics(test["target"], proba)
        row.update(model=name, setting="pooled", train="all", test="all")
        rows.append(row)

    # B) Shift: train on one hospital, test on another
    for tr_site in sites:
        train = data[data["site"] == tr_site]
        model = make().fit(train[FEATURES], train["target"])
        for te_site in sites:
            if te_site == tr_site:
                continue
            test = data[data["site"] == te_site]
            proba = model.predict_proba(test[FEATURES])[:, 1]
            row = all_metrics(test["target"], proba)
            row.update(model=name, setting="cross-site", train=tr_site, test=te_site)
            rows.append(row)

results = pd.DataFrame(rows)
results.to_csv("results/step3_all_results.csv", index=False)

summary = results.groupby(["setting", "model"]).mean(numeric_only=True).round(3)
summary.to_csv("results/step3_summary.csv")

pd.set_option("display.width", 200)
print("\n=== SUMMARY (average over folds / site pairs) ===")
print(summary)