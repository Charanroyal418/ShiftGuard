import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from load_data import load_all
from pipeline import FEATURES, MODELS
from metrics import all_metrics

data = load_all()
sites = list(data["site"].unique())
rows = []

for name, make in MODELS.items():
    for held_out in sites:
        train = data[data["site"] != held_out]
        test = data[data["site"] == held_out]
        model = make().fit(train[FEATURES], train["target"])
        proba = model.predict_proba(test[FEATURES])[:, 1]
        row = all_metrics(test["target"], proba)
        row.update(model=name, held_out=held_out)
        rows.append(row)

res = pd.DataFrame(rows)
res.to_csv("results/loso_results.csv", index=False)

print("=== AUC BY HELD-OUT HOSPITAL ===")
print(res.pivot_table(index="model", columns="held_out", values="auc").round(3))
print("\n=== AVERAGE OVER HELD-OUT HOSPITALS ===")
print(res.groupby("model")[["auc", "bal_acc", "brier", "ece"]].mean().round(3))