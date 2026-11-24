import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from load_data import load_all
from pipeline import FEATURES
from kernel_svm import make_weighted_svm

os.makedirs("results", exist_ok=True)

data = load_all()
sites = list(data["site"].unique())
rows = []

for mode in ["uniform", "mi", "rf"]:
    for held_out in sites:
        train = data[data["site"] != held_out]
        test_df = data[data["site"] == held_out]
        m = make_weighted_svm(mode).fit(train[FEATURES], train["target"])
        s = m.decision_function(test_df[FEATURES])
        y = test_df["target"].values
        rows.append(dict(mode=mode, held_out=held_out,
                         auc=roc_auc_score(y, s),
                         bal_acc=balanced_accuracy_score(y, (s > 0).astype(int))))

res = pd.DataFrame(rows)
res.to_csv("results/kernel_loso.csv", index=False)

pd.set_option("display.width", 200)
print("=== AUC BY HELD-OUT HOSPITAL ===")
print(res.pivot_table(index="mode", columns="held_out", values="auc").round(3))
print("\n=== AVERAGE ===")
print(res.groupby("mode")[["auc", "bal_acc"]].mean().round(3))