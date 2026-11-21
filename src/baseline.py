from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from load_data import load_all
from pipeline import make_logreg, FEATURES

data = load_all()
X, y = data[FEATURES], data["target"]

# A) Normal test: all hospitals mixed together
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
scores = cross_val_score(make_logreg(), X, y, cv=cv, scoring="roc_auc")
print(f"Pooled CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

# B) Distribution shift: train on one hospital, test on another
for train_site in data["site"].unique():
    train = data[data["site"] == train_site]
    model = make_logreg().fit(train[FEATURES], train["target"])
    for test_site in data["site"].unique():
        if test_site == train_site:
            continue
        test = data[data["site"] == test_site]
        proba = model.predict_proba(test[FEATURES])[:, 1]
        auc = roc_auc_score(test["target"], proba)
        print(f"train={train_site:12s} test={test_site:12s} AUC={auc:.3f}")