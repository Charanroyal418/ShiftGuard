import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pipeline import make_preprocessor


def original_feature(col):
    """'num__age' -> 'age', 'cat__cp_3.0' -> 'cp'"""
    name = col.split("__", 1)[1]
    return name.rsplit("_", 1)[0] if col.startswith("cat__") else name


class FeatureWeighter(BaseEstimator, TransformerMixin):
    """Scales each column by sqrt(weight of its original feature).
    mode: 'uniform', 'mi', 'rf', or 'fixed' (pass a dict in `fixed`)."""

    def __init__(self, mode="uniform", fixed=None, seed=42):
        self.mode = mode
        self.fixed = fixed
        self.seed = seed

    def fit(self, X, y=None):
        cols = list(X.columns)
        feats = [original_feature(c) for c in cols]
        names = sorted(set(feats))

        if self.mode == "uniform":
            col_score = np.ones(len(cols))
        elif self.mode == "mi":
            col_score = mutual_info_classif(X.values, y, random_state=self.seed)
        elif self.mode == "rf":
            rf = RandomForestClassifier(n_estimators=300, random_state=self.seed)
            col_score = rf.fit(X.values, y).feature_importances_
        elif self.mode == "fixed":
            col_score = None
        else:
            raise ValueError(f"unknown mode {self.mode}")

        if self.mode == "fixed":
            score = {f: float(self.fixed.get(f, 1.0)) for f in names}
        elif self.mode == "uniform":
            score = {f: 1.0 for f in names}
        else:
            score = {f: 0.0 for f in names}
            for f, s in zip(feats, col_score):
                score[f] += s

        mean = np.mean(list(score.values()))
        w = {f: max(score[f] / mean, 0.05) for f in names}  # mean weight = 1
        self.weights_ = w
        self.scale_ = np.sqrt(np.array([w[f] for f in feats]))
        return self

    def transform(self, X):
        return X.values * self.scale_


def make_weighted_svm(mode="uniform", fixed=None, seed=42):
    return Pipeline([
        ("prep", make_preprocessor().set_output(transform="pandas")),
        ("weights", FeatureWeighter(mode=mode, fixed=fixed, seed=seed)),
        ("model", SVC(kernel="rbf", gamma="scale")),
    ])