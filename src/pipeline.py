from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])

def make_logreg(seed=42):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
def make_rf(seed=42):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, random_state=seed)),
    ])

def make_svm(seed=42):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", SVC(kernel="rbf", probability=True, random_state=seed)),
    ])

def make_mlp(seed=42):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                                random_state=seed)),
    ])

MODELS = {
    "LogReg": make_logreg,
    "RandomForest": make_rf,
    "SVM": make_svm,
    "NeuralNet": make_mlp,
}