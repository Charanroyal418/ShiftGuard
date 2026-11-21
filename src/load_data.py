import pandas as pd
import numpy as np

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

def load_all(raw_dir="data/raw"):
    frames = []
    for site, fname in SITES.items():
        df = pd.read_csv(f"{raw_dir}/{fname}", header=None,
                         names=COLUMNS, na_values="?")
        df["site"] = site
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["target"] = (data["num"] > 0).astype(int)  # 0 = no disease, 1 = disease

    # Zeros here are impossible values, i.e. missing measurements
    for col in ["chol", "trestbps"]:
        data.loc[data[col] == 0, col] = np.nan

    return data

if __name__ == "__main__":
    data = load_all()
    print("Shape:", data.shape)
    print("\nRows per site:\n", data["site"].value_counts())
    print("\nDisease rate per site:\n", data.groupby("site")["target"].mean().round(2))
    print("\nMissing values (%):\n",
          (data.isna().mean() * 100).round(1).sort_values(ascending=False))