from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "KDDTrain+.txt"
LABEL_COL = 41
DIFF_COL = 42

# This script trains a logistic regression model and analyzes the feature importance by looking at the coefficients. It saves a CSV with all features and their coefficients, and creates bar plots for the most influential features.
def main():
    df = pd.read_csv(train_path, header=None)

    y = (df.iloc[:, LABEL_COL] != "normal").astype(int)
    X = df.drop(columns=[LABEL_COL, DIFF_COL])

    # One-hot encode categorical features
    X = pd.get_dummies(X)
    X.columns = X.columns.astype(str)
    feature_names = X.columns.to_list()

    # Scale
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(Xs, y)

    # Coefficients: positive => pushes toward "malicious"
    coefs = model.coef_.ravel()

    fi = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    # Save full table
    out_csv = OUT_DIR / "logreg_feature_importance.csv"
    fi.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)

    # Plot 1: Top 20 by absolute importance
    topN = 20
    top = fi.head(topN).iloc[::-1]  # reverse for nicer barh order

    plt.figure()
    plt.barh(top["feature"], top["abs_coef"])
    plt.title(f"Logistic Regression Feature Importance (Top {topN}, |coef|)")
    plt.xlabel("|Coefficient| (importance)")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", alpha=0.3)
    out_png1 = OUT_DIR / "logreg_top_abs_feature_importance.png"
    plt.savefig(out_png1, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png1)

    # Plot 2: Top 15 features that push toward malicious (positive coefs)
    top_pos = fi.sort_values("coef", ascending=False).head(15).iloc[::-1]
    plt.figure()
    plt.barh(top_pos["feature"], top_pos["coef"])
    plt.title("Top Features Increasing 'Malicious' Score (Positive Coefs)")
    plt.xlabel("Coefficient (+ => more malicious)")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", alpha=0.3)
    out_png2 = OUT_DIR / "logreg_top_positive_features.png"
    plt.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png2)

    # Plot 3: Top 15 features that push toward benign (negative coefs)
    top_neg = fi.sort_values("coef", ascending=True).head(15).iloc[::-1]
    plt.figure()
    plt.barh(top_neg["feature"], top_neg["coef"])
    plt.title("Top Features Decreasing 'Malicious' Score (Negative Coefs)")
    plt.xlabel("Coefficient (- => more benign)")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", alpha=0.3)
    out_png3 = OUT_DIR / "logreg_top_negative_features.png"
    plt.savefig(out_png3, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png3)

if __name__ == "__main__":
    main()