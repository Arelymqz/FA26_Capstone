from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "KDDTrain+.txt"
test_path = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41
DIFF_COL = 42

# This script trains a simple logistic regression model and sweeps through different decision thresholds to analyze the trade-offs between recall, precision, false positives, and alert rates. The results are saved in a CSV file and visualized with line plots.
def main():
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Binary labels: 0 benign (normal), 1 malicious (attack)
    y_train = (train_df.iloc[:, LABEL_COL] != "normal").astype(int)
    y_test  = (test_df.iloc[:, LABEL_COL] != "normal").astype(int)

    X_train = train_df.drop(columns=[LABEL_COL, DIFF_COL])
    X_test  = test_df.drop(columns=[LABEL_COL, DIFF_COL])

    # One-hot encode categorical columns automatically
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align columns between train and test
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Ensure all columns are strings (for safety with sparse matrices)
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Scale numeric space 
    scaler = StandardScaler(with_mean=False)  # with_mean=False is safer for sparse-ish data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Threshold sweep 
    thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate metrics
        recall_mal = tp / (tp + fn) if (tp + fn) else 0.0
        precision_mal = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0  # false positive rate
        fnr = fn / (fn + tp) if (fn + tp) else 0.0  # false negative rate
        alert_rate = (tp + fp) / (tp + fp + tn + fn)

        rows.append({
            "threshold": float(round(t, 2)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "recall_malicious": recall_mal,
            "precision_malicious": precision_mal,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "alert_rate": alert_rate
        })

    results = pd.DataFrame(rows)
    csv_path = OUT_DIR / "threshold_sweep_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"✅ Saved results table: {csv_path}")

    # ---- Graph 1: Recall vs Threshold ----
    plt.figure()
    plt.plot(results["threshold"], results["recall_malicious"], marker="o")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Recall (Malicious)")
    plt.title("Recall vs Threshold (Malicious Class)")
    plt.grid(True, alpha=0.3)
    out1 = OUT_DIR / "recall_vs_threshold.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"✅ Saved plot: {out1}")
    plt.close()

    # ---- Graph 2: False Positives vs Threshold ----
    plt.figure()
    plt.plot(results["threshold"], results["fp"], marker="o")
    plt.xlabel("Decision Threshold")
    plt.ylabel("False Positives (count)")
    plt.title("False Positives vs Threshold")
    plt.grid(True, alpha=0.3)
    out2 = OUT_DIR / "false_positives_vs_threshold.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"✅ Saved plot: {out2}")
    plt.close()

    #-- Quick Summary: Alert Rate vs Threshold ----
    print("\n=== Quick Summary (3 thresholds) ===")
    print(results[results["threshold"].isin([0.2, 0.5, 0.8])][
        ["threshold", "recall_malicious", "precision_malicious", "fp", "fn", "alert_rate"]
    ])

if __name__ == "__main__":
    main()