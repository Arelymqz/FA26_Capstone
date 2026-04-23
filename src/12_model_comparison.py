from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / "KDDTrain+.txt"
test_path = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41
DIFF_COL = 42

# --- NSL-KDD attack -> category mapping (standard) ---
DOS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "apache2", "processtable", "udpstorm"
}
PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "worm", "xlock", "xsnoop"
}
U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps", "sqlattack", "xterm"}

# This script trains multiple models (Logistic Regression, Random Forest, Gradient Boosting) and compares their performance across a range of decision thresholds. It evaluates both overall metrics (recall, precision, false positives) and category-specific detection rates for different attack types. The results are saved in CSV files and visualized with line plots and bar charts.
def attack_to_category(label: str) -> str:
    if label == "normal":
        return "benign"
    if label in DOS:
        return "DoS"
    if label in PROBE:
        return "Probe"
    if label in R2L:
        return "R2L"
    if label in U2R:
        return "U2R"
    return "Other"

# Helper functions for metrics and plotting are defined below, followed by the main() function that orchestrates the training, evaluation, and visualization of the models.
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # NSL-KDD: cols 1,2,3 are categorical in the classic KDD format:
    # 1=protocol_type, 2=service, 3=flag (0-based indexing)
    cat_cols = [1, 2, 3]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # allow sparse output when helpful
    )
    return pre

# The main() function is defined at the end of the file to keep the helper functions organized at the top. It loads the data, trains each model, evaluates metrics across thresholds, and generates plots for comparison.
def threshold_metrics(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    alert_rate = (tp + fp) / len(y_true)

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall_malicious": recall,
        "precision_malicious": precision,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "alert_rate": alert_rate,
    }

def category_detection(y_true, probs, test_attack_names, threshold):
    y_pred = (probs >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)

    missed_attacks = test_attack_names[fn_mask]
    missed_categories = missed_attacks.map(attack_to_category)

    missed_counts = missed_categories.value_counts()
    all_cats = test_attack_names.map(attack_to_category)
    total_attack_counts = all_cats[all_cats != "benign"].value_counts()

    summary = pd.DataFrame({
        "missed_attacks_fn": missed_counts,
        "total_attacks_in_test": total_attack_counts
    }).fillna(0).astype(int)

    summary["detection_rate"] = 1 - (summary["missed_attacks_fn"] / summary["total_attacks_in_test"]).replace({0: np.nan})
    summary["detection_rate"] = summary["detection_rate"].fillna(0)

    summary.index.name = "category"
    summary = summary.reset_index()
    summary.insert(0, "threshold", threshold)
    return summary

def plot_tradeoff(combined_df: pd.DataFrame, outpath: Path):
    plt.figure()
    for model_name, sub in combined_df.groupby("model"):
        plt.plot(sub["threshold"], sub["recall_malicious"], marker="o", label=f"{model_name} recall")
    plt.title("Recall vs Threshold (Malicious) — Model Comparison")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_fp_vs_threshold(combined_df: pd.DataFrame, outpath: Path):
    plt.figure()
    for model_name, sub in combined_df.groupby("model"):
        plt.plot(sub["threshold"], sub["fp"], marker="o", label=f"{model_name} FP")
    plt.title("False Positives vs Threshold — Model Comparison")
    plt.xlabel("Decision Threshold")
    plt.ylabel("False Positives (count)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_category_at_threshold(cat_df: pd.DataFrame, threshold: float, outpath: Path):
    # expects columns: model, category, detection_rate
    sub = cat_df[cat_df["threshold"] == threshold].copy()
    # keep only standard four categories
    sub = sub[sub["category"].isin(["DoS", "Probe", "R2L", "U2R"])]

    # pivot for a grouped-bar style without seaborn
    pivot = sub.pivot(index="category", columns="model", values="detection_rate").fillna(0)
    pivot = pivot.loc[["DoS", "Probe", "R2L", "U2R"]]

    ax = pivot.plot(kind="bar")
    ax.set_title(f"Detection Rate by Category @ threshold={threshold}")
    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Detection Rate (1 - FN/Total)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    test_attack_names = test_df.iloc[:, LABEL_COL].astype(str)

    y_train = (train_df.iloc[:, LABEL_COL] != "normal").astype(int).to_numpy()
    y_test = (test_df.iloc[:, LABEL_COL] != "normal").astype(int).to_numpy()

    X_train = train_df.drop(columns=[LABEL_COL, DIFF_COL])
    X_test = test_df.drop(columns=[LABEL_COL, DIFF_COL])

    pre = build_preprocessor(X_train)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=None),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }

    thresholds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90, 0.95])

    all_metric_rows = []
    all_cat_rows = []

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("model", clf)])
        print(f"\n=== Training: {name} ===")
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]

        # threshold sweep metrics
        for t in thresholds:
            row = threshold_metrics(y_test, probs, t)
            row["model"] = name
            all_metric_rows.append(row)

        # category detection at a few “reporting” thresholds
        for t in [0.05, 0.30, 0.90]:
            cat_summary = category_detection(y_test, probs, test_attack_names, t)
            cat_summary.insert(0, "model", name)
            all_cat_rows.append(cat_summary)

    metrics_df = pd.DataFrame(all_metric_rows).sort_values(["model", "threshold"])
    cats_df = pd.concat(all_cat_rows, ignore_index=True)

    # save CSVs
    metrics_csv = OUT_DIR / "model_threshold_sweep_results.csv"
    cats_csv = OUT_DIR / "model_category_detection_selected_thresholds.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    cats_df.to_csv(cats_csv, index=False)
    print("\n✅ Saved:", metrics_csv)
    print("✅ Saved:", cats_csv)

    # plots
    plot_tradeoff(metrics_df, OUT_DIR / "model_recall_vs_threshold.png")
    plot_fp_vs_threshold(metrics_df, OUT_DIR / "model_fp_vs_threshold.png")
    plot_category_at_threshold(cats_df, 0.30, OUT_DIR / "model_detection_by_category_t0_30.png")
    plot_category_at_threshold(cats_df, 0.05, OUT_DIR / "model_detection_by_category_t0_05.png")
    plot_category_at_threshold(cats_df, 0.90, OUT_DIR / "model_detection_by_category_t0_90.png")

    print("✅ Saved plots into:", OUT_DIR)

if __name__ == "__main__":
    main()