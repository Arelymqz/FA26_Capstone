from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

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

def main():
    # --- load ---
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Keep original attack names for category analysis
    test_attack_names = test_df.iloc[:, LABEL_COL].astype(str)

    # Binary labels for training/testing
    y_train = (train_df.iloc[:, LABEL_COL] != "normal").astype(int)
    y_test = (test_df.iloc[:, LABEL_COL] != "normal").astype(int)

    X_train = train_df.drop(columns=[LABEL_COL, DIFF_COL])
    X_test = test_df.drop(columns=[LABEL_COL, DIFF_COL])

    # One-hot encode
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Fix feature-name types to avoid sklearn error
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Scale (sparse-friendly)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train baseline classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    # Probabilities for threshold sweep
    probs = model.predict_proba(X_test_s)[:, 1]

    # Map all test rows to categories (for totals)
    all_categories = test_attack_names.map(attack_to_category)
    attack_only_categories = all_categories[all_categories != "benign"]

    # Total attacks in each category in the test set (constant across thresholds)
    total_attack_counts = attack_only_categories.value_counts()

    # Choose thresholds to evaluate
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

    rows = []
    for thr in thresholds:
        y_pred = (probs >= thr).astype(int)

        # False negatives = missed attacks
        fn_mask = (y_test == 1) & (y_pred == 0)
        missed_attack_names = test_attack_names[fn_mask]
        missed_categories = missed_attack_names.map(attack_to_category)

        missed_counts = missed_categories.value_counts()

        # Build per-category summary for this threshold
        for cat, total in total_attack_counts.items():
            missed = int(missed_counts.get(cat, 0))
            detection_rate = 1 - (missed / total) if total > 0 else 0.0
            rows.append({
                "threshold": thr,
                "category": cat,
                "missed_attacks_fn": missed,
                "total_attacks_in_test": int(total),
                "detection_rate": float(detection_rate),
            })

    results = pd.DataFrame(rows).sort_values(["threshold", "category"])

    # Save long-form CSV (best format for plotting + reporting)
    out_csv = OUT_DIR / "missed_attacks_by_category_threshold_sweep.csv"
    results.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)
    print("\n=== Preview ===")
    print(results.head(12))

    # Pivot for plotting
    pivot_det = results.pivot(index="threshold", columns="category", values="detection_rate")
    pivot_missed = results.pivot(index="threshold", columns="category", values="missed_attacks_fn")

    # Plot 1: Detection rate vs threshold (one plot, multiple lines)
    plt.figure()
    for cat in pivot_det.columns:
        plt.plot(pivot_det.index, pivot_det[cat], marker="o", label=cat)
    plt.title("Detection Rate vs Threshold by Attack Category")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Detection Rate (1 - FN/Total)")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png1 = OUT_DIR / "detection_rate_vs_threshold_by_category.png"
    plt.savefig(out_png1, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png1)

    # Plot 2: Missed attacks (FN count) vs threshold (one plot, multiple lines)
    plt.figure()
    for cat in pivot_missed.columns:
        plt.plot(pivot_missed.index, pivot_missed[cat], marker="o", label=cat)
    plt.title("Missed Attacks (False Negatives) vs Threshold by Category")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Missed Attacks (FN count)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png2 = OUT_DIR / "missed_attacks_fn_vs_threshold_by_category.png"
    plt.savefig(out_png2, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png2)

    # Optional: print the category table for one “presentation threshold”
    presentation_thr = 0.50
    print(f"\n=== Table at threshold={presentation_thr:.2f} ===")
    print(results[results["threshold"] == presentation_thr].set_index("category"))

if __name__ == "__main__":
    main()