from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
csv_path = OUT_DIR / "threshold_sweep_results.csv"

def main():
    df = pd.read_csv(csv_path)

    # Ensure sorted by threshold
    df = df.sort_values("threshold")

    # --- Combined plot (two y-axes) ---
    fig, ax1 = plt.subplots()

    # Left axis: recall + precision (0 to 1)
    ax1.plot(df["threshold"], df["recall_malicious"], marker="o", label="Recall (malicious)")
    ax1.plot(df["threshold"], df["precision_malicious"], marker="o", label="Precision (malicious)")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Recall / Precision")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Right axis: false positives (count)
    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["fp"], marker="o", linestyle="--", label="False Positives (count)")
    ax2.set_ylabel("False Positives (count)")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Threshold Tradeoff: Recall vs Precision vs False Positives")

    save_path = OUT_DIR / "combined_threshold_tradeoff.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Saved combined plot:", save_path)

if __name__ == "__main__":
    main()