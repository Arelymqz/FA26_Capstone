from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
csv_path = OUT_DIR / "threshold_sweep_results.csv"

# This script reads the results from the threshold sweep experiment and creates a line plot showing how the number of false negatives (missed attacks) changes as we adjust the decision threshold. This helps visualize the trade-off between recall and precision at different thresholds.
df = pd.read_csv(csv_path)

df = df.sort_values("threshold")

plt.figure()

plt.plot(df["threshold"], df["fn"], marker="o")

plt.xlabel("Decision Threshold")
plt.ylabel("False Negatives (Missed Attacks)")
plt.title("Missed Attacks vs Threshold")

plt.grid(True)

save_path = OUT_DIR / "false_negatives_vs_threshold.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print("Saved plot:", save_path)