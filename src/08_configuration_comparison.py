from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
csv_path = OUT_DIR / "threshold_sweep_results.csv"

df = pd.read_csv(csv_path)

# Select representative thresholds
thresholds = [0.05, 0.30, 0.90]

comparison = df[df["threshold"].isin(thresholds)].copy()

comparison["configuration"] = [
    "Recall Optimized",
    "Balanced",
    "Precision Optimized"
]

# Reorder columns for better readability
comparison = comparison[
    [
        "configuration",
        "threshold",
        "recall_malicious",
        "precision_malicious",
        "fp",
        "fn"
    ]
]

print(comparison)

save_path = OUT_DIR / "configuration_comparison.csv"
comparison.to_csv(save_path, index=False)

print("Saved comparison table:", save_path)