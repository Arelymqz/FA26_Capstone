from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

csv_path = DATA_DIR / "model_category_detection_selected_thresholds.csv"

df = pd.read_csv(csv_path)

# Focus on main attack categories for clearer visualization
df = df[df["category"].isin(["DoS", "Probe", "R2L", "U2R"])]

# Create a combined column for model and threshold to use as heatmap columns
df["model_threshold"] = df["model"] + " @ " + df["threshold"].astype(str)

# Pivot the data to create a matrix of detection rates for the heatmap
heatmap_df = df.pivot(
    index="category",
    columns="model_threshold",
    values="detection_rate"
)

# Sort categories for readability
heatmap_df = heatmap_df.loc[["DoS", "Probe", "R2L", "U2R"]]

plt.figure(figsize=(12,5))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1
)

plt.title("Attack Detection Rate by Model and Threshold")
plt.xlabel("Model / Threshold")
plt.ylabel("Attack Category")

plt.tight_layout()

out_file = OUT_DIR / "attack_detection_heatmap.png"
plt.savefig(out_file, dpi=300)

print("Saved:", out_file)