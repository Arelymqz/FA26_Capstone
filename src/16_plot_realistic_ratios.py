from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
csv_path = OUT_DIR / "realistic_ratio_model_results.csv"

def ordered_scenarios(df):
    order = ["original", "10pct_attack", "1pct_attack", "0_1pct_attack"]
    df["scenario"] = pd.Categorical(df["scenario"], categories=order, ordered=True)
    return df.sort_values("scenario")

def make_plot(df, y_col, title, ylabel, out_name):
    plt.figure()
    for model_name, sub in df.groupby("model"):
        plt.plot(sub["scenario"].astype(str), sub[y_col], marker="o", label=model_name)
    plt.title(title)
    plt.xlabel("Test Scenario")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = OUT_DIR / out_name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    df = pd.read_csv(csv_path)
    df = ordered_scenarios(df)

    make_plot(
        df,
        y_col="precision_malicious",
        title="Precision vs Realistic Attack Ratio",
        ylabel="Precision (Malicious)",
        out_name="precision_vs_realistic_ratio.png"
    )

    make_plot(
        df,
        y_col="recall_malicious",
        title="Recall vs Realistic Attack Ratio",
        ylabel="Recall (Malicious)",
        out_name="recall_vs_realistic_ratio.png"
    )

    make_plot(
        df,
        y_col="fp_per_10k_records",
        title="False Positives per 10k Records vs Realistic Attack Ratio",
        ylabel="False Positives per 10k Records",
        out_name="fp_per_10k_vs_realistic_ratio.png"
    )

    make_plot(
        df,
        y_col="alert_rate",
        title="Alert Rate vs Realistic Attack Ratio",
        ylabel="Alert Rate",
        out_name="alert_rate_vs_realistic_ratio.png"
    )

if __name__ == "__main__":
    main()