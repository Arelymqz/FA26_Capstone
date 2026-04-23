from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
csv_path = OUT_DIR / "cost_sensitive_results.csv"

def ordered_scenarios(df):
    order = ["original", "10pct_attack", "1pct_attack", "0_1pct_attack"]
    df["scenario"] = pd.Categorical(df["scenario"], categories=order, ordered=True)
    return df.sort_values("scenario")

def make_plot(df, cost_name, out_name):
    sub = df[df["cost_name"] == cost_name].copy()
    sub = ordered_scenarios(sub)

    plt.figure()
    for model_name, group in sub.groupby("model"):
        plt.plot(
            group["scenario"].astype(str),
            group["cost_per_10k_records"],
            marker="o",
            label=model_name
        )

    plt.title(f"Cost per 10k Records vs Realistic Attack Ratio ({cost_name})")
    plt.xlabel("Test Scenario")
    plt.ylabel("Cost per 10k Records")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = OUT_DIR / out_name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {save_path}")

def main():
    df = pd.read_csv(csv_path)

    make_plot(df, "balanced_5x", "cost_per_10k_balanced_5x.png")
    make_plot(df, "security_10x", "cost_per_10k_security_10x.png")
    make_plot(df, "critical_25x", "cost_per_10k_critical_25x.png")

if __name__ == "__main__":
    main()