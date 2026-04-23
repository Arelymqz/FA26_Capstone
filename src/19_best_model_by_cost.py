from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
csv_path = OUT_DIR / "cost_sensitive_results.csv"

def main():
    df = pd.read_csv(csv_path)

    best_rows = []

    for cost_name in df["cost_name"].unique():
        sub = df[df["cost_name"] == cost_name]

        for scenario in sub["scenario"].unique():
            scenario_df = sub[sub["scenario"] == scenario]
            best = scenario_df.loc[scenario_df["total_cost"].idxmin()]

            best_rows.append({
                "cost_name": cost_name,
                "scenario": scenario,
                "best_model": best["model"],
                "threshold": best["threshold"],
                "total_cost": best["total_cost"],
                "cost_per_10k_records": best["cost_per_10k_records"],
                "precision_malicious": best["precision_malicious"],
                "recall_malicious": best["recall_malicious"],
                "fp": best["fp"],
                "fn": best["fn"],
            })

    best_df = pd.DataFrame(best_rows)

    out_file = OUT_DIR / "best_model_by_cost.csv"
    best_df.to_csv(out_file, index=False)

    print(f"✅ Saved: {out_file}")
    print("\n=== Best model by scenario and cost setting ===")
    print(best_df)

if __name__ == "__main__":
    main()