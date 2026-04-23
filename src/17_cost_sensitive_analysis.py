from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
csv_path = OUT_DIR / "realistic_ratio_model_results.csv"

def main():
    df = pd.read_csv(csv_path)

    # Different cost assumptions
    cost_settings = [
        {"cost_name": "balanced_5x", "fp_cost": 1, "fn_cost": 5},
        {"cost_name": "security_10x", "fp_cost": 1, "fn_cost": 10},
        {"cost_name": "critical_25x", "fp_cost": 1, "fn_cost": 25},
    ]

    rows = []

    for _, row in df.iterrows():
        for cost in cost_settings:
            total_cost = (row["fp"] * cost["fp_cost"]) + (row["fn"] * cost["fn_cost"])
            cost_per_10k = total_cost / row["total_records"] * 10000

            rows.append({
                "scenario": row["scenario"],
                "model": row["model"],
                "threshold": row["threshold"],
                "cost_name": cost["cost_name"],
                "fp": row["fp"],
                "fn": row["fn"],
                "fp_cost": cost["fp_cost"],
                "fn_cost": cost["fn_cost"],
                "total_cost": total_cost,
                "cost_per_10k_records": cost_per_10k,
                "precision_malicious": row["precision_malicious"],
                "recall_malicious": row["recall_malicious"],
            })

    results = pd.DataFrame(rows)

    out_file = OUT_DIR / "cost_sensitive_results.csv"
    results.to_csv(out_file, index=False)

    print(f"✅ Saved: {out_file}")
    print("\n=== Preview ===")
    print(results.head(12))

if __name__ == "__main__":
    main()