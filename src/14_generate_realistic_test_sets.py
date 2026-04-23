from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
OUT_DIR.mkdir(parents=True, exist_ok=True)

test_path = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41

def make_resampled_test(df: pd.DataFrame, attack_ratio: float, random_state: int = 42) -> pd.DataFrame:
    """
    Create a new test set with a target malicious attack ratio.
    Keeps all benign rows and downsamples malicious rows to achieve the ratio.
    """
    benign_df = df[df.iloc[:, LABEL_COL] == "normal"].copy()
    malicious_df = df[df.iloc[:, LABEL_COL] != "normal"].copy()

    benign_count = len(benign_df)

    # desired malicious count from:
    # malicious / (benign + malicious) = attack_ratio
    # => malicious = attack_ratio * benign / (1 - attack_ratio)
    desired_malicious = int((attack_ratio * benign_count) / (1 - attack_ratio))

    desired_malicious = min(desired_malicious, len(malicious_df))

    malicious_sample = malicious_df.sample(
        n=desired_malicious,
        random_state=random_state,
        replace=False
    )

    resampled = pd.concat([benign_df, malicious_sample], axis=0)
    resampled = resampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return resampled

def summarize_ratio(df: pd.DataFrame) -> dict:
    benign = (df.iloc[:, LABEL_COL] == "normal").sum()
    malicious = (df.iloc[:, LABEL_COL] != "normal").sum()
    total = len(df)

    return {
        "total": total,
        "benign": int(benign),
        "malicious": int(malicious),
        "benign_pct": benign / total,
        "malicious_pct": malicious / total,
    }

def main():
    test_df = pd.read_csv(test_path, header=None)

    scenarios = {
        "original": None,
        "10pct_attack": 0.10,
        "1pct_attack": 0.01,
        "0_1pct_attack": 0.001,
    }

    summary_rows = []

    for name, ratio in scenarios.items():
        if ratio is None:
            out_df = test_df.copy()
        else:
            out_df = make_resampled_test(test_df, attack_ratio=ratio)

        out_file = OUT_DIR / f"{name}_test.csv"
        out_df.to_csv(out_file, header=False, index=False)

        stats = summarize_ratio(out_df)
        stats["scenario"] = name
        summary_rows.append(stats)

        print(f"✅ Saved: {out_file}")
        print(f"   benign={stats['benign']} ({stats['benign_pct']:.2%}), "
              f"malicious={stats['malicious']} ({stats['malicious_pct']:.2%})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        ["scenario", "total", "benign", "malicious", "benign_pct", "malicious_pct"]
    ]

    summary_path = OUT_DIR / "realistic_ratio_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Saved summary: {summary_path}")

if __name__ == "__main__":
    main()