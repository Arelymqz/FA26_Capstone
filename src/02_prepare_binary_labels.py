from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH  = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41      # attack-type label
DIFF_COL  = 42      # difficulty (we'll ignore)

def load_nsl_kdd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    return df

def to_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Create y: 0 = benign (normal), 1 = malicious (anything else)
    y = (df.iloc[:, LABEL_COL] != "normal").astype(int)
    X = df.drop(columns=[LABEL_COL, DIFF_COL])
    return X, y

def main():
    train_df = load_nsl_kdd(TRAIN_PATH)
    test_df  = load_nsl_kdd(TEST_PATH)

    X_train, y_train = to_binary_labels(train_df)
    X_test,  y_test  = to_binary_labels(test_df)

    def summarize(y, name):
        total = int(y.shape[0])
        malicious = int(y.sum())
        benign = total - malicious
        print(f"\n=== {name} label distribution (binary) ===")
        print(f"Total: {total}")
        print(f"Benign (0): {benign} ({benign/total:.2%})")
        print(f"Malicious (1): {malicious} ({malicious/total:.2%})")

    summarize(y_train, "TRAIN")
    summarize(y_test, "TEST")

    # Save a small CSV summary for your report/progress
    summary = pd.DataFrame({
        "split": ["train", "test"],
        "total": [len(y_train), len(y_test)],
        "benign": [int((y_train == 0).sum()), int((y_test == 0).sum())],
        "malicious": [int((y_train == 1).sum()), int((y_test == 1).sum())],
    })
    summary["benign_pct"] = summary["benign"] / summary["total"]
    summary["malicious_pct"] = summary["malicious"] / summary["total"]

    out_path = OUT_DIR / "class_balance_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
