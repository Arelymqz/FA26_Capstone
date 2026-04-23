from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def main():
    # Try to find common NSL-KDD filenames
    candidates = [
        "KDDTrain+.txt",
        "KDDTrain+.csv",
        "nsl-kdd-train.csv",
        "NSL_KDD_Train.csv",
    ]

    train_path = None
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            train_path = p
            break

    if train_path is None:
        raise FileNotFoundError(
            f"Could not find NSL-KDD training file in {DATA_DIR}. "
            f"Files found: {[x.name for x in DATA_DIR.glob('*')]}"
        )

    print(f"✅ Using dataset file: {train_path.name}")

    # NSL-KDD .txt is comma-separated, usually with NO header
    df = pd.read_csv(train_path, header=None)

    print("\n=== Basic Shape ===")
    print("Rows, Cols:", df.shape)

    print("\n=== First 3 Rows ===")
    print(df.head(3))

    print("\n=== Last 3 Rows ===")
    print(df.tail(3))

    # In NSL-KDD, label is usually the 2nd-to-last column (attack type)
    # The last column is often 'difficulty' level.
    label_col_guess = df.shape[1] - 2
    print(f"\n=== Label Column Guess (index {label_col_guess}) ===")
    print(df.iloc[:, label_col_guess].value_counts().head(10))

if __name__ == "__main__":
    main()
