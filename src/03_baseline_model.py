from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH  = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41
DIFF_COL  = 42

def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None)

def split_X_y(df: pd.DataFrame):
    y = (df.iloc[:, LABEL_COL] != "normal").astype(int)  # 0 benign, 1 malicious
    X = df.drop(columns=[LABEL_COL, DIFF_COL])
    return X, y

def main():
    train_df = load_df(TRAIN_PATH)
    test_df  = load_df(TEST_PATH)

    X_train, y_train = split_X_y(train_df)
    X_test,  y_test  = split_X_y(test_df)

    # Identify categorical vs numeric columns
    cat_cols = [1, 2, 3]  # protocol_type, service, flag
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    # Baseline model (fast + strong baseline)
    clf = LogisticRegression(max_iter=1000, n_jobs=None)

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    print("🚀 Training baseline Logistic Regression model...")
    model.fit(X_train, y_train)

    print("✅ Evaluating on test set...")
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Confusion Matrix (test) ===")
    print(cm)

    print("\n=== Classification Report (test) ===")
    print(classification_report(y_test, y_pred, target_names=["benign", "malicious"], digits=4))

    # Save results
    cm_path = OUT_DIR / "baseline_confusion_matrix.txt"
    report_path = OUT_DIR / "baseline_classification_report.txt"

    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix (test)\n")
        f.write(str(cm) + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report (test)\n")
        f.write(classification_report(y_test, y_pred, target_names=["benign", "malicious"], digits=4))

    print(f"\n✅ Saved: {cm_path}")
    print(f"✅ Saved: {report_path}")

if __name__ == "__main__":
    main()
