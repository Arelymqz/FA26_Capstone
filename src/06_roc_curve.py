from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

train_path = DATA_DIR / "KDDTrain+.txt"
test_path = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41
DIFF_COL = 42


def main():

    print("Loading dataset...")

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Convert labels to binary
    y_train = (train_df.iloc[:, LABEL_COL] != "normal").astype(int)
    y_test = (test_df.iloc[:, LABEL_COL] != "normal").astype(int)

    # Remove label and difficulty columns
    X_train = train_df.drop(columns=[LABEL_COL, DIFF_COL])
    X_test = test_df.drop(columns=[LABEL_COL, DIFF_COL])

    print("Applying preprocessing...")

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align train and test columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Feature scaling
    scaler = StandardScaler(with_mean=False)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training baseline classifier...")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Get probabilities
    probs = model.predict_proba(X_test)[:, 1]

    print("Computing ROC metrics...")

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    print("AUC Score:", auc)

    # Plot ROC curve
    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # random classifier baseline

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Intrusion Detection Classifier")

    plt.legend()
    plt.grid(True)

    save_path = OUT_DIR / "roc_curve.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print("Saved ROC curve to:", save_path)


if __name__ == "__main__":
    main()