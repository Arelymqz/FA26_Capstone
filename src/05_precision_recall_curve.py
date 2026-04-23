from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

train_path = DATA_DIR / "KDDTrain+.txt"
test_path = DATA_DIR / "KDDTest+.txt"

LABEL_COL = 41
DIFF_COL = 42

# This script trains a simple logistic regression model and generates a precision-recall curve to evaluate the trade-offs between recall and precision across different decision thresholds. The average precision score is also calculated and displayed in the plot title. The resulting plot is saved as a PNG file.
def main():

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    y_train = (train_df.iloc[:, LABEL_COL] != "normal").astype(int)
    y_test  = (test_df.iloc[:, LABEL_COL] != "normal").astype(int)

    X_train = train_df.drop(columns=[LABEL_COL, DIFF_COL])
    X_test  = test_df.drop(columns=[LABEL_COL, DIFF_COL])

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    scaler = StandardScaler(with_mean=False)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]

    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    ap = average_precision_score(y_test, probs)

    plt.figure()
    plt.plot(recall, precision)

    plt.xlabel("Recall (Attack Detection Rate)")
    plt.ylabel("Precision (Alert Accuracy)")
    plt.title(f"Precision-Recall Curve (AP = {ap:.3f})")

    plt.grid(True)

    save_path = OUT_DIR / "precision_recall_curve.png"

    plt.savefig(save_path, dpi=300)
    print("Saved:", save_path)

if __name__ == "__main__":
    main()