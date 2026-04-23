from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RATIO_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "realistic_ratios"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / "KDDTrain+.txt"

LABEL_COL = 41
DIFF_COL = 42

# Build a preprocessor that handles both categorical and numerical features
def build_preprocessor(X: pd.DataFrame):
    cat_cols = [1, 2, 3]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

# Evaluate a model on a given test set and return detailed metrics
def evaluate_model(model_name, pipe, X_test, y_test, scenario_name, threshold=0.30):
    probs = pipe.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    alert_rate = (tp + fp) / len(y_test)
    fp_per_10k = fp / len(y_test) * 10000

    # Return a dictionary with all relevant metrics for this scenario and model
    return {
        "scenario": scenario_name,
        "model": model_name,
        "threshold": threshold,
        "total_records": len(y_test),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision_malicious": precision,
        "recall_malicious": recall,
        "alert_rate": alert_rate,
        "fp_per_10k_records": fp_per_10k,
    }

# Main function to train models and evaluate on different realistic ratio scenarios
def main():
    train_df = pd.read_csv(train_path, header=None)

    y_train = (train_df.iloc[:, LABEL_COL] != "normal").astype(int)
    X_train = train_df.drop(columns=[LABEL_COL, DIFF_COL])

    pre = build_preprocessor(X_train)

    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }

    trained_pipes = {}

    print("Training models on original training set...")
    # We train on the original training set (which has the realistic ratio) and then evaluate on the various test sets with different attack ratios.
    for name, clf in models.items():
        pipe = Pipeline([
            ("pre", pre),
            ("model", clf),
        ])
        pipe.fit(X_train, y_train)
        trained_pipes[name] = pipe
        print(f"✅ Trained: {name}")

    # Define the paths to the test sets for each scenario.
    scenario_files = {
        "original": RATIO_DIR / "original_test.csv",
        "10pct_attack": RATIO_DIR / "10pct_attack_test.csv",
        "1pct_attack": RATIO_DIR / "1pct_attack_test.csv",
        "0_1pct_attack": RATIO_DIR / "0_1pct_attack_test.csv",
    }

    rows = []
    threshold = 0.30  # use your balanced setting for now

    # We evaluate each trained model on each scenario's test set and collect the results in a list of dictionaries, which we will convert to a DataFrame for easier analysis and saving.
    for scenario_name, path in scenario_files.items():
        df = pd.read_csv(path, header=None)

        y_test = (df.iloc[:, LABEL_COL] != "normal").astype(int)
        X_test = df.drop(columns=[LABEL_COL, DIFF_COL])

        # Evaluate each model on this scenario's test set and append the results to our list of rows.
        for model_name, pipe in trained_pipes.items():
            row = evaluate_model(
                model_name=model_name,
                pipe=pipe,
                X_test=X_test,
                y_test=y_test,
                scenario_name=scenario_name,
                threshold=threshold
            )
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = OUT_DIR / "realistic_ratio_model_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n✅ Saved: {results_path}")
    print("\n=== Results Preview ===")
    print(results_df)

if __name__ == "__main__":
    main()