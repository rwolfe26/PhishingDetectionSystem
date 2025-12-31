import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib

# ------------- CONFIG ----------------
CSV_PATH = "Phishing_Legitimate_full.csv"
LABEL_COL = "CLASS_LABEL"  
MODEL_PATH = "rf_phish_model.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# -------------------------------------


def load_data(csv_path: str, label_col: str):
    """Load CSV and split into features X and labels y."""
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop label + non-feature columns
    drop_cols = [label_col]
    if "id" in df.columns:
        drop_cols.append("id")

    X = df.drop(columns=drop_cols)
    y = df[label_col]

    # Map labels if needed (common formats: {-1,1} or {0,1})
    unique_vals = sorted(y.unique().tolist())
    print("Label values found:", unique_vals)

    if set(unique_vals) == {-1, 1}:
        # map to 0/1
        y = y.map({-1: 0, 1: 1})

    return X, y


def train_random_forest(X, y):
    """Train a simple Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"  
    )

    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC: {auc:.4f}")
    except Exception as e:
        print(f"\nCould not compute ROC AUC: {e}")

    return rf


def main():
    print(f"Loading data from {CSV_PATH} ...")
    X, y = load_data(CSV_PATH, LABEL_COL)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print("First few feature columns:", list(X.columns)[:10])

    print("\nTraining Random Forest model ...")
    model = train_random_forest(X, y)

    print(f"\nSaving model to {MODEL_PATH} ...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
