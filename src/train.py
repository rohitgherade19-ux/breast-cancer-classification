"""
train.py

Trains Logistic Regression and Random Forest models on the Breast
Cancer dataset.

Run from the project root:
    python src/train.py
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from preprocessing import load_and_prepare

DATA_PATH = "data/data.csv"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def evaluate_model(name, model, X_test, y_test):
    """Print accuracy, classification report, confusion matrix, and
    ROC-AUC (if supported) for a fitted model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("ROC-AUC:", auc)

    return {"name": name, "accuracy": acc, "auc": auc}


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare(
        csv_path=DATA_PATH, scaler_path=SCALER_PATH
    )

    # --- Logistic Regression ---
    log_reg = train_logistic_regression(X_train, y_train)
    log_reg_result = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

    # --- Random Forest ---
    rf = train_random_forest(X_train, y_train)
    rf_result = evaluate_model("Random Forest", rf, X_test, y_test)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print("\nRandom Forest 5-fold CV scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())

    # --- Pick the best model by test accuracy, save it ---
    results = [log_reg_result, rf_result]
    models = {"Logistic Regression": log_reg, "Random Forest": rf}

    best = max(results, key=lambda r: r["accuracy"])
    best_model = models[best["name"]]

    print(f"\nBest model: {best['name']} (accuracy={best['accuracy']:.4f})")

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    print(f"Saved best model to {model_path}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
