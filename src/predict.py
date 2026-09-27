"""
predict.py

Loads the trained model, scaler, and feature list saved by train.py,
and runs predictions on new, unseen input. This is the piece an API
or UI will call.

Usage (as a library):
    from predict import predict

    result = predict({
        "radius_mean": 17.99,
        "texture_mean": 10.38,
        ...  # all 30 feature columns
    })
    print(result)  # {"diagnosis": "Malignant", "label": 1, "probability_malignant": 0.94}

Usage (from the command line, with a CSV of one or more rows):
    python src/predict.py --csv data/sample_input.csv
"""

import argparse
import os
import joblib
import pandas as pd

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

LABEL_MAP = {0: "Benign", 1: "Malignant"}

_model = None
_scaler = None
_feature_names = None


def _load_artifacts():
    """Load model, scaler, and feature names once and cache them."""
    global _model, _scaler, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _feature_names is None:
        _feature_names = joblib.load(FEATURE_NAMES_PATH)
    return _model, _scaler, _feature_names


def _prepare_input(input_data) -> pd.DataFrame:
    """
    Accepts a single dict of feature_name -> value, or a DataFrame of
    one or more rows, and returns a DataFrame with columns in the
    exact order the model was trained on.
    """
    _, _, feature_names = _load_artifacts()

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a dict or a pandas DataFrame")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Reorder/select columns to match training order exactly
    df = df[feature_names]
    return df


def predict(input_data):
    """
    Run a prediction on a single sample (dict) or multiple samples
    (DataFrame). Returns a single result dict for one row, or a list
    of result dicts for multiple rows.
    """
    model, scaler, _ = _load_artifacts()

    df = _prepare_input(input_data)
    X_scaled = scaler.transform(df)

    labels = model.predict(X_scaled)
    results = []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]  # P(malignant)
    else:
        probs = [None] * len(labels)

    for label, prob in zip(labels, probs):
        results.append({
            "diagnosis": LABEL_MAP[int(label)],
            "label": int(label),
            "probability_malignant": float(prob) if prob is not None else None,
        })

    return results[0] if len(results) == 1 else results


def predict_from_csv(csv_path: str):
    """Run predictions for every row in a CSV file (no 'diagnosis' column needed)."""
    df = pd.read_csv(csv_path)
    return predict(df)


def _cli():
    parser = argparse.ArgumentParser(description="Run breast cancer predictions.")
    parser.add_argument("--csv", type=str, required=True,
                         help="Path to a CSV file containing feature columns.")
    args = parser.parse_args()

    results = predict_from_csv(args.csv)
    if isinstance(results, dict):
        results = [results]

    for i, r in enumerate(results):
        print(f"Row {i}: {r['diagnosis']} (P(malignant)={r['probability_malignant']})")


if __name__ == "__main__":
    _cli()
