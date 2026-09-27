"""
preprocessing.py

Data loading, cleaning, splitting, and scaling for the
Breast Cancer Classification project.

This is the notebook's data-prep logic turned into reusable functions
so train.py and predict.py can both import and reuse it.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_COLUMNS_TO_DROP = ["id", "Unnamed: 32"]
TARGET_MAP = {"B": 0, "M": 1}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns and encode the target (B=0, M=1)."""
    df = df.copy()
    cols_to_drop = [c for c in RAW_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df["diagnosis"] = df["diagnosis"].map(TARGET_MAP)
    return df


def split_features_target(df: pd.DataFrame):
    """Split a cleaned DataFrame into X (features) and y (target)."""
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split (matches the notebook's split)."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(X_train, X_test, scaler_path: str = None):
    """
    Fit a StandardScaler on the training set and transform both sets.
    If scaler_path is given, the fitted scaler is saved to disk so
    predict.py can load the exact same scaling later.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled, scaler


def load_and_prepare(csv_path: str, scaler_path: str = None):
    """
    Full pipeline: load -> clean -> split features/target ->
    train/test split -> scale.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    df = load_data(csv_path)
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_path=scaler_path
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)


if __name__ == "__main__":
    # Quick sanity check when running this file directly:
    #   python src/preprocessing.py
    X_train_s, X_test_s, y_train, y_test, scaler, feature_names = load_and_prepare(
        csv_path="data/data.csv",
        scaler_path="models/scaler.pkl",
    )
    print("X_train_scaled:", X_train_s.shape)
    print("X_test_scaled :", X_test_s.shape)
    print("Number of features:", len(feature_names))
