"""
Phase 4: Train and evaluate models predicting log(resolution_hours+1).

Usage:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

NUMERIC_FEATURES = [
    "title_len_chars",
    "title_len_words",
    "body_len",
    "num_labels",
    "created_hour_utc",
    "created_day_of_week",
    "comments",
    "repo_median_resolution_hours",
]
BOOLEAN_FEATURES = [
    "title_has_number",
    "title_has_question",
    "body_has_code_block",
    "is_owner_or_member",
    "is_bot_author",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["repo", "author_association"]
TARGET = "log_resolution_hours"


def load_data():
    path = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run collect_data.py then build_features.py first.")
    return pd.read_csv(path)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:20s} | MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = load_data()
    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    results = []

    median_pred = np.full_like(y_test, fill_value=y_train.median(), dtype=float)
    results.append(evaluate("Median baseline", y_test, median_pred))

    linear_pipe = Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(alpha=1.0))])
    linear_pipe.fit(X_train, y_train)
    results.append(evaluate("Ridge regression", y_test, linear_pipe.predict(X_test)))

    xgb_pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
            ),
        ]
    )
    xgb_pipe.fit(X_train, y_train)
    results.append(evaluate("XGBoost", y_test, xgb_pipe.predict(X_test)))

    print("\nIf XGBoost isn't meaningfully beating Ridge, that's a useful finding")
    print("to write up: it means the signal is mostly linear/additive, or you")
    print("need richer features (e.g. actual NLP on title/body instead of counts).")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_pipe, os.path.join(MODEL_DIR, "xgb_pipeline.joblib"))
    joblib.dump(linear_pipe, os.path.join(MODEL_DIR, "ridge_pipeline.joblib"))
    pd.DataFrame(results).to_csv(os.path.join(MODEL_DIR, "results.csv"), index=False)
    print(f"\nSaved models + results.csv to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
