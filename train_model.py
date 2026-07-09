
"""
Phase 3.2 - Train and compare ML models.

Usage:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGET = "log_resolution_hours"

NUMERIC = [
    "title_len_chars",
    "title_len_words",
    "body_len",
    "body_word_count",
    "num_labels",
    "comments",
    "reactions",
    "num_urls",
    "num_mentions",
    "num_code_blocks",
    "repo_stars",
    "repo_forks",
    "repo_watchers",
    "repo_open_issues",
    "repo_size",
    "repo_age_days",
    "created_hour_utc",
    "created_day_of_week",
    "created_month",
]

BOOLEAN = [
    "title_has_number",
    "title_has_question",
    "body_has_code_block",
    "is_bot_author",
    "is_owner_or_member",
    "is_weekend",
    "is_business_hours",
    "repo_archived",
    "has_bug_label",
    "has_enhancement_label",
    "has_documentation_label",
    "has_help_wanted",
    "has_good_first_issue",
    "has_milestone",
]

CATEGORICAL = [
    "repo_language",
    "author_association",
    "default_branch",
]


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "features.csv not found. Run collect_data.py and build_features.py first."
        )
    return pd.read_csv(DATA_PATH)


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("bool", "passthrough", BOOLEAN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return {
        "Model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def main():
    df = load_data()

    features = NUMERIC + BOOLEAN + CATEGORICAL

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    results = []
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipe = Pipeline([
            ("prep", preprocessor()),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)

        trained[name] = pipe

        results.append(
            evaluate(name, pipe, X_test, y_test)
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    print("\n================ Results ================\n")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = trained[best_name]

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_model.joblib")
    )

    results_df.to_csv(
        os.path.join(MODEL_DIR, "model_results.csv"),
        index=False,
    )

    print(f"\nBest Model: {best_name}")
    print("Saved:")
    print(" - models/best_model.joblib")
    print(" - models/model_results.csv")


if __name__ == "__main__":
    main()
