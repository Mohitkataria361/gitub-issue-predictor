
"""
Phase 3.1: Build ML-ready features from raw collected GitHub issues.

Usage:
    python build_features.py
"""

import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def latest_raw_file():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        raise FileNotFoundError("No raw dataset found. Run collect_data.py first.")
    return files[-1]


def load_data():
    with open(latest_raw_file(), "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)
    df["repo_created_at"] = pd.to_datetime(df["repo_created_at"], utc=True)

    # Target
    df["resolution_hours"] = (
        (df["closed_at"] - df["created_at"]).dt.total_seconds() / 3600
    )

    df = df[df["resolution_hours"] >= 0].copy()

    df["log_resolution_hours"] = np.log1p(df["resolution_hours"])

    # Title
    df["title_len_chars"] = df["title"].fillna("").str.len()
    df["title_len_words"] = df["title"].fillna("").str.split().str.len()
    df["title_has_number"] = df["title"].fillna("").str.contains(r"\d", regex=True)
    df["title_has_question"] = df["title"].fillna("").str.contains(r"\?")

    # Time
    df["created_hour_utc"] = df["created_at"].dt.hour
    df["created_day_of_week"] = df["created_at"].dt.dayofweek
    df["created_month"] = df["created_at"].dt.month
    df["is_weekend"] = df["created_day_of_week"] >= 5
    df["is_business_hours"] = (
        (df["created_hour_utc"] >= 9) &
        (df["created_hour_utc"] <= 18)
    )

    # Repository
    now = pd.Timestamp.now(tz="UTC")
    df["repo_age_days"] = (
        now - df["repo_created_at"]
    ).dt.days

    # Author
    df["is_bot_author"] = (
        df["author_type"].fillna("").str.lower() == "bot"
    )

    df["is_owner_or_member"] = df["author_association"].fillna("").isin(
        ["OWNER", "MEMBER", "COLLABORATOR"]
    )

    # Fill missing categorical values
    df["repo_language"] = df["repo_language"].fillna("Unknown")
    df["author_association"] = df["author_association"].fillna("NONE")

    return df


def main():
    df = load_data()

    features = build_features(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    output = os.path.join(PROCESSED_DIR, "features.csv")

    features.to_csv(output, index=False)

    print("=" * 60)
    print("Feature Engineering Complete")
    print("=" * 60)
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns)}")
    print(f"Saved to:\n{output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
