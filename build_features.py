"""
Phase 3: Turn raw issue JSON into a clean feature table.

Usage:
    python build_features.py
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def load_all_raw():
    records = []
    files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if not files:
        raise FileNotFoundError(f"No raw data in {RAW_DIR}. Run collect_data.py first.")
    for path in files:
        with open(path) as f:
            records.extend(json.load(f))
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["repo", "number"], keep="last")
    return df


def title_has_number(title: str) -> bool:
    return bool(re.search(r"\d", title))


def title_has_question(title: str) -> bool:
    return "?" in title


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Cleaning decisions ---
    before = len(df)
    df = df.dropna(subset=["title", "created_at"])
    print(f"Dropped {before - len(df)} rows missing title/created_at")

    # Only closed issues have a resolution time -- drop anything still open
    # that slipped through (e.g. if you collected --state all)
    before = len(df)
    df = df.dropna(subset=["closed_at"])
    print(f"Dropped {before - len(df)} rows with no closed_at (still open / censored)")

    df["author_login"] = df["author_login"].fillna("[deleted_account]")
    df["author_association"] = df["author_association"].fillna("NONE")
    df["author_type"] = df["author_type"].fillna("User")
    df["is_bot_author"] = df["author_type"].eq("Bot")

    # --- Time features ---
    df["created_dt"] = pd.to_datetime(df["created_at"], utc=True)
    df["closed_dt"] = pd.to_datetime(df["closed_at"], utc=True)
    df["created_hour_utc"] = df["created_dt"].dt.hour
    df["created_day_of_week"] = df["created_dt"].dt.dayofweek
    df["is_weekend"] = df["created_day_of_week"].isin([5, 6])

    # --- Target: resolution time in hours, log-transformed ---
    resolution_hours = (df["closed_dt"] - df["created_dt"]).dt.total_seconds() / 3600
    # guard against any negative/zero values from bad timestamps
    df["resolution_hours"] = resolution_hours.clip(lower=0.01)
    df["log_resolution_hours"] = np.log1p(df["resolution_hours"])

    # --- Title/body features ---
    df["title_len_chars"] = df["title"].str.len()
    df["title_len_words"] = df["title"].str.split().str.len()
    df["title_has_number"] = df["title"].apply(title_has_number)
    df["title_has_question"] = df["title"].apply(title_has_question)

    # --- Repo-level baseline ---
    repo_median = df.groupby("repo")["resolution_hours"].transform("median")
    df["repo_median_resolution_hours"] = repo_median

    # --- Author association: is this a maintainer, contributor, or first-timer? ---
    df["is_owner_or_member"] = df["author_association"].isin(["OWNER", "MEMBER", "COLLABORATOR"])

    feature_cols = [
        "repo",
        "number",
        "title",
        "title_len_chars",
        "title_len_words",
        "title_has_number",
        "title_has_question",
        "body_len",
        "body_has_code_block",
        "num_labels",
        "author_association",
        "is_owner_or_member",
        "is_bot_author",
        "created_hour_utc",
        "created_day_of_week",
        "is_weekend",
        "comments",
        "repo_median_resolution_hours",
        "resolution_hours",
        "log_resolution_hours",
    ]
    return df[feature_cols]


if __name__ == "__main__":
    raw_df = load_all_raw()
    print(f"Loaded {len(raw_df)} raw issues across {raw_df['repo'].nunique()} repos")

    features_df = build_features(raw_df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    features_df.to_csv(out_path, index=False)
    print(f"Saved {len(features_df)} rows to {out_path}")
