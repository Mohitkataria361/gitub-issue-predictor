"""
Phase 5: Streamlit demo.

Run with:
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "features.csv")

st.set_page_config(page_title="GitHub Issue Resolution Predictor", page_icon="🐙")
st.title("🐙 GitHub Issue Resolution Time Predictor")
st.caption(
    "Predicts expected time-to-close for a hypothetical issue, based on title, "
    "labels, author type, and repo -- trained on real closed issues collected "
    "via the GitHub REST API."
)

model_path = os.path.join(MODEL_DIR, "xgb_pipeline.joblib")
if not os.path.exists(model_path):
    st.error("No trained model found. Run collect_data.py -> build_features.py -> train_model.py first.")
    st.stop()

model = joblib.load(model_path)
df = pd.read_csv(FEATURES_PATH)
repos = sorted(df["repo"].unique())
repo_medians = df.groupby("repo")["resolution_hours"].median().to_dict()
associations = sorted(df["author_association"].unique())

with st.form("predict_form"):
    repo = st.selectbox("Repository", repos)
    title = st.text_input("Issue title", "Bug: crash when parsing empty input")
    body_len = st.slider("Body length (characters)", 0, 3000, 200)
    body_has_code_block = st.checkbox("Includes a code block?", value=True)
    num_labels = st.slider("Number of labels", 0, 10, 1)
    author_association = st.selectbox("Author association", associations)
    is_bot_author = st.checkbox("Opened by a bot?", value=False)
    created_hour_utc = st.slider("Hour opened (UTC)", 0, 23, 14)
    created_day_of_week = st.selectbox(
        "Day of week",
        options=list(range(7)),
        format_func=lambda i: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i],
    )
    submitted = st.form_submit_button("Predict resolution time")

if submitted:
    row = pd.DataFrame(
        [
            {
                "title_len_chars": len(title),
                "title_len_words": len(title.split()),
                "title_has_number": any(c.isdigit() for c in title),
                "title_has_question": "?" in title,
                "body_len": body_len,
                "body_has_code_block": body_has_code_block,
                "num_labels": num_labels,
                "created_hour_utc": created_hour_utc,
                "created_day_of_week": created_day_of_week,
                "comments": 0,
                "repo_median_resolution_hours": repo_medians.get(
                    repo, df["resolution_hours"].median()
                ),
                "is_owner_or_member": author_association in ["OWNER", "MEMBER", "COLLABORATOR"],
                "is_bot_author": is_bot_author,
                "is_weekend": created_day_of_week in [5, 6],
                "repo": repo,
                "author_association": author_association,
            }
        ]
    )

    pred_log_hours = model.predict(row)[0]
    pred_hours = np.expm1(pred_log_hours)

    if pred_hours < 24:
        display = f"{pred_hours:.1f} hours"
    else:
        display = f"{pred_hours / 24:.1f} days"

    st.metric("Predicted time to close", display)
    st.caption(
        f"({repo} median resolution time in training data = "
        f"{repo_medians.get(repo, float('nan')):.1f} hours)"
    )

    st.info(
        "This only predicts resolution time GIVEN the issue eventually closes -- "
        "it says nothing about whether an issue will close at all. Treat as "
        "directional, not precise."
    )
