import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


def predict_resolution_time(features_df):
    """
    Predict issue resolution time.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Single-row dataframe.

    Returns
    -------
    dict
    """

    pred_log = float(model.predict(features_df)[0])

    pred_log = max(pred_log, 0)

    hours = float(np.expm1(pred_log))

    return {
        "hours": round(hours, 2),
        "days": round(hours / 24, 2),
    }