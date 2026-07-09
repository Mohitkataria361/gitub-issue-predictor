import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features.csv"
)

_df = pd.read_csv(FEATURES_PATH)

repo_medians = (
    _df.groupby("repo")["resolution_hours"]
       .median()
       .to_dict()
)

overall_median = _df["resolution_hours"].median()


def get_repo_median(repo_name: str):
    """
    Returns repository median resolution time.

    If repository wasn't part of training,
    returns overall dataset median.
    """

    return repo_medians.get(
        repo_name,
        overall_median
    )