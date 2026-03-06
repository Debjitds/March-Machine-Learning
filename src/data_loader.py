import os
import pandas as pd
from .config import DATA_PATH


# =========================
# MEN DATA
# =========================

def load_mens_games():
    """
    Load men's regular season game results.
    """
    path = os.path.join(DATA_PATH, "MRegularSeasonCompactResults.csv")

    df = pd.read_csv(path)

    print(f"Loaded MEN games: {len(df)} rows")

    return df


def load_mens_teams():
    """
    Load men's team metadata.
    """
    path = os.path.join(DATA_PATH, "MTeams.csv")

    df = pd.read_csv(path)

    print(f"Loaded MEN teams: {len(df)} teams")

    return df


# =========================
# WOMEN DATA
# =========================

def load_womens_games():
    """
    Load women's regular season game results.
    """
    path = os.path.join(DATA_PATH, "WRegularSeasonCompactResults.csv")

    df = pd.read_csv(path)

    print(f"Loaded WOMEN games: {len(df)} rows")

    return df


def load_womens_teams():
    """
    Load women's team metadata.
    """
    path = os.path.join(DATA_PATH, "WTeams.csv")

    df = pd.read_csv(path)

    print(f"Loaded WOMEN teams: {len(df)} teams")

    return df


# =========================
# SAMPLE SUBMISSION
# =========================

def load_sample_submission():
    """
    Automatically load the correct Kaggle submission template.

    Stage1  → ~519k rows (all matchups)
    Stage2  → ~132k rows (tournament matchups)

    This function detects which stage file exists.
    """

    stage1_path = os.path.join(DATA_PATH, "SampleSubmissionStage1.csv")
    stage2_path = os.path.join(DATA_PATH, "SampleSubmissionStage2.csv")

    if os.path.exists(stage2_path):
        path = stage2_path
        stage = "Stage2"

    elif os.path.exists(stage1_path):
        path = stage1_path
        stage = "Stage1"

    else:
        raise FileNotFoundError(
            "No submission template found. "
            "Expected SampleSubmissionStage1.csv or SampleSubmissionStage2.csv."
        )

    df = pd.read_csv(path)

    print("\nSubmission template loaded:")
    print(f"File: {path}")
    print(f"Stage: {stage}")
    print(f"Rows detected: {len(df)}")

    return df