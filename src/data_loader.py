import pandas as pd
from .config import DATA_PATH


# =========================
# MEN DATA
# =========================

def load_mens_games():
    """
    Load men's regular season game results.
    """
    path = DATA_PATH + "MRegularSeasonCompactResults.csv"

    df = pd.read_csv(path)

    print(f"Loaded MEN games: {len(df)} rows")

    return df


def load_mens_teams():
    """
    Load men's team metadata.
    """
    path = DATA_PATH + "MTeams.csv"

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
    path = DATA_PATH + "WRegularSeasonCompactResults.csv"

    df = pd.read_csv(path)

    print(f"Loaded WOMEN games: {len(df)} rows")

    return df


def load_womens_teams():
    """
    Load women's team metadata.
    """
    path = DATA_PATH + "WTeams.csv"

    df = pd.read_csv(path)

    print(f"Loaded WOMEN teams: {len(df)} teams")

    return df


# =========================
# SAMPLE SUBMISSION
# =========================

def load_sample_submission():
    """
    Load the correct Kaggle submission template.

    Stage1 template contains ~519k rows.
    This function prevents accidental loading
    of Stage2 or incorrect files.
    """

    path = DATA_PATH + "SampleSubmissionStage1.csv"

    df = pd.read_csv(path)

    print("\nSubmission template loaded:")
    print(path)
    print(f"Rows detected: {len(df)}")

    # Safety validation
    if len(df) < 500000:
        raise ValueError(
            "Incorrect submission template detected.\n"
            "Expected SampleSubmissionStage1.csv with ~519k rows."
        )

    return df