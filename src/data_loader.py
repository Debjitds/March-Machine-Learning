import pandas as pd
from .config import DATA_PATH


# =========================
# MEN DATA
# =========================

def load_mens_games():

    path = DATA_PATH + "MRegularSeasonCompactResults.csv"

    df = pd.read_csv(path)

    print(f"Loaded MEN games: {len(df)} rows")

    return df


def load_mens_teams():

    path = DATA_PATH + "MTeams.csv"

    df = pd.read_csv(path)

    print(f"Loaded MEN teams: {len(df)} teams")

    return df


# =========================
# WOMEN DATA
# =========================

def load_womens_games():

    path = DATA_PATH + "WRegularSeasonCompactResults.csv"

    df = pd.read_csv(path)

    print(f"Loaded WOMEN games: {len(df)} rows")

    return df


def load_womens_teams():

    path = DATA_PATH + "WTeams.csv"

    df = pd.read_csv(path)

    print(f"Loaded WOMEN teams: {len(df)} teams")

    return df


# =========================
# SEED DATA
# =========================

def load_mens_seeds():

    path = DATA_PATH + "MNCAATourneySeeds.csv"

    df = pd.read_csv(path)

    print(f"Loaded MEN seeds: {len(df)} rows")

    return df


def load_womens_seeds():

    path = DATA_PATH + "WNCAATourneySeeds.csv"

    df = pd.read_csv(path)

    print(f"Loaded WOMEN seeds: {len(df)} rows")

    return df


# =========================
# SAMPLE SUBMISSION
# =========================

def load_sample_submission():

    path = DATA_PATH + "SampleSubmissionStage2.csv"

    df = pd.read_csv(path)

    print("\nSubmission template loaded:")
    print(path)
    print(f"Rows detected: {len(df)}")

    if len(df) != 132133:
        raise ValueError("Expected Stage2 submission file (132133 rows)")

    return df