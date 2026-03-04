import pandas as pd
from .config import DATA_PATH

# -------- MEN --------
def load_mens_games():
    return pd.read_csv(DATA_PATH + "MRegularSeasonCompactResults.csv")

def load_mens_teams():
    return pd.read_csv(DATA_PATH + "MTeams.csv")


# -------- WOMEN --------
def load_womens_games():
    return pd.read_csv(DATA_PATH + "WRegularSeasonCompactResults.csv")

def load_womens_teams():
    return pd.read_csv(DATA_PATH + "WTeams.csv")


# -------- SAMPLE --------
def load_sample_submission():
    return pd.read_csv(DATA_PATH + "SampleSubmissionStage1.csv")