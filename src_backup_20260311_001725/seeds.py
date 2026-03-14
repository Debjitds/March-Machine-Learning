import pandas as pd
from .config import DATA_PATH


def load_mens_seeds():
    """
    Load men's tournament seeds.
    """
    path = DATA_PATH + "MNCAATourneySeeds.csv"

    df = pd.read_csv(path)

    print(f"Loaded MEN seeds: {len(df)} rows")

    return df


def load_womens_seeds():
    """
    Load women's tournament seeds.
    """
    path = DATA_PATH + "WNCAATourneySeeds.csv"

    df = pd.read_csv(path)

    print(f"Loaded WOMEN seeds: {len(df)} rows")

    return df


def build_seed_dict(seed_df):
    """
    Convert seed dataframe to dictionary
    TeamID -> Seed number
    """

    seeds = {}

    for _, row in seed_df.iterrows():

        team = row["TeamID"]

        seed = row["Seed"]

        # Example seed format: W01, X16 etc
        seed_number = int(seed[1:3])

        seeds[team] = seed_number

    return seeds