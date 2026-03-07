import pandas as pd
from .config import DATA_PATH


def load_massey_ordinals(league_name):
    """
    Load Massey Ordinal rankings.

    Only MEN league has Massey rankings.
    Women league will return empty dict safely.
    """

    if league_name == "WOMEN":
        print("No Massey rankings for WOMEN league.")
        return {}

    path = DATA_PATH + "MMasseyOrdinals.csv"

    df = pd.read_csv(path)

    print(f"Loaded Massey Ordinals: {len(df)} rows")

    return compute_massey_rankings(df)


def compute_massey_rankings(df):
    """
    Convert Massey rankings into dictionary:
    TeamID -> average ranking
    """

    massey = {}

    grouped = df.groupby("TeamID")["OrdinalRank"].mean()

    for team, rank in grouped.items():
        massey[team] = rank

    return massey