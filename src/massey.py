import pandas as pd


def load_massey_ratings(path="data/MMasseyOrdinals.csv"):
    """
    Load Massey Ordinal rankings and compute
    an average rating for each team.
    """

    df = pd.read_csv(path)

    # Use latest day rankings
    df = df.sort_values("RankingDayNum")

    # Keep most recent ranking for each team/system
    df = df.groupby(["Season", "TeamID", "SystemName"]).tail(1)

    # Average rankings across systems
    ratings = df.groupby("TeamID")["OrdinalRank"].mean()

    return ratings.to_dict()