import numpy as np
import pandas as pd

def compute_elo_ratings(games, k_factor=20, base_rating=1500):
    """
    Dynamic Elo update across seasons/days. Returns dict team_id -> rating.
    Keeps ordering by (Season, DayNum) if those columns exist.
    Uses a margin multiplier (log) so big wins move rating more.
    """
    ratings = {}

    # sort if possible so Elo updates chronologically
    if {"Season", "DayNum"}.issubset(set(games.columns)):
        games = games.sort_values(["Season", "DayNum"])
    elif "DayNum" in games.columns:
        games = games.sort_values("DayNum")

    for _, row in games.iterrows():
        teamA = int(row["WTeamID"])
        teamB = int(row["LTeamID"])

        scoreA = float(row.get("WScore", 1))
        scoreB = float(row.get("LScore", 0))

        margin = scoreA - scoreB

        ratingA = ratings.get(teamA, base_rating)
        ratingB = ratings.get(teamB, base_rating)

        expectedA = 1.0 / (1.0 + 10.0 ** ((ratingB - ratingA) / 400.0))
        expectedB = 1.0 - expectedA

        # margin multiplier: log margin+1 (keeps stability)
        mov_multiplier = np.log(abs(margin) + 1.0)

        # keep multiplier >= 1e-6 to avoid zero updates
        mov_multiplier = max(mov_multiplier, 1e-6)

        actualA = 1.0
        actualB = 0.0

        ratingA_new = ratingA + k_factor * mov_multiplier * (actualA - expectedA)
        ratingB_new = ratingB + k_factor * mov_multiplier * (actualB - expectedB)

        ratings[teamA] = ratingA_new
        ratings[teamB] = ratingB_new

    return ratings


def compute_elo(games, k_factor=20, base_rating=1500):
    """
    Compatibility wrapper. main.py expects compute_elo(...).
    Calls compute_elo_ratings under the hood.
    """
    return compute_elo_ratings(games, k_factor=k_factor, base_rating=base_rating)