import numpy as np
import pandas as pd


def compute_elo_ratings(games, k_factor=20, base_rating=1500):

    ratings = {}

    games = games.sort_values(["Season", "DayNum"])

    for _, row in games.iterrows():

        teamA = row["WTeamID"]
        teamB = row["LTeamID"]

        scoreA = row["WScore"]
        scoreB = row["LScore"]

        margin = scoreA - scoreB

        ratingA = ratings.get(teamA, base_rating)
        ratingB = ratings.get(teamB, base_rating)

        expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400))
        expectedB = 1 - expectedA

        # Margin of victory multiplier
        mov_multiplier = np.log(abs(margin) + 1)

        actualA = 1
        actualB = 0

        ratingA_new = ratingA + k_factor * mov_multiplier * (actualA - expectedA)
        ratingB_new = ratingB + k_factor * mov_multiplier * (actualB - expectedB)

        ratings[teamA] = ratingA_new
        ratings[teamB] = ratingB_new

    return ratings