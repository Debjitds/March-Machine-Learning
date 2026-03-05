import pandas as pd


def compute_win_rate(games):

    wins = games.groupby("WTeamID").size()
    losses = games.groupby("LTeamID").size()

    total = wins.add(losses, fill_value=0)

    win_rate = wins / total

    return win_rate.fillna(0).to_dict()


def compute_recent_win_rate(games, last_n=10):

    games = games.sort_values("DayNum")

    team_games = {}

    for _, row in games.iterrows():

        w = row["WTeamID"]
        l = row["LTeamID"]

        team_games.setdefault(w, []).append(1)
        team_games.setdefault(l, []).append(0)

    recent_wr = {}

    for team, results in team_games.items():

        last_games = results[-last_n:]
        recent_wr[team] = sum(last_games) / len(last_games)

    return recent_wr