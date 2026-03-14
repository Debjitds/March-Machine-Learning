import pandas as pd


def compute_win_rate(games):

    wins = games.groupby("WTeamID").size()
    losses = games.groupby("LTeamID").size()

    win_rate = {}

    teams = set(wins.index).union(set(losses.index))

    for team in teams:

        w = wins.get(team, 0)
        l = losses.get(team, 0)

        total = w + l

        if total == 0:
            win_rate[team] = 0.5
        else:
            win_rate[team] = w / total

    return win_rate


def compute_recent_form(games, last_n=10):

    games = games.sort_values("DayNum")

    team_games = {}

    for _, row in games.iterrows():

        w = row["WTeamID"]
        l = row["LTeamID"]

        team_games.setdefault(w, []).append(1)
        team_games.setdefault(l, []).append(0)

    recent_form = {}

    for team, results in team_games.items():

        recent = results[-last_n:]

        recent_form[team] = sum(recent) / len(recent)

    return recent_form