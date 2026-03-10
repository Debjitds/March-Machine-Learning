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


# =====================================
# KenPom Style Adjusted Efficiency
# =====================================

def compute_adjusted_efficiency(detailed_games):

    team_stats = {}

    for _, g in detailed_games.iterrows():

        w = g["WTeamID"]
        l = g["LTeamID"]

        w_poss = g["WFGA"] - g["WOR"] + g["WTO"] + 0.475 * g["WFTA"]
        l_poss = g["LFGA"] - g["LOR"] + g["LTO"] + 0.475 * g["LFTA"]

        team_stats.setdefault(w, {"pts":0,"poss":0,"opp_pts":0,"opp_poss":0})
        team_stats.setdefault(l, {"pts":0,"poss":0,"opp_pts":0,"opp_poss":0})

        team_stats[w]["pts"] += g["WScore"]
        team_stats[w]["poss"] += w_poss
        team_stats[w]["opp_pts"] += g["LScore"]
        team_stats[w]["opp_poss"] += l_poss

        team_stats[l]["pts"] += g["LScore"]
        team_stats[l]["poss"] += l_poss
        team_stats[l]["opp_pts"] += g["WScore"]
        team_stats[l]["opp_poss"] += w_poss

    off_eff = {}
    def_eff = {}
    adj_margin = {}

    for team, s in team_stats.items():

        off = 100 * s["pts"] / s["poss"] if s["poss"] > 0 else 100
        deff = 100 * s["opp_pts"] / s["opp_poss"] if s["opp_poss"] > 0 else 100

        off_eff[team] = off
        def_eff[team] = deff
        adj_margin[team] = off - deff

    return off_eff, def_eff, adj_margin