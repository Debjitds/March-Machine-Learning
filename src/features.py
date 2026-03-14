import pandas as pd
import numpy as np
from collections import defaultdict


def compute_win_rate(games):
    """
    Returns dict team_id -> season-agnostic win rate (wins / (wins+losses)).
    If a team has no games, default to 0.5.
    """
    wins = games.groupby("WTeamID").size()
    losses = games.groupby("LTeamID").size()

    win_rate = {}

    all_teams = set(wins.index).union(set(losses.index))

    for t in all_teams:
        w = int(wins.get(t, 0))
        l = int(losses.get(t, 0))
        total = w + l
        win_rate[t] = 0.5 if total == 0 else w / total

    return win_rate


def compute_recent_form(games, last_n=10):
    """
    Per-team recent win fraction over the last_n matches (chronological by DayNum).
    Returns dict team_id -> recent_form (0..1). If no games -> 0.5.
    """
    if "DayNum" in games.columns:
        games_sorted = games.sort_values("DayNum")
    else:
        games_sorted = games.copy()

    team_results = defaultdict(list)

    for _, r in games_sorted.iterrows():
        w = int(r["WTeamID"])
        l = int(r["LTeamID"])
        team_results[w].append(1)
        team_results[l].append(0)

    recent_form = {}
    for team, results in team_results.items():
        recent = results[-last_n:]
        if len(recent) == 0:
            recent_form[team] = 0.5
        else:
            recent_form[team] = float(sum(recent)) / len(recent)

    return recent_form


def compute_tempo_efficiency(detailed_games):
    """
    From detailed results (per-game possessions / box-score fields) compute:
    - off_eff: points per 100 possessions estimate for each team
    - def_eff: opponent points per 100 possessions
    - tempo: average possessions per game estimate
    - net: off - def
    Returns 4 dicts: off_eff, def_eff, tempo, net
    Expects fields like WFGA, WOR, WTO, WFTA, WScore and LFGA, etc.
    """
    rows = []

    # Iterate once and build per-team list of metrics
    for _, g in detailed_games.iterrows():
        W = int(g["WTeamID"])
        L = int(g["LTeamID"])

        W_pts = float(g.get("WScore", 0.0))
        L_pts = float(g.get("LScore", 0.0))

        # possessions estimate (basic)
        W_poss = (
            float(g.get("WFGA", 0)) - float(g.get("WOR", 0)) + float(g.get("WTO", 0)) + 0.44 * float(g.get("WFTA", 0))
        )

        L_poss = (
            float(g.get("LFGA", 0)) - float(g.get("LOR", 0)) + float(g.get("LTO", 0)) + 0.44 * float(g.get("LFTA", 0))
        )

        # guard against zero or tiny possessions
        if W_poss <= 0:
            W_poss = 1.0
        if L_poss <= 0:
            L_poss = 1.0

        W_off = (W_pts / W_poss) * 100.0
        W_def = (L_pts / L_poss) * 100.0

        L_off = (L_pts / L_poss) * 100.0
        L_def = (W_pts / W_poss) * 100.0

        rows.append({"TeamID": W, "OffEff": W_off, "DefEff": W_def, "Tempo": W_poss})
        rows.append({"TeamID": L, "OffEff": L_off, "DefEff": L_def, "Tempo": L_poss})

    df = pd.DataFrame(rows)
    # aggregate by mean (robust)
    team_stats = df.groupby("TeamID").mean()

    off = team_stats["OffEff"].to_dict()
    deff = team_stats["DefEff"].to_dict()
    tempo = team_stats["Tempo"].to_dict()

    net = {t: off.get(t, 0.0) - deff.get(t, 0.0) for t in off.keys()}

    return off, deff, tempo, net


def compute_adjusted_efficiency(off_eff, def_eff, games, min_games=5):
    """
    Compute an opponent-adjusted offensive and defensive efficiency per team.
    For each game, append (team_off / opponent_def) to lists; final value is mean of those ratios.
    If a team has fewer than `min_games`, blend with league average using weight n/(n+min_games).
    Returns: final_adj_off (dict), final_adj_def (dict), adj_net (dict).
    """
    # store per-team ratios
    adj_off_lists = defaultdict(list)
    adj_def_lists = defaultdict(list)

    # Precompute defaults to avoid KeyErrors
    # Use global league mean as fallback
    off_values = np.array(list(off_eff.values())) if len(off_eff) > 0 else np.array([100.0])
    def_values = np.array(list(def_eff.values())) if len(def_eff) > 0 else np.array([100.0])

    league_off_avg = float(np.mean(off_values))
    league_def_avg = float(np.mean(def_values))

    for _, g in games.iterrows():
        W = int(g["WTeamID"])
        L = int(g["LTeamID"])

        W_off = float(off_eff.get(W, league_off_avg))
        L_off = float(off_eff.get(L, league_off_avg))

        W_def = float(def_eff.get(W, league_def_avg))
        L_def = float(def_eff.get(L, league_def_avg))

        # compute ratios (off / opp_def) with guard against zero
        adj_off_lists[W].append(W_off / max(L_def, 1e-6))
        adj_off_lists[L].append(L_off / max(W_def, 1e-6))

        adj_def_lists[W].append(W_def / max(L_off, 1e-6))
        adj_def_lists[L].append(L_def / max(W_off, 1e-6))

    final_adj_off = {}
    final_adj_def = {}
    adj_net = {}

    # compute league-level means of ratios for fallback
    all_off_ratios = []
    all_def_ratios = []
    for t, vals in adj_off_lists.items():
        all_off_ratios.extend(vals)
    for t, vals in adj_def_lists.items():
        all_def_ratios.extend(vals)

    league_off_ratio_mean = float(np.mean(all_off_ratios)) if len(all_off_ratios) > 0 else 1.0
    league_def_ratio_mean = float(np.mean(all_def_ratios)) if len(all_def_ratios) > 0 else 1.0

    # finalize per-team with smoothing/blending
    for team in set(list(adj_off_lists.keys()) + list(adj_def_lists.keys())):
        off_list = adj_off_lists.get(team, [])
        def_list = adj_def_lists.get(team, [])

        n_off = len(off_list)
        n_def = len(def_list)

        # weights (simple) — more games => trust team mean
        w_off = n_off / (n_off + min_games) if (n_off + min_games) > 0 else 0.0
        w_def = n_def / (n_def + min_games) if (n_def + min_games) > 0 else 0.0

        team_off_mean = float(np.mean(off_list)) if n_off > 0 else league_off_ratio_mean
        team_def_mean = float(np.mean(def_list)) if n_def > 0 else league_def_ratio_mean

        final_off = w_off * team_off_mean + (1 - w_off) * league_off_ratio_mean
        final_def = w_def * team_def_mean + (1 - w_def) * league_def_ratio_mean

        final_adj_off[team] = final_off
        final_adj_def[team] = final_def
        adj_net[team] = final_off - final_def

    return final_adj_off, final_adj_def, adj_net