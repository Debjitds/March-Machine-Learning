def compute_win_rate(games):

    wins = games.groupby("WTeamID").size()
    losses = games.groupby("LTeamID").size()

    total_games = wins.add(losses, fill_value=0)

    win_rate = wins / total_games

    return win_rate.fillna(0)