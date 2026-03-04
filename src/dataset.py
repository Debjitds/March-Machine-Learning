import pandas as pd


def compute_team_avg_margin(games):
    """
    Compute average score margin for each team.
    Positive margin = strong team.
    """

    win_margin = games.groupby("WTeamID").apply(
        lambda x: (x["WScore"] - x["LScore"]).mean()
    )

    lose_margin = games.groupby("LTeamID").apply(
        lambda x: (x["LScore"] - x["WScore"]).mean()
    )

    total_margin = win_margin.add(lose_margin, fill_value=0)

    return total_margin.to_dict()


def create_training_data(games, win_rate, elo_ratings):

    rows = []

    # Compute margin feature once
    avg_margin = compute_team_avg_margin(games)

    for _, g in games.iterrows():

        A = g["WTeamID"]
        B = g["LTeamID"]

        A_wr = win_rate.get(A, 0)
        B_wr = win_rate.get(B, 0)

        A_elo = elo_ratings.get(A, 1500)
        B_elo = elo_ratings.get(B, 1500)

        A_margin = avg_margin.get(A, 0)
        B_margin = avg_margin.get(B, 0)

        # Forward direction (A beats B)
        rows.append({
            "A_wr": A_wr,
            "B_wr": B_wr,
            "A_elo": A_elo,
            "B_elo": B_elo,
            "elo_diff": A_elo - B_elo,
            "A_avg_margin": A_margin,
            "B_avg_margin": B_margin,
            "margin_diff": A_margin - B_margin,
            "target": 1
        })

        # Reverse direction (B loses to A)
        rows.append({
            "A_wr": B_wr,
            "B_wr": A_wr,
            "A_elo": B_elo,
            "B_elo": A_elo,
            "elo_diff": B_elo - A_elo,
            "A_avg_margin": B_margin,
            "B_avg_margin": A_margin,
            "margin_diff": B_margin - A_margin,
            "target": 0
        })

    return pd.DataFrame(rows)