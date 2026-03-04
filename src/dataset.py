import pandas as pd


def compute_team_avg_margin(games):

    win_margin = games.groupby("WTeamID").apply(
        lambda x: (x["WScore"] - x["LScore"]).mean()
    )

    lose_margin = games.groupby("LTeamID").apply(
        lambda x: (x["LScore"] - x["WScore"]).mean()
    )

    total_margin = win_margin.add(lose_margin, fill_value=0)

    return total_margin.to_dict()


def create_training_data(games, win_rate, elo_ratings, massey_ratings=None):

    rows = []

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

        if massey_ratings:
            A_massey = massey_ratings.get(A, 100)
            B_massey = massey_ratings.get(B, 100)
        else:
            A_massey = 100
            B_massey = 100

        # Forward direction
        rows.append({
            "A_wr": A_wr,
            "B_wr": B_wr,

            "A_elo": A_elo,
            "B_elo": B_elo,
            "elo_diff": A_elo - B_elo,

            "A_avg_margin": A_margin,
            "B_avg_margin": B_margin,
            "margin_diff": A_margin - B_margin,

            "A_massey": A_massey,
            "B_massey": B_massey,
            "massey_diff": A_massey - B_massey,

            "target": 1
        })

        # Reverse direction
        rows.append({
            "A_wr": B_wr,
            "B_wr": A_wr,

            "A_elo": B_elo,
            "B_elo": A_elo,
            "elo_diff": B_elo - A_elo,

            "A_avg_margin": B_margin,
            "B_avg_margin": A_margin,
            "margin_diff": B_margin - A_margin,

            "A_massey": B_massey,
            "B_massey": A_massey,
            "massey_diff": B_massey - A_massey,

            "target": 0
        })

    return pd.DataFrame(rows)