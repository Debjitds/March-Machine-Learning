import pandas as pd


def create_training_data(
    games,
    win_rate,
    elo,
    massey,
    recent,
    quality,
    adj_margin
):

    rows = []

    for _, g in games.iterrows():

        A = g["WTeamID"]
        B = g["LTeamID"]

        rows.append({

            "A_wr": win_rate.get(A,0.5),
            "B_wr": win_rate.get(B,0.5),

            "A_recent": recent.get(A,0.5),
            "B_recent": recent.get(B,0.5),
            "recent_diff": recent.get(A,0.5) - recent.get(B,0.5),

            "A_elo": elo.get(A,1500),
            "B_elo": elo.get(B,1500),
            "elo_diff": elo.get(A,1500) - elo.get(B,1500),

            "A_massey": massey.get(A,100),
            "B_massey": massey.get(B,100),
            "massey_diff": massey.get(A,100) - massey.get(B,100),

            "A_quality": quality.get(A,0),
            "B_quality": quality.get(B,0),
            "quality_diff": quality.get(A,0) - quality.get(B,0),

            "A_adj": adj_margin.get(A,0),
            "B_adj": adj_margin.get(B,0),
            "adj_diff": adj_margin.get(A,0) - adj_margin.get(B,0),

            "target":1
        })

        rows.append({

            "A_wr": win_rate.get(B,0.5),
            "B_wr": win_rate.get(A,0.5),

            "A_recent": recent.get(B,0.5),
            "B_recent": recent.get(A,0.5),
            "recent_diff": recent.get(B,0.5) - recent.get(A,0.5),

            "A_elo": elo.get(B,1500),
            "B_elo": elo.get(A,1500),
            "elo_diff": elo.get(B,1500) - elo.get(A,1500),

            "A_massey": massey.get(B,100),
            "B_massey": massey.get(A,100),
            "massey_diff": massey.get(B,100) - massey.get(A,100),

            "A_quality": quality.get(B,0),
            "B_quality": quality.get(A,0),
            "quality_diff": quality.get(B,0) - quality.get(A,0),

            "A_adj": adj_margin.get(B,0),
            "B_adj": adj_margin.get(A,0),
            "adj_diff": adj_margin.get(B,0) - adj_margin.get(A,0),

            "target":0
        })

    return pd.DataFrame(rows)