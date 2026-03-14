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


def safe_ratio(a, b):

    if b == 0:
        return 0

    return a / b


def create_training_data(
    games,
    win_rate,
    elo_ratings,
    massey_ratings=None,
    recent_wr=None,
    team_quality=None
):

    rows = []

    avg_margin = compute_team_avg_margin(games)

    for _, g in games.iterrows():

        A = g["WTeamID"]
        B = g["LTeamID"]

        A_wr = win_rate.get(A, 0)
        B_wr = win_rate.get(B, 0)

        A_recent = recent_wr.get(A, 0.5)
        B_recent = recent_wr.get(B, 0.5)

        A_elo = elo_ratings.get(A, 1500)
        B_elo = elo_ratings.get(B, 1500)

        A_margin = avg_margin.get(A, 0)
        B_margin = avg_margin.get(B, 0)

        A_massey = massey_ratings.get(A, 100) if massey_ratings else 100
        B_massey = massey_ratings.get(B, 100) if massey_ratings else 100

        A_q = team_quality.get(A, 0)
        B_q = team_quality.get(B, 0)

        rows.append({

            "A_wr": A_wr,
            "B_wr": B_wr,

            "A_recent_wr": A_recent,
            "B_recent_wr": B_recent,
            "recent_wr_diff": A_recent - B_recent,

            "A_elo": A_elo,
            "B_elo": B_elo,
            "elo_diff": A_elo - B_elo,

            "A_avg_margin": A_margin,
            "B_avg_margin": B_margin,
            "margin_diff": A_margin - B_margin,

            "A_massey": A_massey,
            "B_massey": B_massey,
            "massey_diff": A_massey - B_massey,

            "A_quality": A_q,
            "B_quality": B_q,
            "quality_diff": A_q - B_q,

            "elo_ratio": safe_ratio(A_elo, B_elo),
            "margin_ratio": safe_ratio(A_margin, B_margin),
            "quality_ratio": safe_ratio(A_q, B_q),
            "massey_ratio": safe_ratio(A_massey, B_massey),
            "recent_ratio": safe_ratio(A_recent, B_recent),

            "target": 1
        })

        rows.append({

            "A_wr": B_wr,
            "B_wr": A_wr,

            "A_recent_wr": B_recent,
            "B_recent_wr": A_recent,
            "recent_wr_diff": B_recent - A_recent,

            "A_elo": B_elo,
            "B_elo": A_elo,
            "elo_diff": B_elo - A_elo,

            "A_avg_margin": B_margin,
            "B_avg_margin": A_margin,
            "margin_diff": B_margin - A_margin,

            "A_massey": B_massey,
            "B_massey": A_massey,
            "massey_diff": B_massey - A_massey,

            "A_quality": B_q,
            "B_quality": A_q,
            "quality_diff": B_q - A_q,

            "elo_ratio": safe_ratio(B_elo, A_elo),
            "margin_ratio": safe_ratio(B_margin, A_margin),
            "quality_ratio": safe_ratio(B_q, A_q),
            "massey_ratio": safe_ratio(B_massey, A_massey),
            "recent_ratio": safe_ratio(B_recent, A_recent),

            "target": 0
        })

    return pd.DataFrame(rows)