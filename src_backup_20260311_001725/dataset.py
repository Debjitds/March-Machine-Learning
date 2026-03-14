# src/dataset.py
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
    """
    Build training rows. Includes 'season' column for GroupKFold.
    Feature order is stable and must match submission feature construction.
    """
    rows = []

    for _, g in games.iterrows():
        # keep same mapping as before
        season = g.get("Season", None)
        A = g["WTeamID"]
        B = g["LTeamID"]

        A_adj = adj_margin.get(A, 0)
        B_adj = adj_margin.get(B, 0)

        A_elo = elo.get(A, 1500)
        B_elo = elo.get(B, 1500)

        rows.append({
            "season": season,

            "A_wr": win_rate.get(A, 0.5),
            "B_wr": win_rate.get(B, 0.5),

            "A_recent": recent.get(A, 0.5),
            "B_recent": recent.get(B, 0.5),
            "recent_diff": recent.get(A, 0.5) - recent.get(B, 0.5),

            "A_elo": A_elo,
            "B_elo": B_elo,
            "elo_diff": A_elo - B_elo,

            "A_massey": massey.get(A, 100),
            "B_massey": massey.get(B, 100),
            "massey_diff": massey.get(A, 100) - massey.get(B, 100),

            "A_quality": quality.get(A, 0),
            "B_quality": quality.get(B, 0),
            "quality_diff": quality.get(A, 0) - quality.get(B, 0),

            "A_adj": A_adj,
            "B_adj": B_adj,
            "adj_diff": A_adj - B_adj,

            # interaction features
            "elo_adj_interaction": (A_elo - B_elo) * (A_adj - B_adj),
            "seed_adj_interaction": 0.0,  # seed not available here (safe placeholder)

            "target": 1
        })

        rows.append({
            "season": season,

            "A_wr": win_rate.get(B, 0.5),
            "B_wr": win_rate.get(A, 0.5),

            "A_recent": recent.get(B, 0.5),
            "B_recent": recent.get(A, 0.5),
            "recent_diff": recent.get(B, 0.5) - recent.get(A, 0.5),

            "A_elo": B_elo,
            "B_elo": A_elo,
            "elo_diff": B_elo - A_elo,

            "A_massey": massey.get(B, 100),
            "B_massey": massey.get(A, 100),
            "massey_diff": massey.get(B, 100) - massey.get(A, 100),

            "A_quality": quality.get(B, 0),
            "B_quality": quality.get(A, 0),
            "quality_diff": quality.get(B, 0) - quality.get(A, 0),

            "A_adj": B_adj,
            "B_adj": A_adj,
            "adj_diff": B_adj - A_adj,

            "elo_adj_interaction": (B_elo - A_elo) * (B_adj - A_adj),
            "seed_adj_interaction": 0.0,

            "target": 0
        })

    return pd.DataFrame(rows)