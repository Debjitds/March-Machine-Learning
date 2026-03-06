from src.data_loader import (
    load_mens_games,
    load_mens_teams,
    load_womens_games,
    load_womens_teams,
    load_sample_submission,
)

from src.features import compute_win_rate, compute_recent_win_rate
from src.dataset import create_training_data
from src.model import train_model
from src.elo import compute_elo_ratings
from src.submission import build_submission
from src.massey import load_massey_ratings
from src.team_quality import compute_team_quality


def run_pipeline(games, teams, league_name="MEN"):

    print(f"\n===== {league_name} PIPELINE =====")

    win_rate = compute_win_rate(games)
    recent_wr = compute_recent_win_rate(games)
    elo_ratings = compute_elo_ratings(games)
    team_quality = compute_team_quality(games)

    massey_ratings = load_massey_ratings() if league_name == "MEN" else None

    train_df = create_training_data(
        games,
        win_rate,
        elo_ratings,
        massey_ratings,
        recent_wr,
        team_quality
    )

    feature_cols = [

        "A_wr","B_wr",

        "A_recent_wr","B_recent_wr","recent_wr_diff",

        "A_elo","B_elo","elo_diff",

        "A_avg_margin","B_avg_margin","margin_diff",

        "A_massey","B_massey","massey_diff",

        "A_quality","B_quality","quality_diff",

        "elo_ratio","margin_ratio",
        "quality_ratio","massey_ratio","recent_ratio"
    ]

    X = train_df[feature_cols]
    y = train_df["target"]

    model = train_model(X, y)

    return model, win_rate, elo_ratings, massey_ratings, recent_wr, team_quality


def main():

    m_games = load_mens_games()
    m_teams = load_mens_teams()

    men_model, men_wr, men_elo, men_massey, men_recent, men_quality = run_pipeline(
        m_games, m_teams, "MEN"
    )

    w_games = load_womens_games()
    w_teams = load_womens_teams()

    women_model, women_wr, women_elo, women_massey, women_recent, women_quality = run_pipeline(
        w_games, w_teams, "WOMEN"
    )

    sample_df = load_sample_submission()

    build_submission(
        sample_df,
        men_model, men_wr, men_elo, men_massey, men_recent, men_quality,
        women_model, women_wr, women_elo, women_massey, women_recent, women_quality
    )


if __name__ == "__main__":
    main()