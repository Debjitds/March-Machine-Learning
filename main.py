from src.data_loader import (
    load_mens_games,
    load_mens_teams,
    load_womens_games,
    load_womens_teams,
    load_sample_submission,
)

from src.features import compute_win_rate, compute_recent_win_rate
from src.dataset import create_training_data, compute_team_avg_margin
from src.model import train_model
from src.elo import compute_elo_ratings
from src.submission import build_submission
from src.massey import load_massey_ratings


def run_pipeline(games, teams, league_name="MEN"):

    print(f"\n===== {league_name} PIPELINE =====")

    print("Creating win-rate features...")
    win_rate = compute_win_rate(games)

    print("Computing recent form...")
    recent_wr = compute_recent_win_rate(games)

    print("Computing ELO ratings...")
    elo_ratings = compute_elo_ratings(games)

    print("Computing team average margins...")
    avg_margin = compute_team_avg_margin(games)

    if league_name == "MEN":
        print("Loading Massey Ordinals...")
        massey_ratings = load_massey_ratings()
    else:
        massey_ratings = None

    print("Building training dataset...")

    train_df = create_training_data(
        games,
        win_rate,
        elo_ratings,
        massey_ratings=massey_ratings,
        recent_wr=recent_wr
    )

    feature_cols = [

        "A_wr","B_wr",

        "A_recent_wr","B_recent_wr","recent_wr_diff",

        "A_elo","B_elo","elo_diff",

        "A_avg_margin","B_avg_margin","margin_diff",

        "A_massey","B_massey","massey_diff"
    ]

    X = train_df[feature_cols]
    y = train_df["target"]

    print("Training model...")
    model = train_model(X, y)

    print(f"{league_name} model training complete.")

    return model, win_rate, elo_ratings, avg_margin, massey_ratings, recent_wr


def main():

    print("\n====================================")
    print("  MARCH ML MANIA 2026 - ELO MODEL  ")
    print("====================================\n")

    m_games = load_mens_games()
    m_teams = load_mens_teams()

    men_model, men_wr, men_elo, men_margin, men_massey, men_recent = run_pipeline(
        m_games,
        m_teams,
        league_name="MEN"
    )

    w_games = load_womens_games()
    w_teams = load_womens_teams()

    women_model, women_wr, women_elo, women_margin, women_massey, women_recent = run_pipeline(
        w_games,
        w_teams,
        league_name="WOMEN"
    )

    print("\nLoading SampleSubmissionStage2.csv...")
    sample_df = load_sample_submission()

    print("\nGenerating predictions...")

    build_submission(
        sample_df,
        men_model, men_wr, men_elo, men_margin, men_massey, men_recent,
        women_model, women_wr, women_elo, women_margin, women_massey, women_recent
    )

    print("\nSubmission file generated.")


if __name__ == "__main__":
    main()