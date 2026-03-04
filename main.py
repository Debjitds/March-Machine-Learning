from src.data_loader import (
    load_mens_games,
    load_mens_teams,
    load_womens_games,
    load_womens_teams,
    load_sample_submission,
)

from src.features import compute_win_rate
from src.dataset import create_training_data, compute_team_avg_margin
from src.model import train_model
from src.elo import compute_elo_ratings
from src.submission import build_submission


def run_pipeline(games, teams, league_name="MEN"):
    """
    Runs full training pipeline for one league (Men/Women)
    """

    print(f"\n===== {league_name} PIPELINE =====")

    # 1️⃣ Win Rate
    print("Creating win-rate features...")
    win_rate = compute_win_rate(games)

    # 2️⃣ ELO Ratings
    print("Computing ELO ratings...")
    elo_ratings = compute_elo_ratings(games)

    # 3️⃣ Margin Feature
    print("Computing team average margins...")
    avg_margin = compute_team_avg_margin(games)

    # 4️⃣ Build Training Dataset
    print("Building training dataset...")
    train_df = create_training_data(games, win_rate, elo_ratings)

    # 5️⃣ Define Features
    feature_cols = [
        "A_wr",
        "B_wr",
        "A_elo",
        "B_elo",
        "elo_diff",
        "A_avg_margin",
        "B_avg_margin",
        "margin_diff"
    ]

    X = train_df[feature_cols]
    y = train_df["target"]

    # 6️⃣ Train Model
    print("Training model...")
    model = train_model(X, y)

    print(f"{league_name} model training complete.")

    return model, win_rate, elo_ratings, avg_margin


def main():

    print("\n====================================")
    print("  MARCH ML MANIA 2026 - ELO MODEL  ")
    print("====================================\n")

    # -----------------------------
    # MEN PIPELINE
    # -----------------------------
    m_games = load_mens_games()
    m_teams = load_mens_teams()

    men_model, men_wr, men_elo, men_margin = run_pipeline(
        m_games,
        m_teams,
        league_name="MEN"
    )

    # -----------------------------
    # WOMEN PIPELINE
    # -----------------------------
    w_games = load_womens_games()
    w_teams = load_womens_teams()

    women_model, women_wr, women_elo, women_margin = run_pipeline(
        w_games,
        w_teams,
        league_name="WOMEN"
    )

    # -----------------------------
    # LOAD SAMPLE SUBMISSION
    # -----------------------------
    print("\nLoading SampleSubmissionStage2.csv...")
    sample_df = load_sample_submission()
    print(f"Total required rows: {len(sample_df)}")

    # -----------------------------
    # BUILD FINAL SUBMISSION
    # -----------------------------
    print("\nGenerating predictions for required matchups...")

    build_submission(
        sample_df,
        men_model,
        men_wr,
        men_elo,
        men_margin,
        women_model,
        women_wr,
        women_elo,
        women_margin
    )

    print("\n✅ ELO + Margin submission file successfully created.")
    print("You can now upload it to Kaggle.\n")


if __name__ == "__main__":
    main()