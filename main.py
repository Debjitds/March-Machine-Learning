import pandas as pd

from src.data_loader import (
    load_mens_games,
    load_mens_teams,
    load_womens_games,
    load_womens_teams,
    load_sample_submission,
    load_mens_detailed_games
)

from src.features import (
    compute_win_rate,
    compute_recent_form,
    compute_adjusted_efficiency
)

from src.elo import compute_elo_ratings
from src.massey import load_massey_ordinals
from src.team_quality import compute_team_quality

from src.seeds import (
    load_mens_seeds,
    load_womens_seeds,
    build_seed_dict
)

from src.dataset import create_training_data
from src.model import train_model
from src.submission import build_submission


# =========================================================
# Pipeline Runner
# =========================================================

def run_pipeline(games, teams, league_name):

    print(f"\n===== {league_name} PIPELINE =====")

    print("Computing win rate...")
    win_rate = compute_win_rate(games)

    print("Computing recent form...")
    recent = compute_recent_form(games)

    print("Computing ELO ratings...")
    elo = compute_elo_ratings(games)

    print("Computing team quality...")
    quality = compute_team_quality(games)

    print("Loading Massey rankings...")
    massey = load_massey_ordinals(league_name)

    # ---------------------------------
    # Adjusted Efficiency Margin
    # ---------------------------------

    adj_margin = {}

    if league_name == "MEN":

        print("Loading detailed games...")
        detailed_games = load_mens_detailed_games()

        print("Computing Adjusted Efficiency Margin...")
        off_eff, def_eff, adj_margin = compute_adjusted_efficiency(detailed_games)

    else:
        # Women's dataset does not include detailed stats
        adj_margin = {}

    # ---------------------------------
    # Build Training Dataset
    # ---------------------------------

    print("Building training dataset...")

    train_df = create_training_data(
        games,
        win_rate,
        elo,
        massey,
        recent,
        quality,
        adj_margin
    )

    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    print("Training models...")

    models = train_model(X, y)

    return models, win_rate, elo, massey, recent, quality, adj_margin


# =========================================================
# Main
# =========================================================

def main():

    print("\n=========== MARCH MACHINE LEARNING MANIA PIPELINE ===========")

    # -------------------------------------------------
    # MEN DATA
    # -------------------------------------------------

    print("\nLoading MEN data...")

    men_games = load_mens_games()
    men_teams = load_mens_teams()

    men_models, men_wr, men_elo, men_massey, men_recent, men_quality, men_adj = run_pipeline(
        men_games,
        men_teams,
        "MEN"
    )

    # -------------------------------------------------
    # WOMEN DATA
    # -------------------------------------------------

    print("\nLoading WOMEN data...")

    women_games = load_womens_games()
    women_teams = load_womens_teams()

    women_models, women_wr, women_elo, women_massey, women_recent, women_quality, women_adj = run_pipeline(
        women_games,
        women_teams,
        "WOMEN"
    )

    # -------------------------------------------------
    # LOAD SEEDS
    # -------------------------------------------------

    print("\nLoading tournament seeds...")

    men_seeds_df = load_mens_seeds()
    women_seeds_df = load_womens_seeds()

    men_seeds = build_seed_dict(men_seeds_df)
    women_seeds = build_seed_dict(women_seeds_df)

    # -------------------------------------------------
    # LOAD SUBMISSION TEMPLATE
    # -------------------------------------------------

    print("\nLoading submission template...")

    sample_df = load_sample_submission()

    # -------------------------------------------------
    # BUILD SUBMISSION
    # -------------------------------------------------

    print("\nGenerating predictions...")

    build_submission(
        sample_df,

        men_models,
        men_wr,
        men_elo,
        men_massey,
        men_recent,
        men_quality,
        men_seeds,
        men_adj,

        women_models,
        women_wr,
        women_elo,
        women_massey,
        women_recent,
        women_quality,
        women_seeds,
        women_adj
    )

    print("\nPipeline completed successfully.")


# =========================================================

if __name__ == "__main__":
    main()