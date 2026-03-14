import pandas as pd

from src.data_loader import (
    load_mens_games,
    load_mens_teams,
    load_womens_games,
    load_womens_teams,
    load_sample_submission,
    load_mens_detailed_games,
    load_womens_detailed_games
)

from src.features import (
    compute_win_rate,
    compute_recent_form,
    compute_tempo_efficiency,
    compute_adjusted_efficiency
)

from src.elo import compute_elo

from src.team_quality import compute_team_quality

from src.massey import load_massey_ordinals

from src.dataset import create_training_data

from src.model import train_model

from src.submission import build_submission

from src.seeds import (
    load_mens_seeds,
    load_womens_seeds,
    build_seed_dict
)


print("========== MARCH MACHINE LEARNING MANIA PIPELINE ==========")


def run_pipeline(

    games,
    teams,
    detailed_games,
    league

):

    print(f"\n----- {league} PIPELINE -----")

    print("Computing win rate...")
    win_rate = compute_win_rate(games)

    print("Computing recent form...")
    recent = compute_recent_form(games)

    print("Computing ELO...")
    elo = compute_elo(games)

    print("Computing team quality...")
    quality = compute_team_quality(games)

    print("Loading Massey rankings...")

    if league == "MEN":
        massey = load_massey_ordinals("MEN")
    else:
        massey = {}

    print("Computing tempo efficiency...")

    off_eff,def_eff,tempo,net = compute_tempo_efficiency(
        detailed_games
    )

    print("Computing adjusted efficiency...")

    adj_off,adj_def,adj_net = compute_adjusted_efficiency(

        off_eff,
        def_eff,
        games

    )

    print("Building training data...")

    train_df = create_training_data(

        games,
        win_rate,
        elo,
        massey,
        recent,
        quality,
        adj_net,
        off_eff,
        def_eff,
        tempo,
        net

    )

    y = train_df["target"]

    X = train_df.drop(columns=["target"]).astype(float)

    print("Training models...")

    models = train_model(

        X,
        y

    )

    return (

        models,
        win_rate,
        elo,
        massey,
        recent,
        quality,
        adj_net,
        off_eff,
        def_eff,
        tempo,
        net

    )


def main():

    print("Loading MEN data...")

    men_games = load_mens_games()

    men_teams = load_mens_teams()

    men_detailed = load_mens_detailed_games()


    (

        men_models,
        men_wr,
        men_elo,
        men_massey,
        men_recent,
        men_quality,
        men_adj,
        men_off,
        men_def,
        men_tempo,
        men_net

    ) = run_pipeline(

        men_games,
        men_teams,
        men_detailed,
        "MEN"

    )


    print("\nLoading WOMEN data...")

    women_games = load_womens_games()

    women_teams = load_womens_teams()

    women_detailed = load_womens_detailed_games()


    (

        women_models,
        women_wr,
        women_elo,
        women_massey,
        women_recent,
        women_quality,
        women_adj,
        women_off,
        women_def,
        women_tempo,
        women_net

    ) = run_pipeline(

        women_games,
        women_teams,
        women_detailed,
        "WOMEN"

    )


    print("\nLoading seeds...")

    men_seeds_df = load_mens_seeds()

    women_seeds_df = load_womens_seeds()

    men_seeds = build_seed_dict(men_seeds_df)

    women_seeds = build_seed_dict(women_seeds_df)


    print("Loading submission template...")

    sample_df = load_sample_submission()


    print("Building submission...")

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
        women_adj,

        men_off,
        men_def,
        men_tempo,
        men_net,

        women_off,
        women_def,
        women_tempo,
        women_net

    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()