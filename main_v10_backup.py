from src.data_loader import (
    load_mens_games,
    load_mens_teams,
    load_womens_games,
    load_womens_teams,
    load_sample_submission,
    load_mens_seeds,
    load_womens_seeds
)

from src.features import compute_win_rate, compute_recent_form
from src.model import train_model
from src.elo import compute_elo_ratings
from src.massey import load_massey_ordinals
from src.team_quality import compute_team_quality
from src.submission import build_submission
from src.seeds import build_seed_dict


def run_pipeline(games, teams, league_name="MEN"):

    print(f"\n===== {league_name} PIPELINE =====")

    win_rate = compute_win_rate(games)

    recent = compute_recent_form(games)

    elo = compute_elo_ratings(games)

    massey = load_massey_ordinals(league_name)

    quality = compute_team_quality(games)

    from src.dataset import create_training_data

    train_df = create_training_data(
        games,
        win_rate,
        elo,
        massey,
        recent,
        quality
    )

    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    model = train_model(X, y)

    return model, win_rate, elo, massey, recent, quality


def main():

    print("\nMARCH ML MANIA PIPELINE\n")

    m_games = load_mens_games()
    m_teams = load_mens_teams()

    men_model, men_wr, men_elo, men_massey, men_recent, men_quality = run_pipeline(
        m_games,
        m_teams,
        "MEN"
    )

    w_games = load_womens_games()
    w_teams = load_womens_teams()

    women_model, women_wr, women_elo, women_massey, women_recent, women_quality = run_pipeline(
        w_games,
        w_teams,
        "WOMEN"
    )

    print("\nLoading seeds...")

    men_seeds = build_seed_dict(load_mens_seeds())
    women_seeds = build_seed_dict(load_womens_seeds())

    sample_df = load_sample_submission()

    build_submission(
        sample_df,

        men_model, men_wr, men_elo, men_massey, men_recent, men_quality, men_seeds,

        women_model, women_wr, women_elo, women_massey, women_recent, women_quality, women_seeds
    )


if __name__ == "__main__":
    main()