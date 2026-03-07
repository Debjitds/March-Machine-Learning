import numpy as np
import pandas as pd


def safe_ratio(a, b):
    if b == 0:
        return 0
    return a / b


def ensemble_predict(models, X):

    lgb_prob = models["lgb"].predict_proba(X)[:, 1]
    log_prob = models["log"].predict_proba(X)[:, 1]
    rf_prob = models["rf"].predict_proba(X)[:, 1]

    return (
        0.5 * lgb_prob +
        0.25 * log_prob +
        0.25 * rf_prob
    )


def build_submission(
        sample_df,
        men_models, men_wr, men_elo, men_massey, men_recent, men_quality, men_seeds,
        women_models, women_wr, women_elo, women_massey, women_recent, women_quality, women_seeds):

    print("Preparing feature matrix...")

    features = []

    for id_val in sample_df["ID"]:

        _, A, B = id_val.split("_")
        A = int(A)
        B = int(B)

        if A < 2000:

            wr = men_wr
            elo = men_elo
            massey = men_massey
            recent = men_recent
            quality = men_quality
            seeds = men_seeds
            models = men_models

        else:

            wr = women_wr
            elo = women_elo
            massey = women_massey
            recent = women_recent
            quality = women_quality
            seeds = women_seeds
            models = women_models

        A_wr = wr.get(A, 0)
        B_wr = wr.get(B, 0)

        A_recent = recent.get(A, 0.5)
        B_recent = recent.get(B, 0.5)

        A_elo = elo.get(A, 1500)
        B_elo = elo.get(B, 1500)

        A_q = quality.get(A, 0)
        B_q = quality.get(B, 0)

        A_seed = seeds.get(A, 16)
        B_seed = seeds.get(B, 16)

        A_massey = massey.get(A, 100) if massey else 100
        B_massey = massey.get(B, 100) if massey else 100

        features.append([

            A_wr, B_wr,

            A_recent, B_recent,
            A_recent - B_recent,

            A_elo, B_elo,
            A_elo - B_elo,

            A_seed, B_seed,
            A_seed - B_seed,

            A_massey, B_massey,
            A_massey - B_massey,

            A_q, B_q,
            A_q - B_q,

            safe_ratio(A_elo, B_elo),
            safe_ratio(A_seed, B_seed),
            safe_ratio(A_q, B_q),
            safe_ratio(A_massey, B_massey),
            safe_ratio(A_recent, B_recent)
        ])

    X = np.array(features)

    print("Feature matrix built:", X.shape)

    men_mask = sample_df["ID"].apply(lambda x: int(x.split("_")[1]) < 2000).values

    preds = np.zeros(len(sample_df))

    if men_mask.any():

        preds[men_mask] = ensemble_predict(
            men_models,
            X[men_mask]
        )

    if (~men_mask).any():

        preds[~men_mask] = ensemble_predict(
            women_models,
            X[~men_mask]
        )

    sample_df["Pred"] = preds

    output_path = "outputs/submissions/submission_ensemble_v10.csv"

    sample_df.to_csv(output_path, index=False)

    print(" ✅ Submission saved:")
    print(output_path)