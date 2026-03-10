import numpy as np


def ensemble_predict(models, X):

    lgb_prob = models["lgb"].predict_proba(X)[:,1]
    log_prob = models["log"].predict_proba(X)[:,1]
    rf_prob = models["rf"].predict_proba(X)[:,1]

    return 0.5*lgb_prob + 0.25*log_prob + 0.25*rf_prob


def build_submission(
    sample_df,
    men_models, men_wr, men_elo, men_massey, men_recent, men_quality, men_seeds, men_adj,
    women_models, women_wr, women_elo, women_massey, women_recent, women_quality, women_seeds, women_adj
):

    features = []

    for id_val in sample_df["ID"]:

        _, A, B = id_val.split("_")
        A = int(A)
        B = int(B)

        if A < 2000:

            wr, elo, massey, recent, quality, seeds, adj = \
                men_wr, men_elo, men_massey, men_recent, men_quality, men_seeds, men_adj

        else:

            wr, elo, massey, recent, quality, seeds, adj = \
                women_wr, women_elo, women_massey, women_recent, women_quality, women_seeds, women_adj

        features.append([

            wr.get(A,0.5),
            wr.get(B,0.5),

            recent.get(A,0.5),
            recent.get(B,0.5),
            recent.get(A,0.5) - recent.get(B,0.5),

            elo.get(A,1500),
            elo.get(B,1500),
            elo.get(A,1500) - elo.get(B,1500),

            massey.get(A,100),
            massey.get(B,100),
            massey.get(A,100) - massey.get(B,100),

            quality.get(A,0),
            quality.get(B,0),
            quality.get(A,0) - quality.get(B,0),

            adj.get(A,0),
            adj.get(B,0),
            adj.get(A,0) - adj.get(B,0)
        ])

    X = np.array(features)

    preds = np.zeros(len(sample_df))

    men_mask = sample_df["ID"].apply(lambda x: int(x.split("_")[1]) < 2000).values

    preds[men_mask] = ensemble_predict(men_models, X[men_mask])
    preds[~men_mask] = ensemble_predict(women_models, X[~men_mask])

    sample_df["Pred"] = preds

    output_path = "outputs/submissions/submission_ensemble_v12.csv"

    sample_df.to_csv(output_path, index=False)

    print("✅ Submission saved:", output_path)