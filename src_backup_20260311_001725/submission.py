# src/submission.py
import numpy as np
import pickle

def ensemble_predict_legacy(models, X):
    # old ensemble (with LGB bagging)
    lgb_preds = []
    for m in models["lgb"]:
        lgb_preds.append(m.predict_proba(X)[:,1])
    lgb_prob = np.mean(lgb_preds, axis=0)
    log_prob = models["log"].predict_proba(X)[:,1]
    rf_prob = models["rf"].predict_proba(X)[:,1]
    preds = 0.5*lgb_prob + 0.25*log_prob + 0.25*rf_prob
    preds = np.clip(preds, 0.02, 0.98)
    return preds

def build_submission(
    sample_df,
    men_models, men_wr, men_elo, men_massey, men_recent, men_quality, men_seeds, men_adj,
    women_models, women_wr, women_elo, women_massey, women_recent, women_quality, women_seeds, women_adj,
    meta_model_path=None,
    base_models_path=None
):
    """
    If meta_model_path and base_models_path provided, use stacking:
      - build base-model predictions vector [lgb, log, rf] for each matchup,
      - then meta_model.predict_proba(base_preds) to get final prob.
    Otherwise use legacy ensemble.
    """

    # load stacked models if given
    meta_model = None
    base_models = None
    if meta_model_path and base_models_path:
        with open(meta_model_path, "rb") as f:
            meta_model = pickle.load(f)
        with open(base_models_path, "rb") as f:
            base_models = pickle.load(f)

    features = []
    base_preds_list = []  # when stacked: collect base predictions per matchup

    for id_val in sample_df["ID"]:
        _, A, B = id_val.split("_")
        A = int(A)
        B = int(B)

        if A < 2000:
            wr, elo, massey, recent, quality, seeds, adj, models = \
                men_wr, men_elo, men_massey, men_recent, men_quality, men_seeds, men_adj, men_models
        else:
            wr, elo, massey, recent, quality, seeds, adj, models = \
                women_wr, women_elo, women_massey, women_recent, women_quality, women_seeds, women_adj, women_models

        A_wr = wr.get(A, 0.5)
        B_wr = wr.get(B, 0.5)
        A_recent = recent.get(A, 0.5)
        B_recent = recent.get(B, 0.5)
        A_elo = elo.get(A, 1500)
        B_elo = elo.get(B, 1500)
        A_massey = massey.get(A, 100)
        B_massey = massey.get(B, 100)
        A_quality = quality.get(A, 0)
        B_quality = quality.get(B, 0)
        A_adj = adj.get(A, 0)
        B_adj = adj.get(B, 0)
        A_seed = seeds.get(A, 16)
        B_seed = seeds.get(B, 16)

        # produce feature vector that matches training order (only if needed)
        feat = [
            A_wr, B_wr,
            A_recent, B_recent, A_recent - B_recent,
            A_elo, B_elo, A_elo - B_elo,
            A_massey, B_massey, A_massey - B_massey,
            A_quality, B_quality, A_quality - B_quality,
            A_adj, B_adj, A_adj - B_adj,
            (A_elo - B_elo) * (A_adj - B_adj),
            (A_seed - B_seed) * (A_adj - B_adj)
        ]
        features.append(feat)

    X = np.array(features)

    # If stacking is enabled
    if meta_model is not None and base_models is not None:
        # compute base-model preds for all rows in blocks (memory friendly)
        # LightGBM bagging
        lgb_preds = []
        for m in base_models["lgb"]:
            lgb_preds.append(m.predict_proba(X)[:,1])
        lgb_prob = np.mean(lgb_preds, axis=0)
        log_prob = base_models["log"].predict_proba(X)[:,1]
        rf_prob = base_models["rf"].predict_proba(X)[:,1]

        base_stack = np.vstack([lgb_prob, log_prob, rf_prob]).T
        final_preds = meta_model.predict_proba(base_stack)[:,1]
        final_preds = np.clip(final_preds, 0.02, 0.98)
    else:
        # legacy mode per group (men/women)
        # split and call legacy ensemble_predict_legacy to keep previous behavior
        men_mask = sample_df["ID"].apply(lambda x: int(x.split("_")[1]) < 2000).values
        final_preds = np.zeros(len(sample_df))
        if men_mask.any():
            final_preds[men_mask] = ensemble_predict_legacy(men_models, X[men_mask])
        if (~men_mask).any():
            final_preds[~men_mask] = ensemble_predict_legacy(women_models, X[~men_mask])

    sample_df["Pred"] = final_preds
    out = "outputs/submissions/submission_ensemble_stacked.csv"
    sample_df.to_csv(out, index=False)
    print(" ✅ Submission saved:", out)