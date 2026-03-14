# src/stacker.py
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# import your loaders and feature builders
from src.data_loader import (
    load_mens_games, load_mens_teams, load_womens_games, load_mens_detailed_games
)
from src.features import compute_win_rate, compute_recent_form, compute_adjusted_efficiency
from src.elo import compute_elo_ratings
from src.massey import load_massey_ordinals
from src.team_quality import compute_team_quality
from src.dataset import create_training_data
from src.seeds import load_mens_seeds, build_seed_dict

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def _base_train_predict(X_train, y_train, X_val):
    """
    Train base learners on (X_train, y_train) and return predictions for X_val.
    We use one LGB (single seed) + Logistic + RF for OOF generation.
    """
    # LightGBM (single)
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict_proba(X_val)[:,1]

    # Logistic
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    log_pred = log.predict_proba(X_val)[:,1]

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_val)[:,1]

    return (lgb, log, rf), (lgb_pred, log_pred, rf_pred)


def run_stacking(n_splits=5):
    """
    Build OOF predictions (GroupKFold by season), train a meta-model on OOF preds,
    retrain final base models on full training set (with bagging for LGB), and save them.
    """

    print("Loading MEN data (stacking uses men data; you can adapt for women similarly)...")
    games = load_mens_games()
    teams = load_mens_teams()

    # compute features used by create_training_data
    win_rate = compute_win_rate(games)
    recent = compute_recent_form(games)
    elo = compute_elo_ratings(games)
    print("Loading Massey rankings...")
    massey = load_massey_ordinals("MEN")  # returns dict or mapping depending on your massey impl
    if isinstance(massey, dict) is False:
        # if load_massey_ordinals returns dataframe or computed mappings elsewhere, try compute
        try:
            massey = massey  # assume already mapping from team -> value
        except Exception:
            massey = {}

    quality = compute_team_quality(games)

    # adjusted efficiency from detailed games
    detailed = load_mens_detailed_games()
    off_eff, def_eff, adj_margin = compute_adjusted_efficiency(detailed)

    # Build train dataframe (includes 'season' column)
    train_df = create_training_data(
        games, win_rate, elo, massey, recent, quality, adj_margin
    )

    # feature columns — must match what submission uses (except season & target)
    feature_cols = [c for c in train_df.columns if c not in ("target", "season")]

    X_all = train_df[feature_cols].values
    y_all = train_df["target"].values
    seasons = train_df["season"].values

    print("Using GroupKFold by season with", n_splits, "splits")

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros((len(train_df), 3))  # columns: lgb, log, rf

    fold_idx = 0
    for train_idx, val_idx in gkf.split(X_all, y_all, groups=seasons):
        print(f"Fold {fold_idx+1}/{n_splits} — train {len(train_idx)} val {len(val_idx)}")
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        models, preds = _base_train_predict(X_tr, y_tr, X_val)
        lgb_pred, log_pred, rf_pred = preds

        oof_preds[val_idx, 0] = lgb_pred
        oof_preds[val_idx, 1] = log_pred
        oof_preds[val_idx, 2] = rf_pred

        # optional: save fold base models if you want to debug
        with open(f"{MODEL_DIR}/base_fold_{fold_idx}_lgb.pkl", "wb") as f:
            pickle.dump(models[0], f)
        with open(f"{MODEL_DIR}/base_fold_{fold_idx}_log.pkl", "wb") as f:
            pickle.dump(models[1], f)
        with open(f"{MODEL_DIR}/base_fold_{fold_idx}_rf.pkl", "wb") as f:
            pickle.dump(models[2], f)

        fold_idx += 1

    # Compute OOF Brier for each base
    lgb_oof_brier = brier_score_loss(y_all, oof_preds[:,0])
    log_oof_brier = brier_score_loss(y_all, oof_preds[:,1])
    rf_oof_brier = brier_score_loss(y_all, oof_preds[:,2])
    print("OOF Brier — LGB:", lgb_oof_brier, "Log:", log_oof_brier, "RF:", rf_oof_brier)

    # Train meta-model on OOF preds
    meta_X = oof_preds
    meta_y = y_all
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_X, meta_y)
    meta_oof_pred = meta_model.predict_proba(meta_X)[:,1]
    meta_brier = brier_score_loss(meta_y, meta_oof_pred)
    print("Meta OOF Brier:", meta_brier)

    # Save meta-model
    with open(f"{MODEL_DIR}/meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)

    # Retrain final base models on full training data (LGB bagging)
    print("Retraining final base models on full training set (LGB bagging)...")
    # LGB bagging
    lgb_bag = []
    seeds = [42, 7, 99, 2024, 1337]
    for s in seeds:
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=s)
        m.fit(X_all, y_all)
        lgb_bag.append(m)

    # Fit log and rf on full data
    log_full = LogisticRegression(max_iter=1000).fit(X_all, y_all)
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42).fit(X_all, y_all)

    base_models = {"lgb": lgb_bag, "log": log_full, "rf": rf_full}

    # Save base models
    with open(f"{MODEL_DIR}/base_models.pkl", "wb") as f:
        pickle.dump(base_models, f)

    print("Stacking complete. Models saved to", MODEL_DIR)
    print("Final OOF (meta) brier:", meta_brier)
    return meta_brier

if __name__ == "__main__":
    run_brier = run_stacking(n_splits=5)
    print("Done. OOF meta brier:", run_brier)