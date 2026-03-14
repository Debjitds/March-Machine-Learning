from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_model(X, y):

    lgb_models = []

    seeds = [42,7,99,2024,1337]

    print("Training LightGBM bagging models...")

    for seed in seeds:

        model = LGBMClassifier(

            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            random_state=seed

        )

        model.fit(X,y)

        lgb_models.append(model)


    print("Training Logistic Regression...")

    log_model = LogisticRegression(

        max_iter=3000

    )

    log_model.fit(X,y)


    print("Training Random Forest...")

    rf_model = RandomForestClassifier(

        n_estimators=350,
        max_depth=8,
        random_state=42

    )

    rf_model.fit(X,y)


    print("Building ensemble predictions for calibration...")

    lgb_preds=[]

    for m in lgb_models:

        lgb_preds.append(

            m.predict_proba(X)[:,1]

        )

    lgb_prob=np.mean(lgb_preds,axis=0)

    log_prob=log_model.predict_proba(X)[:,1]

    rf_prob=rf_model.predict_proba(X)[:,1]


    ensemble_pred=(

        0.6*lgb_prob+
        0.2*log_prob+
        0.2*rf_prob

    )


    print("Training probability calibrator...")

    calibrator=LogisticRegression(

        max_iter=2000

    )

    calibrator.fit(

        ensemble_pred.reshape(-1,1),
        y

    )


    return {

        "lgb":lgb_models,
        "log":log_model,
        "rf":rf_model,
        "calibrator":calibrator

    }