from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):

    lgb_models = []

    seeds = [42, 7, 99, 2024, 1337]

    print("Training LightGBM bagging models...")

    for seed in seeds:

        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=seed
        )

        model.fit(X, y)

        lgb_models.append(model)

    print("Training Logistic Regression...")

    log_model = LogisticRegression(max_iter=1000)

    log_model.fit(X, y)

    print("Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    rf_model.fit(X, y)

    return {
        "lgb": lgb_models,
        "log": log_model,
        "rf": rf_model
    }