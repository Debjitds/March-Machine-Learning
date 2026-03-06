from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):

    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42
    )

    log_model = LogisticRegression(
        max_iter=1000
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    print("Training LightGBM...")
    lgb_model.fit(X, y)

    print("Training Logistic Regression...")
    log_model.fit(X, y)

    print("Training Random Forest...")
    rf_model.fit(X, y)

    return {
        "lgb": lgb_model,
        "log": log_model,
        "rf": rf_model
    }