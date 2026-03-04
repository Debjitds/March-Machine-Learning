from lightgbm import LGBMClassifier

def train_model(X, y):

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X, y)

    return model