import pandas as pd
from sklearn.linear_model import LogisticRegression


def compute_team_quality(games):
    """
    Learn a strength parameter for each team using logistic regression.
    """

    rows = []

    teams = set(games["WTeamID"]).union(set(games["LTeamID"]))
    teams = sorted(list(teams))

    team_to_idx = {t: i for i, t in enumerate(teams)}

    for _, g in games.iterrows():

        w = g["WTeamID"]
        l = g["LTeamID"]

        rows.append((team_to_idx[w], team_to_idx[l], 1))
        rows.append((team_to_idx[l], team_to_idx[w], 0))

    data = []

    for a, b, y in rows:
        data.append([a, b, y])

    df = pd.DataFrame(data, columns=["A", "B", "y"])

    X = pd.get_dummies(df[["A", "B"]].astype(str))
    y = df["y"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = model.coef_[0]

    quality = {}

    for team, idx in team_to_idx.items():
        col = f"A_{idx}"
        if col in X.columns:
            quality[team] = coefs[X.columns.get_loc(col)]
        else:
            quality[team] = 0

    return quality