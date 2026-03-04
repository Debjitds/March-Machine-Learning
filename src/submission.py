def build_submission(sample_df,
                     men_model, men_wr, men_elo, men_margin,
                     women_model, women_wr, women_elo, women_margin):
    preds = []

    for id_val in sample_df["ID"]:

        _, teamA, teamB = id_val.split("_")
        teamA = int(teamA)
        teamB = int(teamB)

        if teamA < 2000:
            model = men_model
            wr = men_wr
            elo = men_elo
            avg_margin = men_margin
        else:
            model = women_model
            wr = women_wr
            elo = women_elo
            avg_margin = women_margin

        feat = [[
            wr.get(teamA, 0),
            wr.get(teamB, 0),
            elo.get(teamA, 1500),
            elo.get(teamB, 1500),
            elo.get(teamA, 1500) - elo.get(teamB, 1500),

            avg_margin.get(teamA, 0),
            avg_margin.get(teamB, 0),
            avg_margin.get(teamA, 0) - avg_margin.get(teamB, 0)
        ]]

        prob = model.predict_proba(feat)[0][1]
        preds.append(prob)

    sample_df["Pred"] = preds
    sample_df.to_csv("outputs/submissions/submission_elo_v2.csv", index=False)

    print("ELO submission saved.")