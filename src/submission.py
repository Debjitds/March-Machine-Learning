def build_submission(sample_df,
                     men_model, men_wr, men_elo, men_massey, men_recent, men_quality,
                     women_model, women_wr, women_elo, women_massey, women_recent, women_quality):

    preds = []

    for id_val in sample_df["ID"]:

        _, A, B = id_val.split("_")
        A = int(A)
        B = int(B)

        if A < 2000:
            model = men_model
            wr = men_wr
            elo = men_elo
            massey = men_massey
            recent = men_recent
            quality = men_quality
        else:
            model = women_model
            wr = women_wr
            elo = women_elo
            massey = None
            recent = women_recent
            quality = women_quality

        A_massey = massey.get(A,100) if massey else 100
        B_massey = massey.get(B,100) if massey else 100

        feat = [[

            wr.get(A,0), wr.get(B,0),

            recent.get(A,0.5), recent.get(B,0.5),
            recent.get(A,0.5)-recent.get(B,0.5),

            elo.get(A,1500), elo.get(B,1500),
            elo.get(A,1500)-elo.get(B,1500),

            0,0,0,

            A_massey, B_massey, A_massey-B_massey,

            quality.get(A,0), quality.get(B,0),
            quality.get(A,0)-quality.get(B,0)
        ]]

        prob = model.predict_proba(feat)[0][1]

        preds.append(prob)

    sample_df["Pred"] = preds
    sample_df.to_csv("outputs/submissions/submission_elo_v5.csv", index=False)

    print("submission_elo_v5.csv generated")