import numpy as np
import pandas as pd


def ensemble_predict(models,X):

    lgb_preds=[]

    for m in models["lgb"]:

        lgb_preds.append(

            m.predict_proba(X)[:,1]

        )

    lgb_prob=np.mean(lgb_preds,axis=0)

    log_prob=models["log"].predict_proba(X)[:,1]

    rf_prob=models["rf"].predict_proba(X)[:,1]


    preds=(

        0.6*lgb_prob+
        0.2*log_prob+
        0.2*rf_prob

    )


    # CALIBRATION STEP

    calibrated=models["calibrator"].predict_proba(

        preds.reshape(-1,1)

    )[:,1]


    calibrated=np.clip(

        calibrated,
        0.02,
        0.98

    )

    return calibrated



def build_submission(

sample_df,

men_models,men_wr,men_elo,
men_massey,men_recent,
men_quality,men_seeds,
men_adj,

women_models,women_wr,
women_elo,women_massey,
women_recent,women_quality,
women_seeds,women_adj,

men_off,men_def,
men_tempo,men_net,

women_off,women_def,
women_tempo,women_net

):

    rows=[]

    for id_val in sample_df["ID"]:

        _,A,B=id_val.split("_")

        A=int(A)
        B=int(B)

        if A<2000:

            wr=men_wr
            elo=men_elo
            massey=men_massey
            recent=men_recent
            quality=men_quality
            seeds=men_seeds
            adj=men_adj

            off=men_off
            deff=men_def
            tempo=men_tempo
            net=men_net

            models=men_models

        else:

            wr=women_wr
            elo=women_elo
            massey=women_massey
            recent=women_recent
            quality=women_quality
            seeds=women_seeds
            adj=women_adj

            off=women_off
            deff=women_def
            tempo=women_tempo
            net=women_net

            models=women_models


        A_elo=elo.get(A,1500)
        B_elo=elo.get(B,1500)

        A_adj=adj.get(A,0)
        B_adj=adj.get(B,0)

        A_off=off.get(A,100)
        B_off=off.get(B,100)

        A_def=deff.get(A,100)
        B_def=deff.get(B,100)

        A_tempo=tempo.get(A,70)
        B_tempo=tempo.get(B,70)

        A_net=net.get(A,0)
        B_net=net.get(B,0)


        rows.append({

        "A_wr":wr.get(A,0.5),
        "B_wr":wr.get(B,0.5),

        "A_recent":recent.get(A,0.5),
        "B_recent":recent.get(B,0.5),

        "recent_diff":
        recent.get(A,0.5)-
        recent.get(B,0.5),

        "A_elo":A_elo,
        "B_elo":B_elo,

        "elo_diff":
        A_elo-B_elo,

        "A_massey":
        massey.get(A,100),

        "B_massey":
        massey.get(B,100),

        "massey_diff":
        massey.get(A,100)-
        massey.get(B,100),

        "A_quality":
        quality.get(A,0),

        "B_quality":
        quality.get(B,0),

        "quality_diff":
        quality.get(A,0)-
        quality.get(B,0),

        "A_adj":A_adj,
        "B_adj":B_adj,

        "adj_diff":
        A_adj-B_adj,

        "A_off":A_off,
        "B_off":B_off,

        "off_diff":
        A_off-B_off,

        "A_def":A_def,
        "B_def":B_def,

        "def_diff":
        A_def-B_def,

        "A_tempo":A_tempo,
        "B_tempo":B_tempo,

        "tempo_diff":
        A_tempo-B_tempo,

        "A_net":A_net,
        "B_net":B_net,

        "net_diff":
        A_net-B_net,

        "elo_adj_interaction":
        (A_elo-B_elo)*
        (A_adj-B_adj)

        })


    X=pd.DataFrame(rows)

    preds=ensemble_predict(

        models,
        X

    )


    sample_df["Pred"]=preds


    sample_df.to_csv(

    "outputs/submissions/submission_ensemble_v15.csv",

    index=False

    )


    print("✅ Submission saved.")