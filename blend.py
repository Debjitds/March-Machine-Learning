import pandas as pd


def blend_submissions():

    print("Loading submissions...")

    v5 = pd.read_csv("outputs/submissions/submission_elo_v5.csv")
    v8 = pd.read_csv("outputs/submissions/submission_ensemble_v8.csv")
    v9 = pd.read_csv("outputs/submissions/submission_ensemble_v9.csv")

    print("Submissions loaded")

    # Ensure IDs match
    assert (v5["ID"] == v8["ID"]).all()
    assert (v5["ID"] == v9["ID"]).all()

    print("IDs verified")

    blend = v5.copy()

    print("Blending predictions...")

    blend["Pred"] = (
        0.5 * v5["Pred"]
        + 0.3 * v8["Pred"]
        + 0.2 * v9["Pred"]
    )

    output_path = "outputs/submissions/submission_blend_v10.csv"

    blend.to_csv(output_path, index=False)

    print("\nBlended submission created:")
    print(output_path)

    print("Rows:", len(blend))


if __name__ == "__main__":
    blend_submissions()