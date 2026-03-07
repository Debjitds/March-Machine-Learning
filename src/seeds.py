def seed_to_int(seed):

    if isinstance(seed, str):
        return int(seed[1:3])

    return 16


def build_seed_dict(seed_df):

    seed_dict = {}

    for _, row in seed_df.iterrows():

        team = row["TeamID"]
        seed = seed_to_int(row["Seed"])

        seed_dict[team] = seed

    return seed_dict