import itertools

def generate_matchups(team_ids):

    pairs = []

    for a, b in itertools.combinations(team_ids, 2):
        pairs.append((a, b))

    return pairs