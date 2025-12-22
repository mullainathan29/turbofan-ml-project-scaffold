import pandas as pd
import os

COLS = ["engine_id", "cycle"] + \
       [f"op_setting_{i}" for i in range(1, 4)] + \
       [f"sensor_{i}" for i in range(1, 22)]

def load_single_subset(path, subset):
    train = pd.read_csv(
        os.path.join(path, f"train_{subset}.txt"),
        sep=r"\s+",
        header=None
    )
    train.columns = COLS
    return train

def load_all_subsets(path):
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    data = {}
    for s in subsets:
        data[s] = load_single_subset(path, s)
    return data
