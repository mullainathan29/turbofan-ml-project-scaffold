import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def compute_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df = df.merge(max_cycle.rename("max_c"), on="engine_id")
    df["RUL"] = df["max_c"] - df["cycle"]
    df.drop(columns=["max_c"], inplace=True)
    return df

def scale_features(df, scaler=None):
    feat_cols = [c for c in df.columns if "sensor" in c or "op_setting" in c]
    if scaler is None:
        scaler = MinMaxScaler()
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
    else:
        df[feat_cols] = scaler.transform(df[feat_cols])
    return df, scaler

def make_delta_features(df, features):
    for f in features:
        df[f + "_delta"] = df.groupby("engine_id")[f].diff()
    return df

def make_rolling_features(df, features, window=5):
    for f in features:
        df[f + "_roll_mean"] = (
            df.groupby("engine_id")[f]
            .rolling(window)
            .mean()
            .reset_index(0, drop=True)
        )
    return df
