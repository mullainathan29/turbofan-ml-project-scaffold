import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]

def add_rul(train: pd.DataFrame) -> pd.DataFrame:
    last = train.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle':'max_cycle'})
    df = train.merge(last, on='engine_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop(columns=['max_cycle'])

def scale_features(df: pd.DataFrame, fit: bool=True, scaler: MinMaxScaler|None=None):
    scaler = scaler or MinMaxScaler()
    if fit:
        df[FEATURES] = scaler.fit_transform(df[FEATURES])
        return df, scaler
    else:
        df[FEATURES] = scaler.transform(df[FEATURES])
        return df
