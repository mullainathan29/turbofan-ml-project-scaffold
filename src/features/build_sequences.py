import numpy as np

def create_sequences(df, features, window=30):
    X, y = [], []
    for eid in df.engine_id.unique():
        eng = df[df.engine_id == eid]
        values = eng[features].values
        rul = eng['RUL'].values
        for i in range(len(eng) - window):
            X.append(values[i:i+window])
            y.append(rul[i+window])
    return np.array(X), np.array(y)
