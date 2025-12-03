from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X, y, n_estimators=200, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    return model
