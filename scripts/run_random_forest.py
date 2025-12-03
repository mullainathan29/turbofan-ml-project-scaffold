import argparse
import pandas as pd
from src.data.loader import load_fd_subset
from src.features.build_features import add_rul, scale_features, FEATURES
from src.models.random_forest import train_random_forest
from src.models.baseline import evaluate_regression
from src.visualization.plots import plot_pred_vs_true

def main(subset='FD001'):
    train, test, rul = load_fd_subset('data/raw', subset=subset)
    train = add_rul(train)
    train, scaler = scale_features(train, fit=True, scaler=None)

    X = train[FEATURES]
    y = train['RUL']

    model = train_random_forest(X, y)
    preds = model.predict(X)
    metrics = evaluate_regression(y, preds)

    print("Random Forest metrics:", metrics)
    plot_pred_vs_true(y, preds, title=f"Random Forest on {subset}")

    pd.Series(metrics).to_json(f"reports/{subset}_rf_metrics.json")
    print("Saved metrics to reports/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    args = ap.parse_args()
    main(args.subset)
