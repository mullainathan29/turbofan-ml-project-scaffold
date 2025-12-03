import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.loader import load_fd_subset
from src.features.build_features import add_rul, scale_features, FEATURES
from src.features.build_sequences import create_sequences
from src.models.lstm_torch import RULLSTM

def main(subset='FD001'):
    print("Loading dataset...")
    train, test, rul = load_fd_subset('data/raw', subset=subset)

    train = add_rul(train)
    train, _ = scale_features(train)

    print("Building sequences...")
    X_train, y_train = create_sequences(train, FEATURES, window=30)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=128,
                        shuffle=True)

    model = RULLSTM(input_dim=len(FEATURES))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training LSTM...")
    for epoch in range(10):
        total_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), f"reports/{subset}_lstm_torch.pt")
    print("Saved model to reports/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    args = ap.parse_args()
    main(args.subset)
