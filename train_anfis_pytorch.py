"""Train gradient-based ANFIS implemented in PyTorch on Top-3 features.
Saves loss/epoch and model to `app/ml/models/anfis_pytorch_latest.pt`.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_pytorch import PyTorchANFIS

import torch
import torch.optim as optim
import torch.nn as nn


def main(epochs=50, lr=1e-2, batch_size=128):
    db = SessionLocal()
    processor = DataProcessor(db)
    df = processor.get_training_data()
    if df.empty:
        print('No data')
        return

    # small sample for speed
    df = df.sample(n=min(5000, len(df)), random_state=42)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top_3 = correlations.head(3).index.tolist()

    X = pd.DataFrame()
    for i, feat in enumerate(top_3):
        X[f'match{i+1}'] = df[feat]
    y = df['target']

    # train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X.values.astype(np.float32), y.values.astype(np.float32), test_size=0.2, random_state=42)

    device = 'cpu'
    model = PyTorchANFIS(n_inputs=3, n_mfs=3, consequent_order=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    n = X_train.shape[0]
    steps_per_epoch = max(1, n // batch_size)

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad()
            y_pred, _ = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        losses.append(epoch_loss)
        if ep % 5 == 0 or ep == 1:
            print(f'Epoch {ep}/{epochs} loss={epoch_loss:.6f}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_anfis_pytorch_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training loss')
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(results_dir, 'loss.png')
    plt.savefig(path, dpi=150)
    print('Saved loss plot:', path)

    models_dir = 'app/ml/models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'anfis_pytorch_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    latest = os.path.join(models_dir, 'anfis_pytorch_latest.pt')
    torch.save(model.state_dict(), latest)
    print('Saved model:', model_path)

    # quick eval
    model.eval()
    with torch.no_grad():
        preds = model.predict_numpy(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    db.close()


if __name__ == '__main__':
    main(epochs=15)
