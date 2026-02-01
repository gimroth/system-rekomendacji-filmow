"""Train ANFIS using x-anfis (gradient-based) on Top-3 features.

This script mirrors `train_anfis.py` data preparation but uses `XANFISWrapper`.
It trains a small model for a few epochs and saves loss/epoch plot and model.
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
from app.ml.anfis_xanfis import XANFISWrapper


def main(epochs=20):
    db = SessionLocal()
    processor = DataProcessor(db)
    df = processor.get_training_data()
    if df.empty:
        print('No data')
        return

    # small sample to test quickly
    df = df.sample(n=min(20000, len(df)), random_state=42)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top_3 = correlations.head(3).index.tolist()

    X = pd.DataFrame()
    for i, feat in enumerate(top_3):
        X[f'match{i+1}'] = df[feat]
    y = df['target']

    # simple train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Building XANFIS model...')
    model = XANFISWrapper(n_inputs=3, n_mfs=3) 

    print('Training (this may take a while)...')
    history = model.fit(X_train, y_train, epochs=epochs, lr=1e-3, batch_size=64)

    losses = history.get('loss', []) if isinstance(history, dict) else []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_xanfis_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    if losses:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training loss')
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(results_dir, 'loss.png')
        plt.savefig(path, dpi=150)
        print('Saved loss plot:', path)

    # save model
    models_dir = 'app/ml/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # UŻYWAMY NOWEJ METODY ZAPISU WAG
    latest_path = os.path.join(models_dir, 'xanfis_latest.pkl')
    model.save_stable(latest_path, top_3=top_3)
    
    print('✅ Saved STABLE model weights to:', latest_path)

    # quick eval
    preds = model.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    db.close()


if __name__ == '__main__':
    main(epochs=100)
