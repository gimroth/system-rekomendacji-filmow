import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('.'))
from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_xanfis import XANFISWrapper

def main(epochs=150): # 150 epok wystarczy przy stabilnych danych
    db = SessionLocal()
    processor = DataProcessor(db)
    df = processor.get_training_data()
    
    if df.empty:
        print('Błąd: Brak danych.')
        return

    df = df.sample(n=min(30000, len(df)), random_state=42)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top_3 = correlations.head(3).index.tolist()

    X = pd.DataFrame({f'match{i+1}': df[feat] for i, feat in enumerate(top_3)})
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # POWRÓT DO n_mfs=3 DLA STABILNOŚCI NUMERYCZNEJ
    print(f'Budowanie modelu XANFIS (n_inputs=3, n_mfs=3)...')
    model = XANFISWrapper(n_inputs=3, n_mfs=3)

    # Umiarkowany lr dla n_mfs=3
    print(f'Rozpoczęcie treningu (lr=1e-4, epochs={epochs})...')
    history = model.fit(X_train, y_train, epochs=epochs, lr=1e-4, batch_size=32)

    # Wyniki i Wizualizacja
    results_dir = f'results_xanfis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(results_dir, exist_ok=True)
    
    losses = history.get('loss', []) if isinstance(history, dict) else []
    if losses:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epoka')
        plt.ylabel('Loss')
        plt.title('Krzywa uczenia ANFIS (n_mfs=4)')
        plt.savefig(os.path.join(results_dir, 'loss.png'))

    # Stabilny zapis
    models_dir = 'app/ml/models'
    os.makedirs(models_dir, exist_ok=True)
    latest_path = os.path.join(models_dir, 'xanfis_latest.pkl')
    model.save_stable(latest_path, top_3=top_3)
    print(f'✅ Model zapisany: {latest_path}')

    # METRYKI TESTOWE
    preds = model.predict(X_test)
    y_test_np = np.asarray(y_test).flatten()
    preds_np = np.asarray(preds).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_np, preds_np))
    mae = mean_absolute_error(y_test_np, preds_np)
    r2 = r2_score(y_test_np, preds_np)

    print(f'\n' + "="*40)
    print(f'KOŃCOWE METRYKI TESTOWE:')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE:  {mae:.4f}')
    print(f'R2:   {r2:.4f}')
    print("="*40)

    db.close()

if __name__ == '__main__':
    main()