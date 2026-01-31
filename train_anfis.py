"""
ROZBUDOWANY SKRYPT TRENOWANIA ANFIS Z DYNAMICZNYMI TOP 3
Logika: match1, match2, match3 (Dopasowanie dynamiczne)
Zapisuje model do .pkl i generuje raporty wizualne.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_model import ANFISRecommender

print("=" * 80)
print("🎓 TRENOWANIE ANFIS: DYNAMICZNE TOP 3 + LOGIKA DOPASOWANIA (MATCH)")
print("=" * 80)

# =============================================================================
# KROK 1: Inicjalizacja
# =============================================================================
print("\n📊 KROK 1: Inicjalizacja")
print("-" * 80)

db = SessionLocal()
processor = DataProcessor(db)
print("✅ Połączono z bazą")

# =============================================================================
# KROK 2: Ekstrakcja danych
# =============================================================================
print("\n📊 KROK 2: Ekstrakcja features")
print("-" * 80)

df = processor.get_training_data()

if df.empty:
    print("❌ Brak danych!")
    db.close()
    sys.exit(1)

print(f"✅ Pobrano {len(df)} rekordów")
"""
ROZBUDOWANY SKRYPT TRENOWANIA ANFIS Z DYNAMICZNYMI TOP 3
Dodano: K-Fold cross-validation, metryki regresyjne (RMSE, MAE, R2), wykresy
funkcji przynależności i zapisy modeli.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_model import ANFISRecommender


def main(n_splits=5, max_samples=5000):
    print("=" * 80)
    print("🎓 TRENOWANIE ANFIS: DYNAMICZNE TOP 3 + LOGIKA DOPASOWANIA (MATCH)")
    print("=" * 80)

    db = SessionLocal()
    processor = DataProcessor(db)
    print("✅ Połączono z bazą")

    df = processor.get_training_data()
    if df.empty:
        print("❌ Brak danych!")
        db.close()
        sys.exit(1)

    print(f"✅ Pobrano {len(df)} rekordów")

    if len(df) > max_samples:
        print(f"⚡ Ograniczam do {max_samples} próbek")
        df = df.sample(n=max_samples, random_state=42)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top_3_features = correlations.head(3).index.tolist()
    top_3_names = [f.replace('m_', '') for f in top_3_features]
    print(f"💡 Top3 features: {top_3_names}")

    X = pd.DataFrame()
    for i, feat in enumerate(top_3_features):
        X[f'match{i+1}'] = df[feat]
    y = df['target']

    # K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_preds = []
    all_actuals = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_anfis_match_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold += 1
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        anfis = ANFISRecommender(aspect_names=['match1', 'match2', 'match3'])
        anfis.build_fuzzy_system()
        train_metrics = anfis.train(X_train, y_train, ridge_alpha=1e-3)
        eval_metrics = anfis.evaluate(X_test, y_test)

        print(f" Train RMSE: {train_metrics['rmse']:.4f}, R2: {train_metrics['r2']:.4f}")
        print(f" Test RMSE:  {eval_metrics['rmse']:.4f}, R2: {eval_metrics['r2']:.4f}")

        fold_metrics.append({'fold': fold, 'train_rmse': train_metrics['rmse'], 'train_r2': train_metrics['r2'],
                             'test_rmse': eval_metrics['rmse'], 'test_r2': eval_metrics['r2']})

        all_preds.extend(eval_metrics['predictions'])
        all_actuals.extend(eval_metrics['actuals'])

        # save membership plots per fold (they are static here but helpful)
        mf_dir = os.path.join(results_dir, f'mf_fold_{fold}')
        anfis.plot_membership_functions(mf_dir)

    # Aggregate metrics
    import numpy as _np
    test_rmses = _np.array([m['test_rmse'] for m in fold_metrics])
    test_r2s = _np.array([m['test_r2'] for m in fold_metrics])

    print("\n=== Cross-validation summary ===")
    print(f" Test RMSE mean: {_np.mean(test_rmses):.4f} std: {_np.std(test_rmses):.4f}")
    print(f" Test R2   mean: {_np.mean(test_r2s):.4f} std: {_np.std(test_r2s):.4f}")

    # Save CV metrics plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar([m['fold'] for m in fold_metrics], [m['test_rmse'] for m in fold_metrics], color='C0')
    plt.title('Test RMSE per fold')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')

    plt.subplot(1, 2, 2)
    plt.bar([m['fold'] for m in fold_metrics], [m['test_r2'] for m in fold_metrics], color='C1')
    plt.title('Test R2 per fold')
    plt.xlabel('Fold')
    plt.ylabel('R2')

    plt.tight_layout()
    cv_plot = os.path.join(results_dir, 'cv_metrics.png')
    plt.savefig(cv_plot, dpi=150)
    plt.close()
    print(f"✅ CV metrics plot saved: {cv_plot}")

    # Scatter all folds preds vs actuals
    plt.figure(figsize=(6, 6))
    plt.scatter(all_actuals, all_preds, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual (0-1)')
    plt.ylabel('Predicted (0-1)')
    plt.title('Predicted vs Actual (all folds)')
    pa_plot = os.path.join(results_dir, 'pred_vs_actual.png')
    plt.savefig(pa_plot, dpi=150)
    plt.close()
    print(f"✅ Pred vs Actual plot saved: {pa_plot}")

    # Train final model on full data and save
    print('\nTraining final model on full dataset...')
    final_model = ANFISRecommender(aspect_names=['match1', 'match2', 'match3'])
    final_model.build_fuzzy_system()
    final_model.train(X, y, ridge_alpha=1e-3)
    models_dir = 'app/ml/models'
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f'anfis_match_{timestamp}.pkl'
    model_path = os.path.join(models_dir, model_filename)
    final_model.save(model_path)
    latest_path = os.path.join(models_dir, 'anfis_latest.pkl')
    final_model.save(latest_path)
    print(f"✅ Final model saved: {model_path}")

    # Save summary report
    report = {
        'timestamp': timestamp,
        'n_records': len(df),
        'top_3': top_3_names,
        'cv': fold_metrics,
        'cv_rmse_mean': float(_np.mean(test_rmses)),
        'cv_r2_mean': float(_np.mean(test_r2s)),
    }
    import json
    with open(os.path.join(results_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Summary saved: {os.path.join(results_dir, 'training_summary.json')}")

    db.close()


if __name__ == '__main__':
    main()
