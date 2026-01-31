"""
Train ANFIS and MLP on two datasets (DB + synthetic) and produce comparison plots.
Run: python train_compare_multi.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

sys.path.insert(0, os.path.abspath('.'))
from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_model import ANFISRecommender
from app.ml.generate_synthetic_dataset import generate


def train_anfis_on_df(df, results_dir, n_splits=5, max_samples=5000):
    os.makedirs(results_dir, exist_ok=True)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top_3_features = correlations.head(3).index.tolist()
    top_3_names = [f.replace('m_', '') for f in top_3_features]

    X = pd.DataFrame()
    for i, feat in enumerate(top_3_features):
        X[f'match{i+1}'] = df[feat]
    y = df['target']

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_preds = []
    all_actuals = []
    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        anfis = ANFISRecommender(aspect_names=['match1', 'match2', 'match3'])
        anfis.build_fuzzy_system()
        train_metrics = anfis.train(X_train, y_train, ridge_alpha=1e-3)
        eval_metrics = anfis.evaluate(X_test, y_test)

        fold_metrics.append({'fold': fold, 'train_rmse': train_metrics['rmse'], 'train_r2': train_metrics['r2'],
                             'test_rmse': eval_metrics['rmse'], 'test_r2': eval_metrics['r2']})
        all_preds.extend(eval_metrics['predictions'])
        all_actuals.extend(eval_metrics['actuals'])

        mf_dir = os.path.join(results_dir, f'mf_fold_{fold}')
        anfis.plot_membership_functions(mf_dir)

    test_rmses = np.array([m['test_rmse'] for m in fold_metrics])
    test_r2s = np.array([m['test_r2'] for m in fold_metrics])

    # summary
    summary = {
        'n_records': len(df),
        'top_3': top_3_names,
        'cv': fold_metrics,
        'cv_rmse_mean': float(np.mean(test_rmses)),
        'cv_r2_mean': float(np.mean(test_r2s)),
    }

    # save final model trained on full data
    final_model = ANFISRecommender(aspect_names=['match1', 'match2', 'match3'])
    final_model.build_fuzzy_system()
    final_model.train(X, y, ridge_alpha=1e-3)
    models_dir = os.path.join('app', 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'anfis_{os.path.basename(results_dir)}_{timestamp}.pkl'
    model_path = os.path.join(models_dir, model_filename)
    final_model.save(model_path)

    # plots
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

    # preds vs actuals
    plt.figure(figsize=(6, 6))
    plt.scatter(all_actuals, all_preds, alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual (0-1)')
    plt.ylabel('Predicted (0-1)')
    plt.title('Predicted vs Actual (all folds)')
    pa_plot = os.path.join(results_dir, 'pred_vs_actual.png')
    plt.savefig(pa_plot, dpi=150)
    plt.close()

    # save summary
    import json
    with open(os.path.join(results_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return {'summary': summary, 'model_path': model_path, 'results_dir': results_dir}


def train_mlp_on_df(df, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    MAX_SAMPLES = 5000
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    X = df[feature_cols].copy()
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', alpha=0.001,
                       batch_size=32, learning_rate='adaptive', learning_rate_init=0.001, max_iter=500,
                       random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    mlp.fit(X_train_scaled, y_train)

    y_pred = mlp.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 1)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # save model + scaler
    models_dir = os.path.join('app', 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'mlp_{os.path.basename(results_dir)}_{timestamp}.pkl'
    model_path = os.path.join(models_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': mlp, 'scaler': scaler}, f)

    # plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=30)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlabel('Actual (0-1)')
    plt.ylabel('Predicted (0-1)')
    plt.title(f'MLP: Pred vs Actual\nRMSE={rmse:.4f}')

    errors = y_pred - y_test
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    plt.xlabel('Prediction error')
    plt.title(f'Errors (MAE={mae:.4f})')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'mlp_evaluation.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    summary = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2), 'model_path': model_path, 'results_dir': results_dir}
    with open(os.path.join(results_dir, 'training_report.txt'), 'w', encoding='utf-8') as f:
        f.write(str(summary))

    return summary


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Dataset A: from DB
    db = SessionLocal()
    processor = DataProcessor(db)
    df_db = processor.get_training_data()
    db.close()
    if df_db.empty:
        print('No DB data available, aborting.')
        return

    # Dataset B: synthetic
    syn_path = os.path.join('app', 'ml', 'datasets', 'anfis_synthetic.csv')
    if not os.path.exists(syn_path):
        os.makedirs(os.path.dirname(syn_path), exist_ok=True)
        generate(path=syn_path, n_samples=3000)
    df_syn = pd.read_csv(syn_path)

    results = {}

    # train on DB dataset
    print('Training on DB dataset...')
    rdir_db = f'results_anfis_db_{timestamp}'
    res_anfis_db = train_anfis_on_df(df_db, rdir_db)
    res_mlp_db = train_mlp_on_df(df_db, rdir_db)
    results['db'] = {'anfis': res_anfis_db, 'mlp': res_mlp_db}

    # train on synthetic dataset
    print('Training on synthetic dataset...')
    rdir_syn = f'results_anfis_synthetic_{timestamp}'
    res_anfis_syn = train_anfis_on_df(df_syn, rdir_syn)
    res_mlp_syn = train_mlp_on_df(df_syn, rdir_syn)
    results['synthetic'] = {'anfis': res_anfis_syn, 'mlp': res_mlp_syn}

    # produce comparison plot
    datasets = ['db', 'synthetic']
    anfis_rmse = [results[d]['anfis']['summary']['cv_rmse_mean'] for d in datasets]
    mlp_rmse = [results[d]['mlp']['rmse'] for d in datasets]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(datasets))
    width = 0.35
    plt.bar(x - width/2, anfis_rmse, width, label='ANFIS')
    plt.bar(x + width/2, mlp_rmse, width, label='MLP')
    plt.xticks(x, datasets)
    plt.ylabel('RMSE')
    plt.title('ANFIS vs MLP RMSE on datasets')
    plt.legend()
    comp_dir = f'comparison_anfis_mlp_multi_{timestamp}'
    os.makedirs(comp_dir, exist_ok=True)
    comp_plot = os.path.join(comp_dir, 'rmse_comparison.png')
    plt.tight_layout()
    plt.savefig(comp_plot, dpi=150)
    plt.close()

    # save overall report
    import json
    with open(os.path.join(comp_dir, 'comparison_report.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('Comparison completed. Results saved to', comp_dir)


if __name__ == '__main__':
    main()
