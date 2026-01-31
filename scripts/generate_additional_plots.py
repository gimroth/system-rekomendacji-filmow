import os
import sys
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_model import ANFISRecommender


def load_anfis_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_mlp_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def generate_for_results(results_dir, dataset_type):
    # load comparison_report to find model paths if available
    comp_json = None
    comp_path = os.path.join('comparison_anfis_mlp_multi_20260130_004610', 'comparison_report.json')
    if os.path.exists(comp_path):
        with open(comp_path, 'r', encoding='utf-8') as f:
            comp_json = json.load(f)

    # determine data
    if dataset_type == 'db':
        db = SessionLocal()
        dp = DataProcessor(db)
        df = dp.get_training_data()
        db.close()
    else:
        syn_path = os.path.join('app', 'ml', 'datasets', 'anfis_synthetic.csv')
        df = pd.read_csv(syn_path)

    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    top3 = correlations.head(3).index.tolist()

    X = pd.DataFrame({f'match{i+1}': df[feat].values for i, feat in enumerate(top3)})
    y = df['target'].values

    # try to find model paths
    anfis_path = None
    mlp_path = None
    if comp_json and dataset_type in comp_json:
        anfis_path = comp_json[dataset_type]['anfis'].get('model_path')
        mlp_path = comp_json[dataset_type]['mlp'].get('model_path')

    # fallback: look into models dir
    models_dir = os.path.join('app', 'ml', 'models')
    if not anfis_path:
        for fn in os.listdir(models_dir):
            if fn.startswith('anfis') and dataset_type in fn:
                anfis_path = os.path.join(models_dir, fn)
                break
    if not mlp_path:
        for fn in os.listdir(models_dir):
            if fn.startswith('mlp') and dataset_type in fn:
                mlp_path = os.path.join(models_dir, fn)
                break

    preds = {}

    if anfis_path and os.path.exists(anfis_path):
        try:
            anfis = load_anfis_model(anfis_path)
            # ANFISRecommender is pickled object with evaluate/predict methods
            if hasattr(anfis, 'evaluate'):
                eval_res = anfis.evaluate(X, y)
                y_pred_anfis = np.array(eval_res.get('predictions', []))
            elif hasattr(anfis, 'predict'):
                # fall back: predict per-row
                y_pred_anfis = np.array([anfis.predict(row) for _, row in X.iterrows()])
            else:
                y_pred_anfis = np.array([])
            preds['anfis'] = y_pred_anfis
        except Exception as e:
            print('Failed load ANFIS:', e)

    if mlp_path and os.path.exists(mlp_path):
        try:
            mlp_blob = load_mlp_pickle(mlp_path)
            model = mlp_blob.get('model') if isinstance(mlp_blob, dict) else mlp_blob
            scaler = mlp_blob.get('scaler') if isinstance(mlp_blob, dict) else None
            X_mlp = df[feature_cols].values
            if scaler is not None:
                X_mlp = scaler.transform(X_mlp)
            y_pred_mlp = np.clip(model.predict(X_mlp), 0, 1)
            preds['mlp'] = y_pred_mlp
        except Exception as e:
            print('Failed load MLP:', e)

    outdir = results_dir
    os.makedirs(outdir, exist_ok=True)

    # produce comparison overlay plot if both preds exist
    if 'anfis' in preds and 'mlp' in preds:
        plt.figure(figsize=(6, 6))
        plt.scatter(y, preds['anfis'], alpha=0.4, s=10, label='ANFIS')
        plt.scatter(y, preds['mlp'], alpha=0.4, s=10, label='MLP')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title('Predicted vs Actual (ANFIS vs MLP)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'pred_vs_actual_overlay.png'), dpi=150)
        plt.close()

    # residuals and histograms per model
    for name, y_pred in preds.items():
        residuals = y - y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, residuals, alpha=0.4, s=10)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual (actual - pred)')
        plt.title(f'Residuals vs Predicted ({name})')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'residuals_vs_predicted_{name}.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=40, edgecolor='black', alpha=0.7)
        plt.xlabel('Residual')
        plt.title(f'Residuals Histogram ({name})')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'residuals_hist_{name}.png'), dpi=150)
        plt.close()

    return True


def main():
    # target result dirs
    targets = [
        ('results_anfis_db_20260130_004610', 'db'),
        ('results_anfis_synthetic_20260130_004610', 'synthetic')
    ]
    for d, dtype in targets:
        if os.path.exists(d):
            print('Generating additional plots for', d)
            generate_for_results(d, dtype)
        else:
            print('Missing results dir', d)


if __name__ == '__main__':
    main()
