import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold

sys.path.insert(0, os.path.abspath('.'))
from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_xanfis import XANFISWrapper


def generate_cv_boxplot(cv_scores, save_path):
    plt.figure(figsize=(8, 6))

    bp = plt.boxplot([cv_scores], positions=[1], widths=0.6, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    plt.xticks([1], ['RMSE'])
    plt.ylabel('Error Value', fontsize=12)
    plt.title('Model Stability (Cross-Validation)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    y = cv_scores
    x = np.random.normal(1, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.6, markersize=10, label='Fold Result')
    plt.legend()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_analysis_dashboard(history, y_true, y_pred, feature_names, save_path):
    errors = y_pred - y_true

    fig = plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])

    if len(train_losses) > 0:
        epochs_range = range(1, len(train_losses) + 1)
        plt.plot(epochs_range, train_losses, label='Train Loss', linewidth=2, color='tab:blue')

        if len(val_losses) > 0:
            plt.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2, color='tab:orange')

        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Loss (MSE)', fontsize=11)
        plt.title('Training History', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        final_train = train_losses[-1] if train_losses else 0
        final_val = val_losses[-1] if val_losses else 0
        plt.text(0.02, 0.98, f'Final Train: {final_train:.6f}\nFinal Val: {final_val:.6f}',
                 transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'Loss history not captured', ha='center', va='center')
        plt.title('Training History', fontsize=13, fontweight='bold')

    plt.subplot(1, 3, 2)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', s=40, color='tab:green')

    min_val, max_val = 0, 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Ideal Prediction')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    plt.xlabel('Actual (0-1)', fontsize=11)
    plt.ylabel('Predicted (0-1)', fontsize=11)
    plt.title(f'Actual vs Predicted\nRMSE={rmse:.4f}, R²={r2:.4f}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.subplot(1, 3, 3)
    plt.hist(errors, bins=30, color='lightblue', edgecolor='black', alpha=0.8)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2,
                label=f'Mean Error: {np.mean(errors):.4f}')

    mae = mean_absolute_error(y_true, y_pred)
    plt.xlabel('Error (Pred - True)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.title(f'Error Distribution\nMAE={mae:.4f}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(epochs=150):
    print("=" * 80)
    print("TRAINING ANFIS (XANFIS WRAPPER) - REGRESSION TASK")
    print("=" * 80)

    db = SessionLocal()
    processor = DataProcessor(db)
    df = processor.get_training_data()

    if df.empty:
        print('Error: No data found in database.')
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_xanfis_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results folder: {results_dir}")

    df = df.sample(n=min(5000, len(df)), random_state=42)
    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    top_3 = df[feature_cols].corrwith(df['target']).sort_values(ascending=False).head(3).index.tolist()
    feature_names = [f.replace('m_', '') for f in top_3]

    print(f"Selected Features (Top 3): {feature_names}")

    X = pd.DataFrame({f'match{i + 1}': df[feat] for i, feat in enumerate(top_3)})
    y = df['target']

    print(f"\nSTEP 1: Cross-Validation (3-Fold)...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        cv_model = XANFISWrapper(n_inputs=3, n_mfs=3)
        cv_model.fit(X_fold_train, y_fold_train, epochs=30, lr=1e-3)

        preds = cv_model.predict(X_fold_val)
        rmse_val = np.sqrt(mean_squared_error(y_fold_val, preds))
        cv_scores.append(rmse_val)
        print(f"   Fold {fold + 1} RMSE: {rmse_val:.4f}")

    avg_cv_rmse = np.mean(cv_scores)

    generate_cv_boxplot(cv_scores, os.path.join(results_dir, 'cross_validation.png'))
    print("Saved: cross_validation.png")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f'\nSTEP 2: Building and training final model...')
    model = XANFISWrapper(n_inputs=3, n_mfs=3)

    print(f'   Building network (1 epoch for initialization)...')
    model.fit(X_train, y_train, epochs=1, lr=1e-4, batch_size=32)

    print(f'   Saving initial network state...')
    import copy
    initial_state = copy.deepcopy(model.model.network.state_dict())

    print(f'   Plotting initial membership functions...')
    model.plot_mfs(save_path=os.path.join(results_dir, 'mfs_before.png'))

    print(f'   Resetting network to initial state...')
    model.model.network.load_state_dict(initial_state)

    print(f'   Starting full training (lr=1e-4, epochs={epochs})...')
    history = model.fit(X_train, y_train, epochs=epochs, lr=1e-4, batch_size=32)

    print(f'\nSTEP 3: Generating reports and dashboards...')

    print(f'   Plotting trained membership functions...')
    model.plot_mfs(save_path=os.path.join(results_dir, 'mfs_after.png'))

    preds = model.predict(X_test)
    y_test_np = np.asarray(y_test).flatten()
    preds_np = np.asarray(preds).flatten()

    generate_analysis_dashboard(
        history, y_test_np, preds_np, feature_names,
        os.path.join(results_dir, 'anfis_analysis.png')
    )
    print("Saved: anfis_analysis.png")

    rmse = np.sqrt(mean_squared_error(y_test_np, preds_np))
    mae = mean_absolute_error(y_test_np, preds_np)
    r2 = r2_score(y_test_np, preds_np)

    if rmse < 0.15:
        quality = "EXCELLENT"
    elif rmse < 0.25:
        quality = "GOOD"
    else:
        quality = "AVERAGE"

    model.save_stable(os.path.join('app/ml/models', 'xanfis_latest.pkl'), top_3=top_3)

    report = f"""
================================================================================
TECHNICAL REPORT: XANFIS (REGRESSION)
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CROSS-VALIDATION RESULTS (3-FOLD):
----------------------------------
RMSE: {avg_cv_rmse:.4f} +/- {np.std(cv_scores):.4f}

TEST SET RESULTS:
-----------------
RMSE: {rmse:.4f} (scale 0-1)
MAE:  {mae:.4f} (scale 0-1)
R2:   {r2:.4f}

Model Quality: {quality}

PARAMETERS:
-----------
- Epochs: {epochs}
- Inputs: {', '.join(feature_names)}
- Loss Function: Mean Squared Error (MSE)

ANFIS ARCHITECTURE (5 LAYERS):
-------------------------------
Layer 1: Input Layer (3 inputs)
Layer 2: Fuzzification (Gaussian MFs, 3 per input = 27 total)
Layer 3: Rule Layer (Fuzzy AND operations)
Layer 4: Normalization Layer
Layer 5: Defuzzification (Linear output combination)

GENERATED PLOTS:
----------------
1. anfis_analysis.png  -> Composite Dashboard (Loss, Scatter, Histogram)
2. cross_validation.png -> Stability Boxplot
3. mfs_before/after.png -> Membership Functions (Gaussian)

================================================================================
"""

    with open(os.path.join(results_dir, 'raport_techniczny.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'\nFINAL REPORT:')
    print(f'   RMSE: {rmse:.4f}')
    print(f'   R2:   {r2:.4f}')
    print(f'   Quality: {quality}')
    print(f'Output folder: {results_dir}')
    print("=" * 80)
    db.close()


if __name__ == '__main__':
    main(epochs=150)