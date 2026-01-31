import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch

from app.ml.anfis_pytorch_5layer import ANFIS5Layer


def load_dataset(path):
    df = pd.read_csv(path)
    if 'rating' in df.columns:
        y = df['rating'].values.astype(float)
        X = df.drop(columns=['rating']).select_dtypes(include=[np.number]).values
    elif 'target' in df.columns:
        y = df['target'].values.astype(float)
        X = df.drop(columns=['target']).select_dtypes(include=[np.number]).values
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y = df[num_cols[-1]].values.astype(float)
        X = df[num_cols[:-1]].values.astype(float)
    return X, y


def find_best_checkpoint(out_dir):
    pattern = os.path.join(out_dir, 'anfis5_fold*_best.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    # return first (or you can pick best by filesize/date)
    return sorted(files)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='app/ml/datasets/anfis_synthetic.csv')
    parser.add_argument('--out', type=str, default='results_anfis_5layer_tuned')
    parser.add_argument('--n-mfs', type=int, default=3)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out
    save_dir = args.save_dir or out_dir
    os.makedirs(save_dir, exist_ok=True)

    X, y = load_dataset(args.dataset)
    print('Loaded', X.shape, y.shape)

    # baseline: predict mean
    y_mean = np.mean(y)
    baseline_rmse = mean_squared_error(y, np.full_like(y, y_mean)) ** 0.5

    ckpt = find_best_checkpoint(out_dir)
    if ckpt is None:
        print('No checkpoint found in', out_dir)
        return

    print('Using checkpoint', ckpt)
    state = torch.load(ckpt, map_location='cpu')

    n_inputs = X.shape[1]
    model = ANFIS5Layer(n_inputs=n_inputs, n_mfs=args.n_mfs)
    model.load_state_dict(state)
    model.eval()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32)

    with torch.no_grad():
        y_pred_t, info = model(Xt)
    y_pred = y_pred_t.numpy()

    rmse = mean_squared_error(y, y_pred) ** 0.5

    print(f'Baseline RMSE (mean): {baseline_rmse:.6f}')
    print(f'Model RMSE: {rmse:.6f}')

    # pred vs actual
    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, s=6, alpha=0.6)
    mn = min(y.min(), y_pred.min())
    mx = max(y.max(), y_pred.max())
    plt.plot([mn,mx], [mn,mx], 'r--')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    p1 = os.path.join(save_dir, 'pred_vs_actual_diagnose.png')
    plt.savefig(p1)
    plt.close()

    # residuals
    res = y - y_pred
    plt.figure()
    plt.hist(res, bins=60)
    plt.xlabel('residual (actual - pred)')
    plt.title('Residuals histogram')
    plt.tight_layout()
    p2 = os.path.join(save_dir, 'residuals_hist_diagnose.png')
    plt.savefig(p2)
    plt.close()

    # plot membership centers (first input)
    centers = info.get('centers')
    sigmas = info.get('sigmas')
    if centers is not None:
        centers = np.array(centers)
        sigmas = np.array(sigmas)
        n_inputs = centers.shape[1] if centers.ndim == 3 else centers.shape[0]
        # centers shape from model: (n_inputs, n_mfs)
        plt.figure()
        for i in range(centers.shape[0]):
            for m in range(centers.shape[1]):
                plt.scatter(i, centers[i,m], label=f'in{i}_mf{m}' if i==0 else None)
        plt.xlabel('input index')
        plt.ylabel('center value')
        plt.title('MF centers (per input)')
        plt.tight_layout()
        p3 = os.path.join(save_dir, 'mf_centers_diagnose.png')
        plt.savefig(p3)
        plt.close()

    # save metrics
    metrics = {'baseline_rmse': float(baseline_rmse), 'model_rmse': float(rmse)}
    pd.Series(metrics).to_csv(os.path.join(save_dir, 'diagnose_metrics.csv'))

    print('Saved plots to', save_dir)


if __name__ == '__main__':
    main()
