import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import torch

from app.ml.anfis_pytorch_5layer import ANFIS5Layer


def load_dataset(path):
    df = pd.read_csv(path)
    # Wybieramy tylko kolumny, które nas interesują
    feature_cols = ['story', 'acting', 'visuals', 'sound', 'direction']
    target_col = 'rating'
    
    # Sprawdzamy czy kolumny istnieją
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    
    # Skalowanie targetu do zakresu 0-1 (jeśli oceny są 1-5)
    # Model ANFIS najlepiej pracuje na wartościach znormalizowanych
    y = (y - 1.0) / 4.0 
    
    print(f"✅ Wczytano {len(X)} próbek.")
    print(f"Cechy: {feature_cols}")
    return X, y


def train_fold(X_train, y_train, X_val, y_val, args, out_dir=None, fold_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ANFIS5Layer(n_inputs=X_train.shape[1], n_mfs=args.n_mfs,
                        poly_degree=getattr(args, 'poly_degree', 1),
                        full_poly=getattr(args, 'full_poly', False)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)

    # initialize centers from data if requested
    init_mode = getattr(args, 'init_centers', 'random')
    if init_mode in ('percentile', 'kmeans'):
        centers = np.zeros((X_train.shape[1], args.n_mfs), dtype=float)
        if init_mode == 'percentile':
            for i in range(X_train.shape[1]):
                pct = np.linspace(0, 100, args.n_mfs)
                centers[i, :] = np.percentile(X_train[:, i], pct)
        else:
            # kmeans per feature (1D clusters)
            try:
                from sklearn.cluster import KMeans
                for i in range(X_train.shape[1]):
                    col = X_train[:, i].reshape(-1, 1)
                    km = KMeans(n_clusters=args.n_mfs, random_state=42).fit(col)
                    c = km.cluster_centers_.ravel()
                    centers[i, :] = np.sort(c)
            except Exception:
                # fallback to percentiles
                for i in range(X_train.shape[1]):
                    pct = np.linspace(0, 100, args.n_mfs)
                    centers[i, :] = np.percentile(X_train[:, i], pct)
        # centers expected in feature scale; convert to scaled X space
        centers_scaled = scaler_x.transform(centers.T).T if centers.size else centers
        try:
            model.set_centers(centers_scaled)
        except Exception:
            # ignore if shape mismatch
            pass

    # scale target
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    losses = []
    val_losses = []
    centers_evolution = []

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    best_pred = None
    best_state = None
    patience = getattr(args, 'patience', None)
    wait = 0

    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        y_pred, _ = model(X_train_t)
        loss = loss_fn(y_pred, y_train_t)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        centers_evolution.append(model.centers.detach().cpu().numpy())

        # validation
        model.eval()
        with torch.no_grad():
            y_val_pred_t, _ = model(X_val_t)
            val_loss = loss_fn(y_val_pred_t, y_val_t).item()
        val_losses.append(val_loss)

        # checkpoint on validation loss
        if out_dir is not None and val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_pred = y_val_pred_t.cpu().numpy()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            model_path = os.path.join(out_dir, f'anfis5_fold{fold_idx+1}_best.pt')
            torch.save(best_state, model_path)
            # save scalers alongside checkpoint
            scaler_path = os.path.join(out_dir, f'scalers_fold{fold_idx+1}.npz')
            try:
                np.savez(scaler_path, x_mean=scaler_x.mean_, x_scale=scaler_x.scale_, y_mean=scaler_y.mean_, y_scale=scaler_y.scale_)
            except Exception:
                # fallback: save as text
                np.savetxt(scaler_path + '.x_mean.txt', scaler_x.mean_)
            wait = 0
        else:
            if patience is not None:
                wait += 1
                if wait >= patience:
                    print(f'Early stopping at epoch {ep+1} (no improvement for {patience} epochs)')
                    break

    # if no best_pred was saved, evaluate with last model
    if best_pred is None:
        model.eval()
        with torch.no_grad():
            y_val_pred_t, _ = model(X_val_t)
        best_pred = y_val_pred_t.cpu().numpy()
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # inverse transform predictions to original scale
    best_pred_orig = scaler_y.inverse_transform(best_pred.reshape(-1, 1)).ravel()
    rmse = mean_squared_error(y_val, best_pred_orig) ** 0.5
    r2 = r2_score(y_val, best_pred_orig)

    info = {
        'model_state': best_state,
        'losses': losses,
        'val_losses': val_losses,
        'centers_evolution': np.array(centers_evolution),
        'rmse': rmse,
        'r2': r2,
    }
    return info


def plot_losses(all_losses, out_dir):
    plt.figure()
    for i, losses in enumerate(all_losses):
        plt.plot(losses, label=f'fold{i+1}')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('Training loss per fold')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'losses_per_fold.png')
    plt.savefig(path)
    plt.close()


def plot_centers_evolution(centers_evo, out_dir):
    # centers_evo: (epochs, n_inputs, n_mfs)
    epochs = centers_evo.shape[0]
    n_inputs = centers_evo.shape[1]
    n_mfs = centers_evo.shape[2]

    for i in range(n_inputs):
        plt.figure()
        for m in range(n_mfs):
            plt.plot(range(epochs), centers_evo[:, i, m], label=f'mf{m}')
        plt.xlabel('epoch')
        plt.ylabel('center value')
        plt.title(f'Centers evolution input {i}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'centers_input_{i}.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='app/ml/datasets/anfis_synthetic.csv',
                        help='CSV path to dataset')
    parser.add_argument('--n-mfs', type=int, default=2, dest='n_mfs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--init-centers', type=str, default='random', choices=['random', 'percentile', 'kmeans'])
    parser.add_argument('--poly-degree', type=int, default=1)
    parser.add_argument('--full-poly', action='store_true', help='use full polynomial cross-terms for degree=2')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out or f'results_anfis_5layer_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_dataset(args.dataset)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    all_losses = []
    saved_centers = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'Training fold {fold_idx+1}/{args.n_splits}')
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        info = train_fold(X_train, y_train, X_val, y_val, args, out_dir=out_dir, fold_idx=fold_idx)
        fold_metrics.append({'rmse': info['rmse'], 'r2': info['r2']})
        all_losses.append(info['losses'])
        # save centers evolution for first fold
        if fold_idx == 0:
            saved_centers = info['centers_evolution']

    # aggregate metrics
    rmses = [m['rmse'] for m in fold_metrics]
    r2s = [m['r2'] for m in fold_metrics]
    summary = {
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'r2_mean': float(np.mean(r2s)),
        'r2_std': float(np.std(r2s)),
    }
    print('CV summary:', summary)
    pd.DataFrame(fold_metrics).to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)

    plot_losses(all_losses, out_dir)
    if saved_centers is not None:
        plot_centers_evolution(saved_centers, out_dir)

    # save summary
    pd.Series(summary).to_csv(os.path.join(out_dir, 'summary.csv'))


if __name__ == '__main__':
    main()
