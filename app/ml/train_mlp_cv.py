import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='app/ml/datasets/anfis_synthetic.csv')
    parser.add_argument('--n-splits', type=int, default=3)
    parser.add_argument('--hidden', type=str, default='64,32,16')
    args = parser.parse_args()

    X, y = load_dataset(args.dataset)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    hidden = tuple(int(x) for x in args.hidden.split(',')) if args.hidden else (64,32,16)

    rmses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_val_s = scaler_x.transform(X_val)

        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                           alpha=0.001, batch_size=32, learning_rate_init=1e-3,
                           max_iter=500, early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=20, random_state=42)

        mlp.fit(X_train_s, y_train_s)

        y_val_pred_s = mlp.predict(X_val_s)
        y_val_pred = scaler_y.inverse_transform(y_val_pred_s.reshape(-1,1)).ravel()

        # sklearn versions differ; compute RMSE as sqrt of MSE
        rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
        rmses.append(rmse)
        print(f'Fold {fold_idx+1} RMSE: {rmse:.6f}')

    print('MLP CV RMSE mean:', np.mean(rmses), 'std:', np.std(rmses))


if __name__ == '__main__':
    main()
