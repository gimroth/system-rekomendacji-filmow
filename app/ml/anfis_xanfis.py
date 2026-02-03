import os
import joblib
import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
import inspect
import sys
from io import StringIO


class LossCapture:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.original_stdout

    def write(self, text):
        self.original_stdout.write(text)
        if 'Train Loss:' in text and 'Validation Loss:' in text:
            parts = text.split(',')
            for part in parts:
                if 'Train Loss:' in part:
                    try:
                        self.train_losses.append(float(part.split(':')[1].strip()))
                    except:
                        pass
                if 'Validation Loss:' in part:
                    try:
                        self.val_losses.append(float(part.split(':')[1].strip()))
                    except:
                        pass

    def flush(self):
        self.original_stdout.flush()


def _find_anfis_module():
    names = ['x_anfis', 'xanfis', 'xanfis_torch', 'anfis']
    for n in names:
        try:
            return importlib.import_module(n)
        except:
            pass
    return None


_xanfis_module = _find_anfis_module()


class XANFISWrapper:
    def __init__(self, n_inputs: int = 3, n_mfs: int = 4, device: str = 'cpu'):
        if _xanfis_module is None:
            raise ImportError('Biblioteka xanfis nie została znaleziona.')

        self.device, self.n_inputs, self.n_mfs = device, n_inputs, n_mfs
        cls = getattr(_xanfis_module, 'GdAnfisRegressor', None)
        if cls is None:
            for name in ('AnfisRegressor', 'GdAnfis', 'Anfis'):
                cls = getattr(_xanfis_module, name, None)
                if cls is not None:
                    break

        if cls is None:
            raise ImportError('Nie znaleziono odpowiedniej klasy ANFIS.')

        self.model = cls()
        self.model.n_inputs = self.n_inputs
        self.model.n_mfs = self.n_mfs
        self.model.size_output = 1

    def fit(self, X, y, epochs: int = 150, lr: float = 1e-4, batch_size: int = 32):
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        print(f"Training with: epochs={epochs}, lr={lr}, batch_size={batch_size}")

        if hasattr(self.model, 'n_epochs'):
            self.model.n_epochs = epochs
        elif hasattr(self.model, 'max_epochs'):
            self.model.max_epochs = epochs
        elif hasattr(self.model, 'epochs'):
            self.model.epochs = epochs

        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            try:
                for param_group in self.model.optimizer.param_groups:
                    param_group['lr'] = lr
            except Exception as e:
                print(f"Warning: Could not set learning rate: {e}")

        fit_result = None

        sig = inspect.signature(self.model.fit)
        params = sig.parameters

        loss_capture = LossCapture()

        with loss_capture:
            try:
                if 'epochs' in params:
                    fit_result = self.model.fit(X_np, y_np, epochs=epochs, lr=lr, batch_size=batch_size)
                elif 'n_epochs' in params:
                    fit_result = self.model.fit(X_np, y_np, n_epochs=epochs, lr=lr, batch_size=batch_size)
                else:
                    fit_result = self.model.fit(X_np, y_np)
            except Exception as e:
                print(f"Warning during fit: {e}")
                fit_result = self.model.fit(X_np, y_np)

        history = {
            'train_loss': loss_capture.train_losses,
            'val_loss': loss_capture.val_losses
        }

        if len(history['train_loss']) == 0:
            old_history = self._extract_loss_history(fit_result, epochs)
            if old_history['loss']:
                history['train_loss'] = old_history['loss']

        return history

    def _extract_loss_history(self, fit_result, max_epochs):
        if hasattr(self.model, 'errors') and self.model.errors:
            errors = self.model.errors
            if isinstance(errors, list) and errors:
                loss_list = [e[0] if isinstance(e, list) else e for e in errors]
                return {'loss': loss_list[:max_epochs]}

        if isinstance(fit_result, dict):
            return {'loss': fit_result.get('loss', [])[:max_epochs]}
        elif isinstance(fit_result, list):
            return {'loss': fit_result[:max_epochs]}

        return {'loss': []}

    def predict(self, X):
        if isinstance(X, dict):
            X_np = np.array([[X.get(f'match{i + 1}', 0.5) for i in range(self.n_inputs)]], dtype=np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        if X_np.shape[1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} features, got {X_np.shape[1]}")

        if np.isnan(X_np).any():
            raise ValueError("Input contains NaN values")

        preds = self.model.predict(X_np)
        return np.asarray(preds).flatten()

    def plot_mfs(self, save_path=None):
        if self.model and hasattr(self.model, 'network') and self.model.network is not None:
            try:
                if hasattr(self.model.network, 'plot_mfs'):
                    plt.figure(figsize=(10, 6))
                    self.model.network.plot_mfs()
                    if save_path:
                        plt.savefig(save_path)
                        plt.close()
                        print(f"Saved MF plot: {save_path}")
                    return

                print(f"Attempting manual MF plotting...")
                self._manual_plot_mfs(save_path)

            except Exception as e:
                print(f"Warning: Could not plot MFs: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Model network not built yet. Train the model first before plotting MFs.")

    def _manual_plot_mfs(self, save_path=None):
        try:
            import torch

            if hasattr(self.model.network, 'memberships'):
                memberships_list = self.model.network.memberships
            else:
                print(f"ERROR: Could not find memberships")
                return

            x = torch.linspace(0, 1, 200).unsqueeze(1)

            fig, axes = plt.subplots(1, self.n_inputs, figsize=(15, 4))
            if self.n_inputs == 1:
                axes = [axes]

            for i in range(self.n_inputs):
                ax = axes[i]

                if i < len(memberships_list):
                    mf_module = memberships_list[i]

                    mu_vals = mf_module(x).detach().numpy()

                    print(f"Feature {i + 1}: MF output shape {mu_vals.shape}")

                    labels = ['Low', 'Medium', 'High']
                    for j in range(min(mu_vals.shape[1], self.n_mfs)):
                        ax.plot(x.numpy().flatten(), mu_vals[:, j], linewidth=2.5,
                                label=labels[j] if j < len(labels) else f'MF{j + 1}')

                ax.set_xlabel(f'Input {i + 1}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Membership Degree', fontsize=11)
                ax.set_title(f'Feature {i + 1}', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.05, 1.1)

            plt.suptitle('Membership Functions (Gaussian)', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved MF plot: {save_path}")
            else:
                plt.show()

        except Exception as e:
            print(f"Warning: Manual MF plotting failed: {e}")
            import traceback
            traceback.print_exc()

    def save_stable(self, path: str, top_3: list):
        state = {
            'model_state': self.model.network.state_dict(),
            'n_inputs': self.n_inputs,
            'n_mfs': self.n_mfs,
            'top_3': top_3
        }
        joblib.dump(state, path)

    @staticmethod
    def load_stable(path: str):
        data = joblib.load(path)
        wrapper = XANFISWrapper(n_inputs=data['n_inputs'], n_mfs=data.get('n_mfs', 4))
        wrapper.fit(np.zeros((2, data['n_inputs'])), np.zeros(2), epochs=1)
        wrapper.model.network.load_state_dict(data['model_state'])
        wrapper.model.network.eval()
        return wrapper