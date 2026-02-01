"""Wrapper for x-anfis (gradient-based ANFIS).

This module provides a lightweight `XANFISWrapper` that tries to import
common x-anfis classes and exposes `fit`, `predict`, `save` methods
with a small adapter layer so `train_anfis_xanfis.py` can use it.

The x-anfis API surface can vary; this wrapper attempts to call
`fit`/`train`/`fit_epoch` if present. If not present, it will raise a
helpful error.
"""

import os
import pickle
import numpy as np
from typing import Optional

import importlib
import pkgutil

# Try to find a suitable x-anfis/anfis package by scanning installed modules.
_model_candidates = None
_candidates_tried = []

def _find_anfis_module():
    names = ['x_anfis', 'xanfis', 'xanfis_torch', 'xanfis_pytorch', 'x-anfis', 'anfis', 'anfis_torch']
    for n in names:
        try:
            mod = importlib.import_module(n)
            return mod
        except Exception:
            pass
    return None

# KLUCZOWA LINIA: To tutaj definiujemy zmienną, której brakuje
_xanfis_module = _find_anfis_module()


class XANFISWrapper:
    # Dodajemy n_mfs: int = 3 do listy parametrów
    def __init__(self, n_inputs: int = 3, n_mfs: int = 3, device: str = 'cpu'):
        if _xanfis_module is None:
            raise ImportError('x-anfis library not found.')
        
        self.device = device
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs  # Zapamiętujemy n_mfs
        self.model = None

        # Szukanie klasy regresora
        cls = None
        for name in ('GdAnfisRegressor', 'AnfisRegressor', 'GdAnfis', 'Anfis'):
            cls = getattr(_xanfis_module, name, None)
            if cls is not None: break
        
        if cls is None:
            raise ImportError('Could not find ANFIS class in package.')

        # Przekazujemy n_mfs do instancji modelu biblioteki
        try:
            self.model = cls(n_inputs=self.n_inputs, n_mfs=self.n_mfs)
        except TypeError:
            # Jeśli biblioteka używa innej nazwy, np. n_rules
            try:
                self.model = cls(n_inputs=self.n_inputs, n_rules=self.n_mfs)
            except:
                self.model = cls(self.n_inputs)

    def fit(self, X, y, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64, verbose: bool = True):
        """Fit model. Accepts pandas DataFrame / numpy arrays.

        Returns history dict {"loss": [...]} if available.
        """
        # convert to numpy
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()

        history = {"loss": []}

        # Try common fit signatures
        if hasattr(self.model, 'fit'):
            try:
                res = self.model.fit(X_np, y_np, epochs=epochs, lr=lr, batch_size=batch_size)
                # if fit returns history, attempt to extract
                if isinstance(res, dict) and 'loss' in res:
                    history = res
                return history
            except TypeError:
                # try signature without named args
                try:
                    res = self.model.fit(X_np, y_np, epochs)
                    if isinstance(res, dict) and 'loss' in res:
                        history = res
                    return history
                except Exception:
                    pass
        # try 'train' method
        if hasattr(self.model, 'train'):
            try:
                res = self.model.train(X_np, y_np, epochs=epochs, lr=lr)
                if isinstance(res, dict) and 'loss' in res:
                    history = res
                return history
            except Exception:
                pass

        raise RuntimeError('x-anfis model does not expose a compatible fit/train method. Check x-anfis API.')

    def predict(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if hasattr(self.model, 'predict'):
            return self.model.predict(X_np)
        if hasattr(self.model, 'forward'):
            return self.model.forward(X_np)
        # fallback: try calling model(X)
        try:
            return np.asarray(self.model(X_np))
        except Exception as e:
            raise RuntimeError('Could not call model.predict; inspect the model API') from e

    def save_stable(self, path: str, top_3: list):
        """Zapisuje wagi i metadane zamiast całego obiektu."""
        import joblib
        state = {
            'model_state': self.model.network.state_dict(),
            'n_inputs': self.n_inputs,
            'top_3': top_3
        }
        joblib.dump(state, path)

    @staticmethod
    def load_stable(path: str):
        """Odbudowuje model z wag."""
        import joblib
        data = joblib.load(path)
        wrapper = XANFISWrapper(n_inputs=data['n_inputs'])
        
        # Inicjalizacja sieci przed wczytaniem wag (dummy fit)
        dummy_x = np.zeros((2, data['n_inputs']), dtype=np.float32)
        dummy_y = np.zeros(2, dtype=np.float32)
        wrapper.fit(dummy_x, dummy_y, epochs=0)
        
        wrapper.model.network.load_state_dict(data['model_state'])
        wrapper.model.network.eval()
        return wrapper
