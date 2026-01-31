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
    # common names to try directly
    names = ['x_anfis', 'xanfis', 'xanfis_torch', 'xanfis_pytorch', 'x-anfis', 'anfis', 'anfis_torch']
    for n in names:
        try:
            mod = importlib.import_module(n)
            return mod
        except Exception:
            _candidates_tried.append(n)

    # fallback: scan installed modules for any name containing 'anfis' or 'xanfis'
    for finder, name, ispkg in pkgutil.iter_modules():
        if 'anfis' in name.lower() or 'xanfis' in name.lower():
            try:
                mod = importlib.import_module(name)
                return mod
            except Exception:
                _candidates_tried.append(name)
    return None

_model_candidates = _find_anfis_module()


class XANFISWrapper:
    def __init__(self, n_inputs: int = 3, hidden_rules: Optional[int] = None, device: str = 'cpu'):
        if _model_candidates is None:
            raise ImportError('x-anfis library not found. Install x-anfis in the environment.')

        # prefer gradient-based regressor if available
        self.device = device
        self.n_inputs = n_inputs
        self.model = None

        # common class names from x-anfis
        cls = None
        for name in ('GdAnfisRegressor', 'GdAnfis', 'GdAnfisModel'):
            cls = getattr(_model_candidates, name, None)
            if cls is not None:
                break
        if cls is None:
            # try classic regressor class name
            for name in ('AnfisRegressor', 'Anfis'):
                cls = getattr(_model_candidates, name, None)
                if cls is not None:
                    break
        if cls is None:
            raise ImportError('Could not find ANFIS regressor class in x-anfis package (checked common names).')

        # instantiate with sensible defaults; the constructor signature can vary
        try:
            # many implementations accept n_inputs or input_features
            self.model = cls(n_inputs=self.n_inputs)
        except Exception:
            try:
                self.model = cls(self.n_inputs)
            except Exception:
                # last resort: call without args
                self.model = cls()

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

    def save(self, path: str):
        # prefer model-specific save if available
        if hasattr(self.model, 'save'):
            try:
                self.model.save(path)
                return
            except Exception:
                pass
        # fallback to pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path: str):
        # try pickle load
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        wrapper = object.__new__(XANFISWrapper)
        wrapper.model = obj
        return wrapper
