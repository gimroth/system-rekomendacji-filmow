"""Central ANFIS model loader used by the application.

Tries to load PyTorch model `anfis_pytorch_latest.pt` first, then falls back
to legacy pickle `anfis_latest.pkl`.
"""
import os
import pickle
import numpy as np
try:
    import torch
    from app.ml.anfis_pytorch import PyTorchANFIS
except Exception:
    torch = None

_model = None
_raw_model = None


def load_model(models_dir=None):
    """Load model into module-level variable `_model`.

    models_dir: optional base path to `app/ml/models`.
    """
    global _model, _raw_model
    base = models_dir or os.path.join(os.path.dirname(__file__), 'models')
    pt_path = os.path.join(base, 'anfis_pytorch_latest.pt')
    pkl_path = os.path.join(base, 'anfis_latest.pkl')

    # Try PyTorch
    if torch is not None and os.path.exists(pt_path):
        try:
            m = PyTorchANFIS(n_inputs=3, n_mfs=3, consequent_order=0)
            state = torch.load(pt_path, map_location='cpu')
            m.load_state_dict(state)

            class PTWrapper:
                def __init__(self, model):
                    self.model = model

                def predict(self, features: dict) -> float:
                    keys = [f'match{i+1}' for i in range(self.model.n_inputs)]
                    vals = [features.get(k, 0.5) for k in keys]
                    arr = np.array(vals, dtype=np.float32).reshape(1, -1)
                    return float(self.model.predict_numpy(arr)[0])

            _raw_model = m
            _model = PTWrapper(m)
            print(f"Loaded PyTorch ANFIS from {pt_path}")
            return _model
        except Exception as e:
            print(f"Failed loading PyTorch ANFIS: {e}")

    # Fallback pickle
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                _model = pickle.load(f)
                # also expose raw pickle model for explainability
                _raw_model = _model
            print(f"Loaded legacy ANFIS pickle from {pkl_path}")
            return _model
        except Exception as e:
            print(f"Failed loading pickle ANFIS: {e}")

    print("No ANFIS model found in", base)
    _model = None
    _raw_model = None
    return None


def get_model():
    return _model


def get_raw_model():
    """Return the underlying raw model object (PyTorchANFIS or ANFISRecommender)
    Useful for explainability/debug endpoints.
    """
    return _raw_model
