import os
import joblib
import numpy as np
import torch
import importlib

def _find_anfis_module():
    names = ['x_anfis', 'xanfis', 'xanfis_torch', 'anfis']
    for n in names:
        try: return importlib.import_module(n)
        except: pass
    return None

_xanfis_module = _find_anfis_module()

class XANFISWrapper:
    def __init__(self, n_inputs: int = 3, n_mfs: int = 4, device: str = 'cpu'):
        if _xanfis_module is None: 
            raise ImportError('Biblioteka xanfis nie została znaleziona.')
        
        self.device, self.n_inputs, self.n_mfs = device, n_inputs, n_mfs
        
        cls = getattr(_xanfis_module, 'GdAnfisRegressor', None)
        if cls is None:
            # Fallback do innych nazw klas jeśli GdAnfisRegressor nie istnieje
            for name in ('AnfisRegressor', 'GdAnfis', 'Anfis'):
                cls = getattr(_xanfis_module, name, None)
                if cls is not None: break
        
        if cls is None:
            raise ImportError('Nie znaleziono odpowiedniej klasy ANFIS w pakiecie.')

        # POPRAWKA: Jawne przekazanie parametrów, aby uniknąć błędu z typem funkcji przynależności
        # Biblioteka xanfis często używa 'n_mfs' jako liczby funkcji, ale pozycyjnie oczekuje MF jako klasy.
        try:
            self.model = cls(datadim=self.n_inputs, n_mfs=self.n_mfs)
        except TypeError:
            # Próba bez słów kluczowych, ale z zachowaniem kolejności oczekiwanej przez xanfis
            try:
                self.model = cls(n_inputs=self.n_inputs, n_mfs=self.n_mfs)
            except Exception:
                # Ostateczny fallback - domyślne parametry
                self.model = cls()

    def fit(self, X, y, epochs: int = 100, lr: float = 1e-4, batch_size: int = 32):
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()
        if hasattr(self.model, 'fit'):
            return self.model.fit(X_np, y_np, epochs=epochs, lr=lr, batch_size=batch_size)
        return {"loss": []}

    def predict(self, X):
        if isinstance(X, dict):
            X_np = np.array([[X.get(f'match{i+1}', 0.5) for i in range(self.n_inputs)]], dtype=np.float32)
        else:
            X_np = X.values.astype(np.float32) if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        
        preds = self.model.predict(X_np)
        return np.asarray(preds).flatten()

    def save_stable(self, path: str, top_3: list):
        """Zapisuje wagi i konfigurację modelu."""
        state = {
            'model_state': self.model.network.state_dict(),
            'n_inputs': self.n_inputs,
            'n_mfs': self.n_mfs,
            'top_3': top_3
        }
        joblib.dump(state, path)

    @staticmethod
    def load_stable(path: str):
        """Odtwarza model z zapisanego stanu wag."""
        data = joblib.load(path)
        wrapper = XANFISWrapper(n_inputs=data['n_inputs'], n_mfs=data.get('n_mfs', 4))
        # Inicjalizacja sieci przed wstrzyknięciem wag
        wrapper.fit(np.zeros((2, data['n_inputs'])), np.zeros(2), epochs=0)
        wrapper.model.network.load_state_dict(data['model_state'])
        wrapper.model.network.eval()
        return wrapper