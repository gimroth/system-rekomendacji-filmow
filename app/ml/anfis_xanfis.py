import os
import joblib
import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt

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
            for name in ('AnfisRegressor', 'GdAnfis', 'Anfis'):
                cls = getattr(_xanfis_module, name, None)
                if cls is not None: break
        
        if cls is None:
            raise ImportError('Nie znaleziono odpowiedniej klasy ANFIS.')

        # Bezpieczna inicjalizacja
        self.model = cls()
        self.model.n_inputs = self.n_inputs
        self.model.n_mfs = self.n_mfs
        self.model.size_output = 1
        
        # Wymuszenie budowy modelu
        if hasattr(self.model, 'build_model'):
            try:
                self.model.build_model()
            except:
                pass

    def fit(self, X, y, epochs: int = 150, lr: float = 1e-4, batch_size: int = 32):
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        
        print(f"🔧 DEBUG WRAPPER: Wymuszanie {epochs} epok na obiekcie modelu...")

        # --- FIX: FORSOWNE USTAWIANIE EPOK ---
        if hasattr(self.model, 'epochs'): self.model.errors = epochs
        self.model.epochs = epochs
        self.model.n_epochs = epochs
        self.model.max_epochs = epochs
        self.model.train_iters = epochs
        
        # --- FIX BŁĘDU: Zabezpieczenie przed brakiem optymalizatora ---
        # Sprawdzamy czy optimizer w ogóle istnieje (is not None)
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            try:
                for param_group in self.model.optimizer.param_groups:
                    param_group['lr'] = lr
            except:
                pass # Jeśli struktura optimizera jest inna, ignorujemy

        fit_result = None
        if hasattr(self.model, 'fit'):
            # PRÓBA 1: Standardowa
            try:
                fit_result = self.model.fit(X_np, y_np, epochs=epochs, lr=lr, batch_size=batch_size)
            except TypeError:
                # PRÓBA 2: Parametr n_epochs
                try:
                    fit_result = self.model.fit(X_np, y_np, n_epochs=epochs, lr=lr, batch_size=batch_size)
                except TypeError:
                    # PRÓBA 3: Argumenty pozycyjne
                    try:
                        fit_result = self.model.fit(X_np, y_np, epochs)
                    except:
                        # Ostatnia deska ratunku
                        fit_result = self.model.fit(X_np, y_np)

        # --- EKSTRAKCJA HISTORII BŁĘDÓW ---
        history = {'loss': []}

        # 1. Sprawdź atrybut .errors
        if hasattr(self.model, 'errors') and self.model.errors:
            raw_errors = self.model.errors
            # Spłaszczanie listy jeśli trzeba
            if isinstance(raw_errors, list):
                if len(raw_errors) > 0 and isinstance(raw_errors[0], list):
                     history['loss'] = [e[0] for e in raw_errors]
                else:
                    history['loss'] = raw_errors
        
        # 2. Sprawdź słownik zwrotny
        elif isinstance(fit_result, dict) and 'loss' in fit_result:
            history = fit_result
            
        # 3. Sprawdź listę zwrotną
        elif isinstance(fit_result, list):
            history['loss'] = fit_result
        
        # Obcinanie historii do żądanej liczby epok
        if len(history['loss']) > epochs:
            history['loss'] = history['loss'][:epochs]

        return history

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32) if not isinstance(X, dict) else \
               np.array([[X.get(f'match{i+1}', 0.5) for i in range(self.n_inputs)]], dtype=np.float32)
        preds = self.model.predict(X_np)
        return np.asarray(preds).flatten()

    def plot_mfs(self, save_path=None):
        if self.model and hasattr(self.model, 'network'):
            try:
                plt.figure(figsize=(10, 6))
                if hasattr(self.model.network, 'plot_mfs'):
                    self.model.network.plot_mfs()
                    if save_path:
                        plt.savefig(save_path)
                        plt.close()
            except:
                pass

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