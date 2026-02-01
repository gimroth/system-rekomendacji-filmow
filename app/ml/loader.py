"""Central ANFIS model loader - XANFIS STABLE VERSION"""
import os
import logging
from pathlib import Path
from app.ml.anfis_xanfis import XANFISWrapper

_model = None
_raw_model = None

def load_model(models_dir=None):
    global _model, _raw_model
    
    base = models_dir or os.path.join(os.path.dirname(__file__), 'models')
    pkl_path = os.path.join(base, 'xanfis_latest.pkl')

    if os.path.exists(pkl_path):
        try:
            # Używamy stabilnej metody ładowania wag do nowego obiektu
            print(f"Loading XANFIS from weights: {pkl_path}...")
            wrapper = XANFISWrapper.load_stable(pkl_path)
            
            _model = wrapper
            _raw_model = wrapper.model
            print("✅ XANFIS loaded successfully (Stable StateDict mode)")
            return _model
        except Exception as e:
            print(f"❌ Failed loading XANFIS: {e}")
            # Jeśli błąd to 'CustomANFIS', spróbuj usunąć stary plik
            if 'CustomANFIS' in str(e):
                print("Detected corrupt legacy pickle. Please run train_anfis_xanfis.py again.")

    _model = None
    _raw_model = None
    return None

def get_model():
    if _model is None:
        load_model()
    return _model

def get_raw_model():
    return _raw_model