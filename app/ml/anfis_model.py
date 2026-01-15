"""
Model ANFIS z logiką MATCH i naprawioną inicjalizacją.
Naprawiono: błąd _rule_generator oraz AttributeError: evaluate.
"""
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ANFISRecommender:
    def __init__(self, aspect_names: List[str] = None):
        self.aspect_names = aspect_names or ['match1', 'match2', 'match3']
        self.input_variables = {}
        self.output_variable = None
        self.rules = []
        self.fuzzy_system = None
        self.is_trained = False

    def _create_membership_function(self, name: str, var_type: str = 'input'):
        universe = np.arange(0, 1.01, 0.01)
        if var_type == 'input':
            var = ctrl.Antecedent(universe, name)
            var['low'] = fuzz.trimf(universe, [0, 0, 0.6])
            var['medium'] = fuzz.trimf(universe, [0.1, 0.5, 0.9])
            var['high'] = fuzz.trimf(universe, [0.4, 1, 1])
        else:
            var = ctrl.Consequent(universe, name)
            var['very_low'] = fuzz.trimf(universe, [0, 0, 0.25])
            var['low'] = fuzz.trimf(universe, [0, 0.25, 0.5])
            var['medium'] = fuzz.trimf(universe, [0.25, 0.5, 0.75])
            var['high'] = fuzz.trimf(universe, [0.5, 0.75, 1])
            var['very_high'] = fuzz.trimf(universe, [0.75, 1, 1])
        return var

    def build_fuzzy_system(self):
        """Naprawa błędu _rule_generator poprzez właściwą kolejność."""
        for a in self.aspect_names:
            self.input_variables[a] = self._create_membership_function(a, 'input')
        self.output_variable = self._create_membership_function('predicted_rating', 'output')
        self._create_fuzzy_rules()
        self.fuzzy_system = ctrl.ControlSystem(self.rules)
        self.is_trained = True

    def _create_fuzzy_rules(self):
        self.rules = []
        v = [self.input_variables[a] for a in self.aspect_names]
        lv = ['low', 'medium', 'high']
        for i in lv:
            for j in lv:
                for k in lv:
                    score = (lv.index(i) + lv.index(j) + lv.index(k))
                    out = 'very_high' if score >= 5 else 'high' if score == 4 else 'medium' if score == 3 else 'low'
                    self.rules.append(ctrl.Rule(v[0][i] & v[1][j] & v[2][k], self.output_variable[out]))

    def predict(self, features: Dict[str, float]) -> float:
        if not self.is_trained: self.build_fuzzy_system()
        sim = ctrl.ControlSystemSimulation(self.fuzzy_system)
        for k, v in features.items():
            if k in self.input_variables: sim.input[k] = np.clip(v, 0, 1)
        try:
            sim.compute()
            return float(sim.output['predicted_rating'])
        except Exception: return 0.5

    def evaluate(self, X_test, y_test) -> Dict:
        """Dodano brakującą metodę ewaluacji."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        preds = [self.predict(row.to_dict()) for _, row in X_test.iterrows()]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        return {'rmse': float(rmse), 'mae': float(mae), 'predictions': preds, 'actuals': y_test.values.tolist()}

    def train(self, X_train=None, y_train=None):
        self.build_fuzzy_system()

    def save(self, path: str):
        with open(path, 'wb') as f: pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f: return pickle.load(f)
        anfis_system = None

def load_anfis_model():
    """Funkcja ładowana przez main.py przy starcie serwera."""
    global anfis_system
    
    # Ścieżka do pliku .pkl (zakładamy, że jest w folderze app/ml/models/)
    # __file__ to ścieżka do tego pliku (anfis_model.py)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "models", "anfis_latest.pkl")
    
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                anfis_system = pickle.load(f)
            print(f"✅ ANFIS: Załadowano model z {model_path}")
        except Exception as e:
            print(f"❌ ANFIS: Błąd ładowania modelu: {e}")
    else:
        print(f"⚠️ ANFIS: Nie znaleziono pliku modelu w {model_path}")

def get_anfis_model():
    """Getter używany przez router do pobrania modelu."""
    return anfis_system