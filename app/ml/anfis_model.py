"""
ANFIS Model dla Systemu Rekomendacji Filmów
Struktura: 3 cechy wejściowe (dynamiczne) + 27 reguł fuzzy
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pickle
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ANFISRecommender:
    """
    ANFIS (Adaptive Neuro-Fuzzy Inference System) dla rekomendacji filmów.

    Struktura zgodna z dokumentacją:
    - 3 cechy wejściowe dynamiczne (wybrane przez DataProcessor)
    - Każda cecha ma 3 funkcje przynależności: LOW, MEDIUM, HIGH
    - 27 reguł fuzzy (3^3 = 27 kombinacji)
    - Output: predicted_rating w zakresie [0, 1] (później denormalizacja do 1-5)
    """

    def __init__(self, aspect_names: List[str] = None):
        """
        Args:
            aspect_names: Lista 3 aspektów, np. ['story', 'acting', 'visuals']
                         Jeśli None, użyje domyślnych
        """
        self.aspect_names = aspect_names or ['story', 'acting', 'visuals']
        if len(self.aspect_names) != 3:
            raise ValueError(f"ANFIS wymaga dokładnie 3 aspektów, otrzymano: {len(self.aspect_names)}")

        self.input_variables = {}
        self.output_variable = None
        self.rules = []
        self.fuzzy_system = None
        self.is_trained = False

        logger.info(f"Inicjalizacja ANFIS z aspektami: {self.aspect_names}")

    def _create_membership_function(self, variable_name: str, var_type: str = 'input'):
        """
        Tworzy funkcje przynależności (trójkątne) dla zmiennej.

        Args:
            variable_name: Nazwa zmiennej, np. 'u_story', 'm_story', 'output'
            var_type: 'input' lub 'output'

        Returns:
            Antecedent lub Consequent z funkcjami przynależności
        """
        universe = np.arange(0, 1.01, 0.01)  # Zakres [0, 1]

        if var_type == 'input':
            variable = ctrl.Antecedent(universe, variable_name)

            # 3 funkcje przynależności: LOW, MEDIUM, HIGH (trójkątne)
            variable['low'] = fuzz.trimf(variable.universe, [0, 0, 0.5])
            variable['medium'] = fuzz.trimf(variable.universe, [0, 0.5, 1])
            variable['high'] = fuzz.trimf(variable.universe, [0.5, 1, 1])

        else:  # output
            variable = ctrl.Consequent(universe, variable_name)

            # 5 funkcji przynależności dla outputu (bardziej granularne)
            variable['very_low'] = fuzz.trimf(variable.universe, [0, 0, 0.25])
            variable['low'] = fuzz.trimf(variable.universe, [0, 0.25, 0.5])
            variable['medium'] = fuzz.trimf(variable.universe, [0.25, 0.5, 0.75])
            variable['high'] = fuzz.trimf(variable.universe, [0.5, 0.75, 1])
            variable['very_high'] = fuzz.trimf(variable.universe, [0.75, 1, 1])

        return variable

    def build_fuzzy_system(self):
        """
        Buduje system fuzzy zgodnie ze strukturą z dokumentacji:
        - 6 zmiennych wejściowych: u_X1, u_X2, u_X3, m_X1, m_X2, m_X3
        - 1 zmienna wyjściowa: predicted_rating
        - 27 reguł fuzzy
        """
        logger.info(" Budowanie systemu ANFIS...")

        # INPUT VARIABLES (6 zmiennych: 3 user + 3 movie)
        for aspect in self.aspect_names:
            self.input_variables[f'u_{aspect}'] = self._create_membership_function(f'u_{aspect}', 'input')
            self.input_variables[f'm_{aspect}'] = self._create_membership_function(f'm_{aspect}', 'input')

        # OUTPUT VARIABLE
        self.output_variable = self._create_membership_function('predicted_rating', 'output')

        # REGUŁY FUZZY (27 reguł zgodnie z tabelą w dokumentacji)
        self._create_fuzzy_rules()

        # Stwórz system kontrolny
        self.fuzzy_system = ctrl.ControlSystem(self.rules)

        logger.info(f" System ANFIS zbudowany:")
        logger.info(f"   - 6 zmiennych wejściowych: {list(self.input_variables.keys())}")
        logger.info(f"   - {len(self.rules)} reguł fuzzy")

    def _create_fuzzy_rules(self):
        """
        Tworzy 27 reguł fuzzy zgodnie z tabelą w dokumentacji.

        Struktura:
        IF u_X1 is [LOW/MEDIUM/HIGH] AND u_X2 is [LOW/MEDIUM/HIGH] AND u_X3 is [LOW/MEDIUM/HIGH]
        AND m_X1 is [LOW/MEDIUM/HIGH] AND m_X2 is [LOW/MEDIUM/HIGH] AND m_X3 is [LOW/MEDIUM/HIGH]
        THEN predicted_rating is [VERY_LOW/LOW/MEDIUM/HIGH/VERY_HIGH]

        Uproszczenie: Skupiamy się na user features (zgodnie z dokumentacją Nadii)
        """
        self.rules = []

        # Pobierz nazwy zmiennych (dynamiczne)
        u_vars = [self.input_variables[f'u_{aspect}'] for aspect in self.aspect_names]
        m_vars = [self.input_variables[f'm_{aspect}'] for aspect in self.aspect_names]

        # Mapowanie kombinacji na output (zgodnie z tabelą z dokumentacji)
        rules_mapping = [
            # (u1, u2, u3, output) - uproszczona wersja
            # Reguła 1: Wszystko HIGH → VERY_HIGH
            ('high', 'high', 'high', 'very_high'),
            ('high', 'high', 'medium', 'high'),
            ('high', 'high', 'low', 'high'),

            ('high', 'medium', 'high', 'high'),
            ('high', 'medium', 'medium', 'high'),
            ('high', 'medium', 'low', 'medium'),

            ('high', 'low', 'high', 'medium'),
            ('high', 'low', 'medium', 'medium'),
            ('high', 'low', 'low', 'low'),

            ('medium', 'high', 'high', 'high'),
            ('medium', 'high', 'medium', 'medium'),
            ('medium', 'high', 'low', 'medium'),

            ('medium', 'medium', 'high', 'medium'),
            ('medium', 'medium', 'medium', 'medium'),
            ('medium', 'medium', 'low', 'low'),

            ('medium', 'low', 'high', 'medium'),
            ('medium', 'low', 'medium', 'low'),
            ('medium', 'low', 'low', 'low'),

            ('low', 'high', 'high', 'medium'),
            ('low', 'high', 'medium', 'low'),
            ('low', 'high', 'low', 'low'),

            ('low', 'medium', 'high', 'low'),
            ('low', 'medium', 'medium', 'low'),
            ('low', 'medium', 'low', 'very_low'),

            ('low', 'low', 'high', 'low'),
            ('low', 'low', 'medium', 'very_low'),
            ('low', 'low', 'low', 'very_low'),
        ]

        # Tworzenie reguł - uwzględniamy też movie features
        for u1_level, u2_level, u3_level, output_level in rules_mapping:
            # Prosta reguła: jeśli user i movie są na podobnym poziomie → wysoki rating
            rule = ctrl.Rule(
                u_vars[0][u1_level] & u_vars[1][u2_level] & u_vars[2][u3_level] &
                m_vars[0][u1_level] & m_vars[1][u2_level] & m_vars[2][u3_level],
                self.output_variable[output_level]
            )
            self.rules.append(rule)

        # Dodatkowe reguły wzmacniające (boost gdy user i movie match)
        # Przykład: IF u_story HIGH AND m_story HIGH THEN boost
        for i, aspect in enumerate(self.aspect_names):
            # High match
            self.rules.append(ctrl.Rule(
                u_vars[i]['high'] & m_vars[i]['high'],
                self.output_variable['very_high']
            ))
            # Low mismatch
            self.rules.append(ctrl.Rule(
                u_vars[i]['low'] & m_vars[i]['low'],
                self.output_variable['very_low']
            ))

        logger.info(f" Utworzono {len(self.rules)} reguł fuzzy")

    def train(self, X_train=None, y_train=None):
        """
        "Trenowanie" ANFIS.

        W uproszczonej wersji ANFIS (bez backpropagation) po prostu
        budujemy system fuzzy. Pełna implementacja ANFIS wymaga
        optymalizacji parametrów funkcji przynależności.

        Args:
            X_train: DataFrame z cechami treningowymi (opcjonalne)
            y_train: Series z targetem (opcjonalne)
        """
        logger.info("🎓 Trenowanie ANFIS...")

        if self.fuzzy_system is None:
            self.build_fuzzy_system()

        # W pełnej implementacji tutaj byłaby optymalizacja parametrów
        # Na razie używamy statycznych reguł z dokumentacji

        self.is_trained = True
        logger.info(" Trenowanie zakończone (reguły statyczne)")

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predykcja ratingu dla danego zestawu cech.

        Args:
            features: Dict z cechami, np.:
                {
                    'u_story': 0.75, 'u_acting': 0.5, 'u_visuals': 1.0,
                    'm_story': 0.8, 'm_acting': 0.6, 'm_visuals': 0.9
                }

        Returns:
            Predicted rating w zakresie [0, 1] (ZNORMALIZOWANY!)
        """
        if not self.is_trained:
            raise ValueError("Model nie jest wytrenowany! Wywołaj train() najpierw.")

        # Walidacja inputów
        required_keys = [f'u_{a}' for a in self.aspect_names] + [f'm_{a}' for a in self.aspect_names]
        missing = set(required_keys) - set(features.keys())
        if missing:
            raise ValueError(f"Brakujące cechy: {missing}")

        # Stwórz simulator
        simulator = ctrl.ControlSystemSimulation(self.fuzzy_system)

        # Ustaw input values
        for key, value in features.items():
            if key in self.input_variables:
                # Clip do zakresu [0, 1]
                simulator.input[key] = np.clip(value, 0, 1)

        # Compute output
        try:
            simulator.compute()
            return float(simulator.output['predicted_rating'])
        except Exception as e:
            logger.warning(f"Błąd podczas compute: {e}. Zwracam domyślną wartość 0.5")
            return 0.5

    def predict_denormalized(self, features: Dict[str, float]) -> float:
        """
        Predykcja z automatyczną denormalizacją do skali 1-5.

        Returns:
            Rating w zakresie [1, 5]
        """
        normalized_rating = self.predict(features)
        denormalized = (normalized_rating * 4) + 1  # 0-1 → 1-5
        return np.clip(denormalized, 1.0, 5.0)

    def evaluate(self, X_test, y_test) -> Dict:
        """
        Ewaluacja modelu na zbiorze testowym.

        Args:
            X_test: DataFrame z cechami testowymi
            y_test: Series z prawdziwymi ratingami (ZNORMALIZOWANYMI 0-1)

        Returns:
            Dict z metrykami: {'rmse': ..., 'mae': ..., 'predictions': [...]}
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        logger.info(" Ewaluacja modelu...")

        predictions = []
        for idx, row in X_test.iterrows():
            features = row.to_dict()
            try:
                pred = self.predict(features)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Błąd predykcji dla row {idx}: {e}")
                predictions.append(0.5)  # Domyślna wartość

        predictions = np.array(predictions)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   MAE:  {mae:.4f}")

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'predictions': predictions.tolist(),
            'actuals': y_test.values.tolist()
        }

    def save(self, filepath: str):
        """Zapisz model do pliku pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f" Model zapisany: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ANFISRecommender':
        """Wczytaj model z pliku pickle."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f" Model wczytany: {filepath}")
        return model

    def get_model_info(self) -> Dict:
        """Zwraca informacje o modelu."""
        return {
            'aspect_names': self.aspect_names,
            'num_input_vars': len(self.input_variables),
            'num_rules': len(self.rules),
            'is_trained': self.is_trained
        }


if __name__ == "__main__":
    # Test modelu
    print("=== Test ANFIS Model ===")

    # Inicjalizacja
    anfis = ANFISRecommender(aspect_names=['story', 'acting', 'visuals'])
    anfis.train()

    # Test prediction
    test_features = {
        'u_story': 0.75,  # User lubi story (waga 4/5)
        'u_acting': 0.5,  # User średnio lubi acting (waga 3/5)
        'u_visuals': 1.0,  # User kocha visuals (waga 5/5)
        'm_story': 0.8,  # Film ma świetny story
        'm_acting': 0.6,  # Film ma ok acting
        'm_visuals': 0.9  # Film ma świetne visuals
    }

    prediction = anfis.predict(test_features)
    prediction_denorm = anfis.predict_denormalized(test_features)

    print(f"\nPredykcja (znormalizowana 0-1): {prediction:.3f}")
    print(f"Predykcja (skala 1-5): {prediction_denorm:.2f}")

    print(f"\nInfo o modelu: {anfis.get_model_info()}")