"""
ANFISRecommender (lightweight hybrid training)

This implementation keeps skfuzzy-based membership functions and rules
but implements a simple hybrid learning for consequents:
- compute rule firing strengths for each sample
- normalize them (h_ni = firing_ni / sum_i firing_ni)
- solve for consequent constants via ridge regression: y = H @ w

It also provides evaluation metrics (RMSE, MAE, R2) and plotting
utilities for membership functions.

Note: this is a pragmatic, small implementation to get regression-style
ANFIS behaviour (zero-order Sugeno consequents). For full XANFIS/PyTorch
gradient-based ANFIS consider porting to the x-anfis library.
"""

import os
from typing import Dict, List

import logging
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ANFISRecommender:
    def __init__(self, aspect_names: List[str] = None, universe=None):
        self.aspect_names = aspect_names or ['match1', 'match2', 'match3']
        self.universe = universe if universe is not None else np.arange(0, 1.01, 0.01)
        self.input_variables = {}  # name -> Antecedent
        self.output_variable = None
        self.rules = []
        self.rule_labels = []  # list of tuples like ('low','medium','high') per rule
        self.n_rules = 0
        self.is_built = False

        # Learned consequents (one scalar per rule for zero-order Sugeno)
        self.consequents = None

    def _create_membership_function(self, name: str, var_type: str = 'input'):
        if var_type == 'input':
            var = ctrl.Antecedent(self.universe, name)
            # Gaussian membership functions: centers at 0.0, 0.5, 1.0
            # sigma chosen to give reasonable overlap on [0,1]
            var['low'] = fuzz.gaussmf(self.universe, 0.0, 0.18)
            var['medium'] = fuzz.gaussmf(self.universe, 0.5, 0.18)
            var['high'] = fuzz.gaussmf(self.universe, 1.0, 0.18)
        else:
            var = ctrl.Consequent(self.universe, name)
            # Gaussian MFs for output visualization (very_low..very_high)
            var['very_low'] = fuzz.gaussmf(self.universe, 0.0, 0.12)
            var['low'] = fuzz.gaussmf(self.universe, 0.25, 0.12)
            var['medium'] = fuzz.gaussmf(self.universe, 0.5, 0.12)
            var['high'] = fuzz.gaussmf(self.universe, 0.75, 0.12)
            var['very_high'] = fuzz.gaussmf(self.universe, 1.0, 0.12)
        return var

    def build_fuzzy_system(self):
        """Create Antecedents, Consequent (for visualization) and generate rules."""
        self.input_variables = {}
        for a in self.aspect_names:
            self.input_variables[a] = self._create_membership_function(a, 'input')
        self.output_variable = self._create_membership_function('predicted_rating', 'output')
        self._create_fuzzy_rules()
        self.fuzzy_system = ctrl.ControlSystem(self.rules) if self.rules else None
        self.is_built = True

    def _create_fuzzy_rules(self):
        self.rules = []
        self.rule_labels = []
        v = [self.input_variables[a] for a in self.aspect_names]
        lv = ['low', 'medium', 'high']
        for i in lv:
            for j in lv:
                for k in lv:
                    score = (lv.index(i) + lv.index(j) + lv.index(k))
                    out = 'very_high' if score >= 5 else 'high' if score == 4 else 'medium' if score == 3 else 'low'
                    # keep skfuzzy control rule for compatibility/visualization
                    self.rules.append(ctrl.Rule(v[0][i] & v[1][j] & v[2][k], self.output_variable[out]))
                    self.rule_labels.append((i, j, k))
        self.n_rules = len(self.rules)

    def _sample_membership(self, var_name: str, term: str, value: float) -> float:
        """Return membership degree for a single variable-term and scalar value."""
        var = self.input_variables[var_name]
        term_obj = var[term]
        # get numeric mf array (Antecedent/Consequent term has attribute .mf)
        if hasattr(term_obj, 'mf'):
            mf_vals = np.asarray(term_obj.mf)
        else:
            mf_vals = np.asarray(term_obj)
        # mf_vals should be same length as universe
        return float(fuzz.interp_membership(self.universe, mf_vals, np.clip(value, self.universe[0], self.universe[-1])))

    def _compute_firing_matrix(self, X):
        """Compute firing strengths matrix F (n_samples x n_rules).
        X can be a pandas DataFrame or an iterable of dicts.
        """
        import pandas as pd

        if not self.is_built:
            self.build_fuzzy_system()

        if isinstance(X, pd.DataFrame):
            X_iter = (row.to_dict() for _, row in X.iterrows())
            n_samples = len(X)
        else:
            X_list = list(X)
            n_samples = len(X_list)
            X_iter = iter(X_list)

        F = np.zeros((n_samples, self.n_rules), dtype=float)
        for n, sample in enumerate(X_iter):
            for r, labels in enumerate(self.rule_labels):
                # product of membership degrees
                mvals = [self._sample_membership(self.aspect_names[idx], labels[idx], sample.get(self.aspect_names[idx], 0.0))
                         for idx in range(len(self.aspect_names))]
                F[n, r] = np.prod(mvals)

        return F

    def train(self, X_train=None, y_train=None, ridge_alpha: float = 1e-3):
        """Hybrid training that fits zero-order Sugeno consequents via ridge regression.

        After this call `self.consequents` will be a numpy array of shape (n_rules,).
        """
        if X_train is None or y_train is None:
            raise ValueError('Provide X_train and y_train for training')

        # ensure system built
        if not self.is_built:
            self.build_fuzzy_system()

        F = self._compute_firing_matrix(X_train)  # n x m
        # normalize per-sample to get H
        row_sums = F.sum(axis=1, keepdims=True)
        # avoid division by zero
        row_sums[row_sums == 0] = 1e-8
        H = F / row_sums

        # Fit ridge regression (no intercept): H @ w = y
        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        model.fit(H, np.array(y_train).ravel())
        self.consequents = model.coef_.astype(float)

        # compute training metrics
        preds = (H @ self.consequents)
        rmse = float(np.sqrt(mean_squared_error(y_train, preds)))
        mae = float(mean_absolute_error(y_train, preds))
        r2 = float(r2_score(y_train, preds))
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def predict(self, features: Dict[str, float]) -> float:
        """Predict for a single sample (dict of feature_name -> value).

        Falls back to fuzzy.ControlSystemSimulation if consequents are not trained.
        """
        if not self.is_built:
            self.build_fuzzy_system()

        # if consequents learned, use hybrid prediction
        if self.consequents is not None:
            # compute firing vector
            vals = [features.get(a, 0.0) for a in self.aspect_names]
            F = np.zeros((self.n_rules,), dtype=float)
            for r, labels in enumerate(self.rule_labels):
                mvals = [self._sample_membership(self.aspect_names[idx], labels[idx], vals[idx])
                         for idx in range(len(self.aspect_names))]
                F[r] = np.prod(mvals)
            s = F.sum()
            if s == 0:
                return float(np.mean(self.consequents))
            H = F / s
            return float(H.dot(self.consequents))

        # fallback: use control system simulation
        try:
            sim = ctrl.ControlSystemSimulation(self.fuzzy_system)
            for k, v in features.items():
                if k in self.input_variables:
                    sim.input[k] = np.clip(v, 0, 1)
            sim.compute()
            return float(sim.output['predicted_rating'])
        except Exception:
            return 0.5

    def evaluate(self, X_test, y_test) -> Dict:
        import pandas as pd

        if isinstance(X_test, pd.DataFrame):
            X_iter = (row.to_dict() for _, row in X_test.iterrows())
        else:
            X_iter = iter(X_test)

        preds = [self.predict(row) for row in X_iter]
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': preds, 'actuals': list(y_test)}

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ------------------ Visualization helpers ------------------
    def plot_membership_functions(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for name, var in self.input_variables.items():
            plt.figure(figsize=(6, 3))
            for term in var.terms:
                # compute membership values over the universe using _sample_membership
                mf_vals = [self._sample_membership(name, term, float(x)) for x in self.universe]
                plt.plot(self.universe, mf_vals, label=term)
            plt.title(f'Membership functions - {name}')
            plt.xlabel('Value')
            plt.ylabel('Membership')
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            path = os.path.join(save_dir, f'mf_{name}.png')
            plt.savefig(path, dpi=150)
            plt.close()


def load_anfis_model():
    """Helper used by `app.main` to preload model if available."""
    global anfis_system
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "models", "anfis_latest.pkl")
    anfis_system = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                anfis_system = pickle.load(f)
            logger.info('Loaded ANFIS model from %s', model_path)
        except Exception as e:
            logger.exception('Failed loading ANFIS model: %s', e)
    return anfis_system


def get_anfis_model():
    return globals().get('anfis_system', None)
