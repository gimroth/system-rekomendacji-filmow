import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle

sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.data_processor import DataProcessor

print("=" * 80)
print("TRENOWANIE MLP BASELINE DO PORÓWNANIA Z ANFIS")
print("=" * 80)

print("\nKROK 1: Inicjalizacja połączenia z bazą danych")
print("-" * 80)

db = SessionLocal()
processor = DataProcessor(db)

print("Połączono z bazą danych")

print("\nKROK 2: Ekstrakcja features z bazy danych")
print("-" * 80)

print("Pobieranie danych z tabeli ratings...")
df = processor.get_training_data()

if df.empty:
    print("BŁĄD: Brak danych w tabeli ratings!")
    print("Dodaj oceny filmów przez system przed trenowaniem.")
    db.close()
    sys.exit(1)

print(f"Pobrano {len(df)} rekordów z bazy danych")
print(f"\nKolumny: {list(df.columns)}")

print(f"\nStatystyki danych:")
print(f"   - Unikalnych użytkowników: {df['user_id'].nunique()}")
print(f"   - Unikalnych filmów: {df['movie_id'].nunique()}")
print(f"   - Średni rating (znormalizowany): {df['target'].mean():.3f}")

print("\nKROK 3: Sampling danych (tak samo jak ANFIS)")
print("-" * 80)

MAX_SAMPLES = 5000

print(f"\nDataset zawiera {len(df)} próbek")
if len(df) > MAX_SAMPLES:
    print(f"   Ograniczam do {MAX_SAMPLES} próbek (dla spójności z ANFIS)")
    df = df.sample(n=MAX_SAMPLES, random_state=42)
    print(f"Wybrano losowo {len(df)} próbek")
else:
    print(f"Dataset OK - {len(df)} próbek")

print("\nKROK 4: Przygotowanie danych treningowych/testowych")
print("-" * 80)

feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
X = df[feature_cols].copy()
y = df['target']

print(f"Features (X): {list(X.columns)}")
print(f"Target (y): rating (znormalizowany 0-1)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nPodział danych:")
print(f"   - Train: {len(X_train)} próbek ({len(X_train) / len(X) * 100:.1f}%)")
print(f"   - Test:  {len(X_test)} próbek ({len(X_test) / len(X) * 100:.1f}%)")

print("\nKROK 5: Normalizacja features (StandardScaler)")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features znormalizowane (mean=0, std=1)")

print("\nKROK 6: Budowa modelu MLP")
print("-" * 80)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)

print(f"Model MLP zbudowany:")
print(f"   - Architektura: {mlp.hidden_layer_sizes}")
print(f"   - Activation: {mlp.activation}")
print(f"   - Solver: {mlp.solver}")
print(f"   - Max epochs: {mlp.max_iter}")

print("\nTrenowanie modelu MLP...")
mlp.fit(X_train_scaled, y_train)
print(f"Trenowanie zakończone po {mlp.n_iter_} epochs!")

print("\nKROK 7: Ewaluacja modelu na zbiorze testowym")
print("-" * 80)

y_pred = mlp.predict(X_test_scaled)

y_pred = np.clip(y_pred, 0, 1)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nMetryki (znormalizowane 0-1):")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE:  {mae:.4f}")

rmse_denorm = rmse * 4
mae_denorm = mae * 4

print(f"\nMetryki (skala 1-5):")
print(f"   RMSE: {rmse_denorm:.4f} gwiazdek")
print(f"   MAE:  {mae_denorm:.4f} gwiazdek")

print(f"\nInterpretacja:")
if rmse < 0.15:
    print("   DOSKONAŁY wynik! (RMSE < 0.15)")
    quality = "EXCELLENT"
elif rmse < 0.25:
    print("   DOBRY wynik! (RMSE < 0.25)")
    quality = "GOOD"
elif rmse < 0.35:
    print("   ŚREDNI wynik (RMSE < 0.35)")
    quality = "AVERAGE"
else:
    print("   SŁABY wynik (RMSE > 0.35)")
    quality = "POOR"

print("\nKROK 8: Generowanie wizualizacji")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_mlp_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=30)
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Idealna predykcja')
plt.xlabel('Prawdziwe wartości (0-1)', fontsize=11)
plt.ylabel('Predykcje (0-1)', fontsize=11)
plt.title(f'MLP: Predykcje vs Rzeczywiste\nRMSE={rmse:.4f}', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

errors = y_pred - y_test
plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
plt.xlabel('Błąd predykcji', fontsize=11)
plt.ylabel('Liczba próbek', fontsize=11)
plt.title(f'Rozkład błędów\nMAE={mae:.4f}', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
plt.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2,
            label=f'Średni błąd: {errors.mean():.4f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = os.path.join(results_dir, 'mlp_evaluation.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Wykres zapisany: {plot_path}")
plt.close()

plt.figure(figsize=(8, 6))
plt.boxplot([errors], labels=['MLP'], widths=0.5)
plt.ylabel('Błąd predykcji', fontsize=12)
plt.title('Rozkład błędów MLP', fontsize=14)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3, axis='y')
plot_path = os.path.join(results_dir, 'mlp_errors_boxplot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Wykres zapisany: {plot_path}")
plt.close()

if hasattr(mlp, 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_curve_, label='Training Loss', linewidth=2)
    if hasattr(mlp, 'validation_scores_'):
        val_loss = [1 - score for score in mlp.validation_scores_]
        plt.plot(val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('MLP: Krzywa uczenia', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(results_dir, 'mlp_loss_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Wykres zapisany: {plot_path}")
    plt.close()

print("\nKROK 9: Zapis wytrenowanego modelu")
print("-" * 80)

models_dir = 'app/ml/models'
os.makedirs(models_dir, exist_ok=True)

model_filename = f'mlp_v1_{timestamp}.pkl'
model_path = os.path.join(models_dir, model_filename)

with open(model_path, 'wb') as f:
    pickle.dump({'model': mlp, 'scaler': scaler}, f)

latest_path = os.path.join(models_dir, 'mlp_latest.pkl')
with open(latest_path, 'wb') as f:
    pickle.dump({'model': mlp, 'scaler': scaler}, f)

print(f"Model zapisany:")
print(f"   - Wersjonowany: {model_path}")
print(f"   - Latest: {latest_path}")

print("\nKROK 10: Generowanie raportu")
print("-" * 80)

report = f"""
================================================================================
RAPORT TRENOWANIA MLP BASELINE
================================================================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DANE:
-----
- Liczba rekordów: {len(df)}
- Unikalnych użytkowników: {df['user_id'].nunique()}
- Unikalnych filmów: {df['movie_id'].nunique()}
- Średni rating: {df['target'].mean():.3f} (znormalizowany 0-1)

CECHY:
------
Wszystkie 5 aspektów: {', '.join(feature_cols)}

PODZIAŁ DANYCH:
---------------
- Train: {len(X_train)} próbek ({len(X_train) / len(X) * 100:.1f}%)
- Test:  {len(X_test)} próbek ({len(X_test) / len(X) * 100:.1f}%)

MODEL MLP:
----------
- Architektura: {mlp.hidden_layer_sizes}
- Activation: {mlp.activation}
- Solver: {mlp.solver}
- Epochs: {mlp.n_iter_} (max: {mlp.max_iter})
- Early stopping: {mlp.early_stopping}

WYNIKI:
-------
Metryki (znormalizowane 0-1):
  - RMSE: {rmse:.4f}
  - MAE:  {mae:.4f}

Metryki (skala 1-5):
  - RMSE: {rmse_denorm:.4f} gwiazdek
  - MAE:  {mae_denorm:.4f} gwiazdek

Jakość modelu: {quality}

Interpretacja błędu MAE:
  Średnio model myli się o {mae_denorm:.2f} gwiazdki.
  Przykład: Jeśli prawdziwy rating to 4.0, 
            model przewiduje średnio {4.0 - mae_denorm:.2f} - {4.0 + mae_denorm:.2f}

PLIKI WYGENEROWANE:
-------------------
- Model: {model_path}
- Wykresy: {results_dir}/
- Raport: {os.path.join(results_dir, 'raport_techniczny.txt')}

================================================================================
"""

report_path = os.path.join(results_dir, 'raport_techniczny.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Raport zapisany: {report_path}")

print(report)

print("\nKROK 11: Test predykcji na przykładzie")
print("-" * 80)

sample_idx = np.random.randint(0, len(X_test))
sample_features = X_test.iloc[sample_idx].values.reshape(1, -1)
sample_scaled = scaler.transform(sample_features)
true_rating = y_test.iloc[sample_idx]

predicted_rating = mlp.predict(sample_scaled)[0]
predicted_rating = np.clip(predicted_rating, 0, 1)

predicted_denorm = (predicted_rating * 4) + 1
true_denorm = (true_rating * 4) + 1

print(f"Przykład #{sample_idx}:")
print(f"\nCechy:")
for i, col in enumerate(feature_cols):
    value = sample_features[0][i]
    denorm_value = (value * 4) + 1
    print(f"   {col}: {value:.3f} (skala 1-5: {denorm_value:.2f})")

print(f"\nPrawdziwy rating: {true_rating:.3f} (skala 1-5: {true_denorm:.2f})")
print(f"Predykcja MLP:    {predicted_rating:.3f} (skala 1-5: {predicted_denorm:.2f})")
print(f"Błąd: {abs(predicted_rating - true_rating):.3f} (skala 1-5: {abs(predicted_denorm - true_denorm):.2f})")

print("\n" + "=" * 80)
print("TRENOWANIE MLP ZAKOŃCZONE POMYŚLNIE!")
print("=" * 80)

print(f"""
Pliki wygenerowane:
   - Model (wersjonowany): {model_path}
   - Model (latest): {latest_path}
   - Raport: {report_path}
   - Wykresy: {results_dir}/

Następne kroki:
   1. Sprawdź wykresy w folderze {results_dir}/
   2. Przeczytaj raport: {report_path}
   3. Porównaj MLP z ANFIS (skrypt compare_models.py)

Użycie modelu w API:
   import pickle
   with open('app/ml/models/mlp_latest.pkl', 'rb') as f:
       data = pickle.load(f)
   mlp = data['model']
   scaler = data['scaler']

   features_scaled = scaler.transform(features)
   prediction = mlp.predict(features_scaled)
""")

db.close()
print("\nPołączenie z bazą zamknięte")
print("=" * 80)