"""
ROZBUDOWANY SKRYPT TRENOWANIA ANFIS Z DYNAMICZNYMI TOP 3
Logika: match1, match2, match3 (Dopasowanie dynamiczne)
Zapisuje model do .pkl i generuje raporty wizualne.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_model import ANFISRecommender

print("=" * 80)
print("🎓 TRENOWANIE ANFIS: DYNAMICZNE TOP 3 + LOGIKA DOPASOWANIA (MATCH)")
print("=" * 80)

# =============================================================================
# KROK 1: Inicjalizacja
# =============================================================================
print("\n📊 KROK 1: Inicjalizacja")
print("-" * 80)

db = SessionLocal()
processor = DataProcessor(db)
print("✅ Połączono z bazą")

# =============================================================================
# KROK 2: Ekstrakcja danych
# =============================================================================
print("\n📊 KROK 2: Ekstrakcja features")
print("-" * 80)

df = processor.get_training_data()

if df.empty:
    print("❌ Brak danych!")
    db.close()
    sys.exit(1)

print(f"✅ Pobrano {len(df)} rekordów")
print(f"   - Users: {df['user_id'].nunique()}")
print(f"   - Movies: {df['movie_id'].nunique()}")

# =============================================================================
# KROK 3: Sampling
# =============================================================================
print("\n📊 KROK 3: Sampling")
print("-" * 80)

MAX_SAMPLES = 5000
if len(df) > MAX_SAMPLES:
    print(f"⚡ Ograniczam do {MAX_SAMPLES} próbek")
    df = df.sample(n=MAX_SAMPLES, random_state=42)
    print(f"✅ Wybrano {len(df)} próbek")

# =============================================================================
# KROK 4: DYNAMICZNE TOP 3 - PRZYGOTOWANIE KLUCZY MATCH
# =============================================================================
print("\n📊 KROK 4: Przygotowanie danych (LOGIKA MATCH)")
print("-" * 80)

feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)

print("📈 Korelacja cech filmu z ratingiem:")
for feature, corr in correlations.items():
    print(f"   {feature}: {corr:.4f}")

# Wybór Top 3 cech na podstawie korelacji
top_3_features = correlations.head(3).index.tolist()
top_3_names = [f.replace('m_', '') for f in top_3_features]
print(f"\n💡 Najbardziej skorelowane: {top_3_names}")
print(f"   (Model będzie używał GENERYCZNYCH kluczy: match1, match2, match3)")

# Budowa zestawu danych treningowych (Logika Match)
X = pd.DataFrame()
for i, feat in enumerate(top_3_features):
    # W treningu używamy cech filmu (normalizacja 0-1) jako wejść matchX
    X[f'match{i+1}'] = df[feat]

y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Split: {len(X_train)} train, {len(X_test)} test")

# =============================================================================
# KROK 5: Budowa i Trening ANFIS
# =============================================================================
print("\n📊 KROK 5: Budowa ANFIS")
print("-" * 80)

# Inicjalizacja modelu z generycznymi nazwami
anfis = ANFISRecommender(aspect_names=['match1', 'match2', 'match3'])
anfis.build_fuzzy_system()

print(f"✅ ANFIS zbudowany pomyślnie:")
print(f"   - Zmiennych: {len(anfis.input_variables)}")
print(f"   - Reguł: {len(anfis.rules)}")

# Wywołanie metody train() zsynchronizowanej z anfis_model.py
print("\n🎓 Trenowanie (Inicjalizacja logiki)...")
anfis.train(X_train, y_train)
print("✅ Trenowanie zakończone!")

# =============================================================================
# KROK 6: Ewaluacja
# =============================================================================
print("\n📊 KROK 6: Ewaluacja")
print("-" * 80)

metrics = anfis.evaluate(X_test, y_test)

rmse_denorm = metrics['rmse'] * 4
mae_denorm = metrics['mae'] * 4

# Oblicz średni błąd (Mean Error)
preds = np.array(metrics['predictions'])
actuals = np.array(metrics['actuals'])
errors = preds - actuals
mean_error = errors.mean()
mean_error_denorm = mean_error * 4

print(f"\n📈 Metryki (Skala 0-1):")
print(f"   RMSE: {metrics['rmse']:.4f}")
print(f"   MAE:  {metrics['mae']:.4f}")

print(f"\n⭐ Metryki (Skala 1-5):")
print(f"   RMSE: {rmse_denorm:.4f}⭐")
print(f"   MAE:  {mae_denorm:.4f}⭐")
print(f"   Średni błąd: {mean_error_denorm:.4f}⭐")

# Określenie jakości modelu
if metrics['rmse'] < 0.15: quality = "EXCELLENT"
elif metrics['rmse'] < 0.25: quality = "GOOD"
else: quality = "AVERAGE"

# =============================================================================
# KROK 7: Wizualizacje (Zgodnie z image_8d2d43.png)
# =============================================================================
print("\n📊 KROK 7: Wizualizacje")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_anfis_match_{timestamp}"
os.makedirs(results_dir, exist_ok=True)



plt.figure(figsize=(15, 6))

# Lewy wykres: Predykcje vs Rzeczywiste
plt.subplot(1, 2, 1)
plt.scatter(actuals, preds, alpha=0.6, edgecolors='black', s=40)
plt.plot([0, 1], [0, 1], 'r--', linewidth=3, label='Idealna predykcja')
plt.xlabel('Prawdziwe (0-1)', fontsize=12)
plt.ylabel('Predykcje (0-1)', fontsize=12)
plt.title(f'ANFIS Match: Predykcje vs Rzeczywiste\nRMSE={metrics["rmse"]:.4f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Prawy wykres: Rozkład błędów
plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, color='lightblue', edgecolor='black', alpha=0.8)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
plt.axvline(x=mean_error, color='orange', linestyle='-', linewidth=2, label=f'Średni błąd: {mean_error:.4f}')
plt.xlabel('Błąd', fontsize=12)
plt.ylabel('Liczba', fontsize=12)
plt.title(f'Rozkład błędów\nMAE={metrics["mae"]:.4f}, Średni błąd={mean_error:.4f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.2)

plt.tight_layout()
plot_path = os.path.join(results_dir, 'anfis_performance.png')
plt.savefig(plot_path, dpi=150)
print(f"✅ Wykres zapisany: {plot_path}")

# =============================================================================
# KROK 8: Zapis modelu (.pkl)
# =============================================================================
print("\n📊 KROK 8: Zapis modelu")
print("-" * 80)

models_dir = 'app/ml/models'
os.makedirs(models_dir, exist_ok=True)

# Zapis pliku z datą oraz pliku "latest" dla routera
model_filename = f'anfis_match_{timestamp}.pkl'
model_path = os.path.join(models_dir, model_filename)
anfis.save(model_path)

latest_path = os.path.join(models_dir, 'anfis_latest.pkl')
anfis.save(latest_path)

print(f"✅ Model zapisany pomyślnie:")
print(f"   - {model_path}")
print(f"   - {latest_path}")

# =============================================================================
# KROK 9: Raport (Twój format)
# =============================================================================
print("\n📊 KROK 9: Raport Końcowy")
print("-" * 80)

report = f"""
================================================================================
RAPORT: ANFIS Z DYNAMICZNYMI TOP 3 (LOGIKA MATCH)
================================================================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DANE:
-----
- Rekordów: {len(df)}
- Users: {df['user_id'].nunique()}
- Movies: {df['movie_id'].nunique()}

MODEL:
------
- Typ: ANFIS DYNAMICZNY (MATCH)
- Klucze wejściowe: match1, match2, match3
- Dla tego datasetu aspect1={top_3_names[0]}, aspect2={top_3_names[1]}, aspect3={top_3_names[2]}
- Zmiennych: {len(anfis.input_variables)}
- Reguł: {len(anfis.rules)}

WYNIKI:
-------
RMSE: {metrics['rmse']:.4f} (0-1) = {rmse_denorm:.4f}⭐ (1-5)
MAE:  {metrics['mae']:.4f} (0-1) = {mae_denorm:.4f}⭐ (1-5)
Średni błąd: {mean_error:.4f} (0-1) = {mean_error_denorm:.4f}⭐ (1-5)
Jakość: {quality}

Interpretacja:
- MAE (Mean Absolute Error): Model myli się średnio o {mae_denorm:.2f} gwiazdki
- Średni błąd: {mean_error_denorm:+.2f} gwiazdki (+ = przewiduje za wysoko, - = za nisko)
- Przykład: Prawdziwy rating 4.0⭐ → Model przewiduje {4.0 + mean_error_denorm:.2f}⭐

WAŻNE:
------
Model przyjmuje GENERYCZNE klucze (match1, match2, match3).
Endpoint może teraz używać DOWOLNYCH 3 aspektów dla każdego użytkownika!

Przykład użycia (Logika MATCH):
  match_score = 1.0 - abs(user_preference - movie_attribute)
  anfis_input = {{'match1': match_score1, 'match2': match_score2, 'match3': match_score3}}

================================================================================
"""

report_path = os.path.join(results_dir, 'training_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✅ Raport zapisany: {report_path}")
print(report)

db.close()
print("\n" + "=" * 80)
print("🚀 PROCES ZAKOŃCZONY POMYŚLNIE!")
print("=" * 80)