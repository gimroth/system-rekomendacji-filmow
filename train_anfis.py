"""
Skrypt trenowania ANFIS z DYNAMICZNYMI TOP 3
Model przyjmuje aspect1, aspect2, aspect3 (bez nazw konkretnych aspektów)

ZMIANA: Klucze są teraz GENERYCZNE (u_aspect1, m_aspect1, ...)
        zamiast KONKRETNYCH (u_story, m_story, ...)
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
print("🎓 TRENOWANIE ANFIS Z DYNAMICZNYMI TOP 3")
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
# KROK 4: DYNAMICZNE TOP 3 - KLUCZOWA ZMIANA!
# =============================================================================
print("\n📊 KROK 4: Przygotowanie danych (DYNAMICZNE TOP 3)")
print("-" * 80)

# Analiza korelacji (tylko info)
feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)

print("📈 Korelacja cech:")
for feature, corr in correlations.items():
    print(f"   {feature}: {corr:.4f}")

# Top 3 (dla informacji)
top_3_features = correlations.head(3).index.tolist()
top_3_names = [f.replace('m_', '') for f in top_3_features]
print(f"\n💡 Najbardziej skorelowane: {top_3_names}")
print(f"   (Model będzie używał GENERYCZNYCH kluczy: aspect1, aspect2, aspect3)")

# ⚠️ KLUCZOWA ZMIANA: Używamy GENERYCZNYCH nazw!
X = df[top_3_features].copy()

# PRZED (konkretne nazwy):
# X.columns = ['u_story', 'u_acting', 'u_direction']

# PO (generyczne nazwy):
X.columns = ['u_aspect1', 'u_aspect2', 'u_aspect3']  # ✅ DYNAMICZNE!

# Dodaj movie features (też generyczne)
for i in range(3):
    X[f'm_aspect{i+1}'] = df[top_3_features[i]]

y = df['target']

print(f"\n✅ Features (X): {list(X.columns)}")
print("   Znaczenie:")
print(f"   - aspect1 = {top_3_names[0]}")
print(f"   - aspect2 = {top_3_names[1]}")
print(f"   - aspect3 = {top_3_names[2]}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Split: {len(X_train)} train, {len(X_test)} test")

# =============================================================================
# KROK 5: Budowa ANFIS
# =============================================================================
print("\n📊 KROK 5: Budowa ANFIS")
print("-" * 80)

# ANFIS z GENERYCZNYMI nazwami
anfis = ANFISRecommender(aspect_names=['aspect1', 'aspect2', 'aspect3'])
anfis.build_fuzzy_system()

print(f"✅ ANFIS zbudowany:")
print(f"   - Zmiennych: {len(anfis.input_variables)}")
print(f"   - Reguł: {len(anfis.rules)}")

print("\n🎓 Trenowanie...")
anfis.train(X_train, y_train)
print("✅ Trenowanie zakończone!")

# =============================================================================
# KROK 6: Ewaluacja
# =============================================================================
print("\n📊 KROK 6: Ewaluacja")
print("-" * 80)

metrics = anfis.evaluate(X_test, y_test)

print(f"\n📈 Metryki (0-1):")
print(f"   RMSE: {metrics['rmse']:.4f}")
print(f"   MAE:  {metrics['mae']:.4f}")

rmse_denorm = metrics['rmse'] * 4
mae_denorm = metrics['mae'] * 4

# Oblicz średni błąd
errors = np.array(metrics['predictions']) - np.array(metrics['actuals'])
mean_error = errors.mean()
mean_error_denorm = mean_error * 4

print(f"\n📈 Metryki (1-5):")
print(f"   RMSE: {rmse_denorm:.4f}⭐")
print(f"   MAE:  {mae_denorm:.4f}⭐")
print(f"   Średni błąd: {mean_error_denorm:.4f}⭐")

if metrics['rmse'] < 0.15:
    print("\n✅ DOSKONAŁY wynik!")
    quality = "EXCELLENT"
elif metrics['rmse'] < 0.25:
    print("\n✅ DOBRY wynik!")
    quality = "GOOD"
else:
    print("\n⚠️  ŚREDNI wynik")
    quality = "AVERAGE"

# =============================================================================
# KROK 7: Wizualizacje
# =============================================================================
print("\n📊 KROK 7: Wizualizacje")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_anfis_dynamic_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Wykres
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(metrics['actuals'], metrics['predictions'], alpha=0.5, edgecolors='k', s=30)
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Idealna predykcja')
plt.xlabel('Prawdziwe (0-1)', fontsize=11)
plt.ylabel('Predykcje (0-1)', fontsize=11)
plt.title(f'ANFIS Dynamic: Predykcje vs Rzeczywiste\nRMSE={metrics["rmse"]:.4f}', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

# Wykres 2: Rozkład błędów
errors = np.array(metrics['predictions']) - np.array(metrics['actuals'])
mean_error = errors.mean()  # Średni błąd
abs_mean_error = np.abs(errors).mean()  # Średni błąd bezwzględny

plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Błąd', fontsize=11)
plt.ylabel('Liczba', fontsize=11)
plt.title(f'Rozkład błędów\nMAE={metrics["mae"]:.4f}, Średni błąd={mean_error:.4f}', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
plt.axvline(x=mean_error, color='orange', linestyle='-', linewidth=2,
            label=f'Średni błąd: {mean_error:.4f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = os.path.join(results_dir, 'anfis_dynamic_evaluation.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Wykres: {plot_path}")
plt.close()

# =============================================================================
# KROK 8: Zapis modelu
# =============================================================================
print("\n📊 KROK 8: Zapis modelu")
print("-" * 80)

models_dir = 'app/ml/models'
os.makedirs(models_dir, exist_ok=True)

model_filename = f'anfis_dynamic_{timestamp}.pkl'
model_path = os.path.join(models_dir, model_filename)
anfis.save(model_path)

latest_path = os.path.join(models_dir, 'anfis_latest.pkl')
anfis.save(latest_path)

print(f"✅ Model zapisany:")
print(f"   - {model_path}")
print(f"   - {latest_path}")

# =============================================================================
# KROK 9: Raport
# =============================================================================
print("\n📊 KROK 9: Raport")
print("-" * 80)

report = f"""
================================================================================
RAPORT: ANFIS Z DYNAMICZNYMI TOP 3
================================================================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DANE:
-----
- Rekordów: {len(df)}
- Users: {df['user_id'].nunique()}
- Movies: {df['movie_id'].nunique()}

MODEL:
------
- Typ: ANFIS DYNAMICZNY
- Klucze wejściowe: u_aspect1, m_aspect1, u_aspect2, m_aspect2, u_aspect3, m_aspect3
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
Model przyjmuje GENERYCZNE klucze (aspect1, aspect2, aspect3).
Endpoint może teraz używać DOWOLNYCH 3 aspektów dla każdego użytkownika!

Przykład użycia:
  User 1 (lubi story, acting, direction):
    anfis_input = {{'u_aspect1': story, 'm_aspect1': story, 
                    'u_aspect2': acting, 'm_aspect2': acting,
                    'u_aspect3': direction, 'm_aspect3': direction}}
  
  User 2 (lubi visuals, sound, acting):
    anfis_input = {{'u_aspect1': visuals, 'm_aspect1': visuals,
                    'u_aspect2': sound, 'm_aspect2': sound,
                    'u_aspect3': acting, 'm_aspect3': acting}}

================================================================================
"""

report_path = os.path.join(results_dir, 'training_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✅ Raport: {report_path}")
print(report)

# =============================================================================
# Test
# =============================================================================
print("\n📊 KROK 10: Test predykcji")
print("-" * 80)

sample_idx = 0
sample_features = X_test.iloc[sample_idx].to_dict()
true_rating = y_test.iloc[sample_idx]

predicted_rating = anfis.predict(sample_features)
predicted_denorm = (predicted_rating * 4) + 1
true_denorm = (true_rating * 4) + 1

print(f"Przykład:")
print(f"  Input: {sample_features}")
print(f"  Prawdziwy: {true_denorm:.2f}⭐")
print(f"  Predykcja: {predicted_denorm:.2f}⭐")
print(f"  Błąd: {abs(predicted_denorm - true_denorm):.2f}⭐")

# =============================================================================
# Zakończenie
# =============================================================================
print("\n" + "=" * 80)
print("✅ GOTOWE!")
print("=" * 80)

print(f"""
📁 Pliki:
   - Model: {latest_path}
   - Raport: {report_path}
   - Wykresy: {results_dir}/

🎯 Model jest teraz DYNAMICZNY!
   Endpoint może używać DOWOLNYCH top 3 aspektów dla każdego użytkownika.
""")

db.close()