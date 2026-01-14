"""
Skrypt trenowania ANFIS na prawdziwych danych z bazy
Używa DataProcessor od Nadii do ekstrakcji features
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime


sys.path.insert(0, os.path.abspath('.'))

from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_model import ANFISRecommender

print("=" * 80)
print("TRENOWANIE ANFIS NA PRAWDZIWYCH DANYCH Z BAZY")
print("=" * 80)

# =============================================================================
# KROK 1: Inicjalizacja
# =============================================================================
print("\nKROK 1: Inicjalizacja połączenia z bazą danych")
print("-" * 80)

db = SessionLocal()
processor = DataProcessor(db)

print(" Połączono z bazą danych")

# =============================================================================
# KROK 2: Ekstrakcja danych treningowych
# =============================================================================
print("\nKROK 2: Ekstrakcja features z bazy danych")
print("-" * 80)

print("Pobieranie danych z tabeli ratings...")
df = processor.get_training_data()

if df.empty:
    print(" BŁĄD: Brak danych w tabeli ratings!")
    print("Dodaj oceny filmów przez system przed trenowaniem.")
    db.close()
    sys.exit(1)

print(f" Pobrano {len(df)} rekordów z bazy danych")
print(f"\nKolumny: {list(df.columns)}")
print(f"\nPrzykładowe wiersze:")
print(df.head())

# Statystyki
print(f"\n Statystyki danych:")
print(f"   - Unikalnych użytkowników: {df['user_id'].nunique()}")
print(f"   - Unikalnych filmów: {df['movie_id'].nunique()}")
print(f"   - Średni rating (znormalizowany): {df['target'].mean():.3f}")
print(f"   - Std rating: {df['target'].std():.3f}")

# =============================================================================
# KROK 3: Wybór cech dla ANFIS
# =============================================================================
print("\n KROK 3: Wybór 3 najważniejszych cech")
print("-" * 80)

# Analiza korelacji z targetem
feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)

print("Korelacja cech z ratingiem:")
for feature, corr in correlations.items():
    print(f"   - {feature}: {corr:.4f}")

# Wybierz top 3
top_3_features = correlations.head(3).index.tolist()
top_3_names = [f.replace('m_', '') for f in top_3_features]

print(f"\n Wybrano top 3 cechy: {top_3_names}")

# =============================================================================
# KROK 4: Przygotowanie danych dla ANFIS
# =============================================================================
print("\n KROK 4: Przygotowanie danych treningowych/testowych")
print("-" * 80)

# ⚡ SAMPLING: Ogranicz dane jeśli dataset jest bardzo duży
print(f"\n  Dataset zawiera {len(df)} próbek")
MAX_SAMPLES = 5000  # Dla szybkiego treningu

if len(df) > MAX_SAMPLES:
    print(f"   ⚡ UWAGA: Ograniczam do {MAX_SAMPLES} próbek (ANFIS jest bardzo wolny!)")
    print(f"   Powód: Ewaluacja na {len(df)*0.2:.0f} próbkach zajęłaby ~{len(df)*0.2/60:.0f} minut")
    df = df.sample(n=MAX_SAMPLES, random_state=42)
    print(f" Wybrano losowo {len(df)} próbek")
else:
    print(f" Dataset OK - {len(df)} próbek")

# Dla uproszczenia: używamy tylko cech filmu (m_*)
# W prawdziwym systemie user features są dynamiczne (per user)
# Ale do treningu modelu używamy agregowanych statystyk

# Wybierz tylko top 3 cechy
X = df[top_3_features].copy()
X.columns = [f'u_{name}' for name in top_3_names]  # Rename dla ANFIS

# Dodaj movie features (te same wartości - uproszczenie)
for name in top_3_names:
    X[f'm_{name}'] = df[f'm_{name}']

y = df['target']

print(f"Features (X): {list(X.columns)}")
print(f"Target (y): rating (znormalizowany 0-1)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Podział danych:")
print(f"   - Train: {len(X_train)} próbek ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Test:  {len(X_test)} próbek ({len(X_test)/len(X)*100:.1f}%)")

# =============================================================================
# KROK 5: Budowa i trenowanie ANFIS
# =============================================================================
print("\n KROK 5: Budowa systemu ANFIS")
print("-" * 80)

anfis = ANFISRecommender(aspect_names=top_3_names)
anfis.build_fuzzy_system()

print(f" System ANFIS zbudowany:")
print(f"   - Aspekty: {anfis.aspect_names}")
print(f"   - Zmiennych wejściowych: {len(anfis.input_variables)}")
print(f"   - Reguł fuzzy: {len(anfis.rules)}")

print("\n🎓 Trenowanie modelu...")
anfis.train(X_train, y_train)
print(" Trenowanie zakończone!")

# =============================================================================
# KROK 6: Ewaluacja na zbiorze testowym
# =============================================================================
print("\n KROK 6: Ewaluacja modelu na zbiorze testowym")
print("-" * 80)

metrics = anfis.evaluate(X_test, y_test)

print(f"\n Metryki (znormalizowane 0-1):")
print(f"   RMSE: {metrics['rmse']:.4f}")
print(f"   MAE:  {metrics['mae']:.4f}")

# Denormalizacja do skali 1-5
rmse_denorm = metrics['rmse'] * 4  # zakres błędu
mae_denorm = metrics['mae'] * 4

print(f"\n Metryki (skala 1-5):")
print(f"   RMSE: {rmse_denorm:.4f} gwiazdek")
print(f"   MAE:  {mae_denorm:.4f} gwiazdek")

# Interpretacja
print(f"\nInterpretacja:")
if metrics['rmse'] < 0.15:
    print("   DOSKONAŁY wynik! (RMSE < 0.15)")
    quality = "EXCELLENT"
elif metrics['rmse'] < 0.25:
    print("   DOBRY wynik! (RMSE < 0.25)")
    quality = "GOOD"
elif metrics['rmse'] < 0.35:
    print("   ŚREDNI wynik (RMSE < 0.35)")
    quality = "AVERAGE"
else:
    print("   SŁABY wynik (RMSE > 0.35)")
    quality = "POOR"

# =============================================================================
# KROK 7: Wizualizacje
# =============================================================================
print("\n KROK 7: Generowanie wizualizacji")
print("-" * 80)

# Twórz folder dla wyników
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_anfis_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Wykres 1: Predykcje vs Prawdziwe wartości
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(metrics['actuals'], metrics['predictions'], alpha=0.5, edgecolors='k', s=30)
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Idealna predykcja')
plt.xlabel('Prawdziwe wartości (0-1)', fontsize=11)
plt.ylabel('Predykcje (0-1)', fontsize=11)
plt.title(f'ANFIS: Predykcje vs Rzeczywiste\nRMSE={metrics["rmse"]:.4f}', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

# Wykres 2: Rozkład błędów
errors = np.array(metrics['predictions']) - np.array(metrics['actuals'])
plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Błąd predykcji', fontsize=11)
plt.ylabel('Liczba próbek', fontsize=11)
plt.title(f'Rozkład błędów\nMAE={metrics["mae"]:.4f}', fontsize=12)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
plt.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2,
            label=f'Średni błąd: {errors.mean():.4f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = os.path.join(results_dir, 'anfis_evaluation.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f" Wykres zapisany: {plot_path}")
plt.close()

# Wykres 3: Boxplot błędów
plt.figure(figsize=(8, 6))
plt.boxplot([errors], labels=['ANFIS'], widths=0.5)
plt.ylabel('Błąd predykcji', fontsize=12)
plt.title('Rozkład błędów ANFIS', fontsize=14)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3, axis='y')
plot_path = os.path.join(results_dir, 'anfis_errors_boxplot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f" Wykres zapisany: {plot_path}")
plt.close()

# =============================================================================
# KROK 8: Zapis modelu
# =============================================================================
print("\n KROK 8: Zapis wytrenowanego modelu")
print("-" * 80)

# Stwórz folder models jeśli nie istnieje
models_dir = 'app/ml/models'
os.makedirs(models_dir, exist_ok=True)

# Zapisz model
model_filename = f'anfis_v1_{timestamp}.pkl'
model_path = os.path.join(models_dir, model_filename)
anfis.save(model_path)

# Zapisz też jako "latest"
latest_path = os.path.join(models_dir, 'anfis_latest.pkl')
anfis.save(latest_path)

print(f" Model zapisany:")
print(f"   - Wersjonowany: {model_path}")
print(f"   - Latest: {latest_path}")

# =============================================================================
# KROK 9: Raport tekstowy
# =============================================================================
print("\n KROK 9: Generowanie raportu")
print("-" * 80)

report = f"""
================================================================================
RAPORT TRENOWANIA ANFIS
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
Top 3 wybrane cechy: {', '.join(top_3_names)}

Korelacje z targetem:
{chr(10).join([f'  - {f}: {c:.4f}' for f, c in correlations.items()])}

PODZIAŁ DANYCH:
---------------
- Train: {len(X_train)} próbek ({len(X_train)/len(X)*100:.1f}%)
- Test:  {len(X_test)} próbek ({len(X_test)/len(X)*100:.1f}%)

MODEL ANFIS:
------------
- Aspekty: {', '.join(anfis.aspect_names)}
- Zmiennych wejściowych: {len(anfis.input_variables)}
- Reguł fuzzy: {len(anfis.rules)}

WYNIKI:
-------
Metryki (znormalizowane 0-1):
  - RMSE: {metrics['rmse']:.4f}
  - MAE:  {metrics['mae']:.4f}

Metryki (skala 1-5):
  - RMSE: {rmse_denorm:.4f} gwiazdek
  - MAE:  {mae_denorm:.4f} gwiazdek

Jakość modelu: {quality}

Interpretacja błędu MAE:
  Średnio model myli się o {mae_denorm:.2f} gwiazdki.
  Przykład: Jeśli prawdziwy rating to 4.0⭐, 
            model przewiduje średnio {4.0-mae_denorm:.2f} - {4.0+mae_denorm:.2f}⭐

PLIKI WYGENEROWANE:
-------------------
- Model: {model_path}
- Wykresy: {results_dir}/
- Raport: {os.path.join(results_dir, 'training_report.txt')}

================================================================================
"""

# Zapisz raport
report_path = os.path.join(results_dir, 'training_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f" Raport zapisany: {report_path}")

# Wyświetl raport
print(report)

# =============================================================================
# KROK 10: Test predykcji na przykładzie
# =============================================================================
print("\n KROK 10: Test predykcji na przykładzie")
print("-" * 80)

# Weź losowy przykład z test set
sample_idx = np.random.randint(0, len(X_test))
sample_features = X_test.iloc[sample_idx].to_dict()
true_rating = y_test.iloc[sample_idx]

predicted_rating = anfis.predict(sample_features)
predicted_denorm = (predicted_rating * 4) + 1
true_denorm = (true_rating * 4) + 1

print(f"Przykład #{sample_idx}:")
print(f"\nCechy:")
for key, value in sample_features.items():
    denorm_value = (value * 4) + 1
    print(f"   {key}: {value:.3f} (skala 1-5: {denorm_value:.2f})")

print(f"\nPrawdziwy rating: {true_rating:.3f} (skala 1-5: {true_denorm:.2f}⭐)")
print(f"Predykcja ANFIS:  {predicted_rating:.3f} (skala 1-5: {predicted_denorm:.2f}⭐)")
print(f"Błąd: {abs(predicted_rating - true_rating):.3f} (skala 1-5: {abs(predicted_denorm - true_denorm):.2f})")

# =============================================================================
# ZAKOŃCZENIE
# =============================================================================
print("\n" + "=" * 80)
print(" TRENOWANIE ZAKOŃCZONE POMYŚLNIE!")
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
   3. Jeśli metryki są dobre → przejdź do MLP Baseline
   4. Jeśli metryki są słabe → sprawdź dane i reguły fuzzy

 Użycie modelu w API:
   from app.ml.anfis_model import ANFISRecommender
   anfis = ANFISRecommender.load('app/ml/models/anfis_latest.pkl')
   prediction = anfis.predict(features)
""")

db.close()
print("\n Połączenie z bazą zamknięte")
print("=" * 80)