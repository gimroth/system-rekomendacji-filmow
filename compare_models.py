"""
Skrypt porównania ANFIS vs MLP
Generuje kompletne porównanie dla dokumentacji projektu
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ścieżki do raportów
ANFIS_REPORT = "results_anfis_20260112_175541/training_report.txt"
MLP_REPORT = "results_mlp_20260112_182434/training_report.txt"

# Output folder
OUTPUT_DIR = f"comparison_anfis_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PORÓWNANIE ANFIS vs MLP")
print("=" * 80)

# =============================================================================
# KROK 1: Zbierz metryki z raportów
# =============================================================================
print("\nKROK 1: Zbieranie metryk z raportów")
print("-" * 80)


def extract_metrics(report_path):
    """Wyciąga metryki z raportu tekstowego."""
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Znajdź RMSE i MAE (znormalizowane)
    import re
    rmse_match = re.search(r'RMSE:\s+([\d.]+)', content)
    mae_match = re.search(r'MAE:\s+([\d.]+)', content)

    # Znajdź RMSE i MAE (gwiazdki)
    rmse_denorm_match = re.search(r'RMSE:\s+([\d.]+)\s+gwiazdek', content)
    mae_denorm_match = re.search(r'MAE:\s+([\d.]+)\s+gwiazdek', content)

    # Znajdź epochs (tylko MLP)
    epochs_match = re.search(r'Epochs:\s+(\d+)', content)

    return {
        'rmse': float(rmse_match.group(1)) if rmse_match else None,
        'mae': float(mae_match.group(1)) if mae_match else None,
        'rmse_denorm': float(rmse_denorm_match.group(1)) if rmse_denorm_match else None,
        'mae_denorm': float(mae_denorm_match.group(1)) if mae_denorm_match else None,
        'epochs': int(epochs_match.group(1)) if epochs_match else 1
    }


anfis_metrics = extract_metrics(ANFIS_REPORT)
mlp_metrics = extract_metrics(MLP_REPORT)

print(f"ANFIS metrics: RMSE={anfis_metrics['rmse']:.4f}, MAE={anfis_metrics['mae']:.4f}")
print(f"MLP metrics:   RMSE={mlp_metrics['rmse']:.4f}, MAE={mlp_metrics['mae']:.4f}")

# =============================================================================
# KROK 2: Tabela porównawcza
# =============================================================================
print("\nKROK 2: Generowanie tabeli porównawczej")
print("-" * 80)

comparison_data = {
    'Metryka': ['MAE (0-1)', 'RMSE (0-1)', 'MAE (gwiazdki)', 'RMSE (gwiazdki)', 'Epochs', 'Czas treningu'],
    'ANFIS': [
        f"{anfis_metrics['mae']:.4f}",
        f"{anfis_metrics['rmse']:.4f}",
        f"{anfis_metrics['mae_denorm']:.2f}⭐",
        f"{anfis_metrics['rmse_denorm']:.2f}⭐",
        "1 (statyczny)",
        "~10s"
    ],
    'MLP': [
        f"{mlp_metrics['mae']:.4f}",
        f"{mlp_metrics['rmse']:.4f}",
        f"{mlp_metrics['mae_denorm']:.2f}⭐",
        f"{mlp_metrics['rmse_denorm']:.2f}⭐",
        f"{mlp_metrics['epochs']} (early stop)",
        "~20s"
    ],
    'Różnica': [
        f"{((anfis_metrics['mae'] - mlp_metrics['mae']) / anfis_metrics['mae'] * 100):.1f}%",
        f"{((anfis_metrics['rmse'] - mlp_metrics['rmse']) / anfis_metrics['rmse'] * 100):.1f}%",
        f"{(anfis_metrics['mae_denorm'] - mlp_metrics['mae_denorm']):.2f}⭐",
        f"{(anfis_metrics['rmse_denorm'] - mlp_metrics['rmse_denorm']):.2f}⭐",
        "-",
        "-"
    ],
    'Zwycięzca': [
        '🏆 MLP' if mlp_metrics['mae'] < anfis_metrics['mae'] else '🏆 ANFIS',
        '🏆 MLP' if mlp_metrics['rmse'] < anfis_metrics['rmse'] else '🏆 ANFIS',
        '🏆 MLP' if mlp_metrics['mae_denorm'] < anfis_metrics['mae_denorm'] else '🏆 ANFIS',
        '🏆 MLP' if mlp_metrics['rmse_denorm'] < anfis_metrics['rmse_denorm'] else '🏆 ANFIS',
        '-',
        '🏆 ANFIS'
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Zapisz tabelę
table_path = os.path.join(OUTPUT_DIR, 'comparison_table.csv')
df_comparison.to_csv(table_path, index=False, encoding='utf-8')
print(f"\nTabela zapisana: {table_path}")

# =============================================================================
# KROK 3: Wykres porównawczy metryk
# =============================================================================
print("\nKROK 3: Generowanie wykresów porównawczych")
print("-" * 80)

# Wykres 1: Bar chart metryk
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MAE
metrics_names = ['MAE', 'RMSE']
anfis_values = [anfis_metrics['mae_denorm'], anfis_metrics['rmse_denorm']]
mlp_values = [mlp_metrics['mae_denorm'], mlp_metrics['rmse_denorm']]

x = np.arange(len(metrics_names))
width = 0.35

axes[0].bar(x - width / 2, anfis_values, width, label='ANFIS', color='skyblue', edgecolor='navy')
axes[0].bar(x + width / 2, mlp_values, width, label='MLP', color='lightcoral', edgecolor='darkred')
axes[0].set_xlabel('Metryka', fontsize=12)
axes[0].set_ylabel('Błąd (gwiazdki)', fontsize=12)
axes[0].set_title('Porównanie błędów predykcji', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Dodaj wartości na słupkach
for i, (a, m) in enumerate(zip(anfis_values, mlp_values)):
    axes[0].text(i - width / 2, a + 0.02, f'{a:.2f}', ha='center', fontsize=10, fontweight='bold')
    axes[0].text(i + width / 2, m + 0.02, f'{m:.2f}', ha='center', fontsize=10, fontweight='bold')

# Improvement percentage
improvements = [
    ((anfis_metrics['mae'] - mlp_metrics['mae']) / anfis_metrics['mae'] * 100),
    ((anfis_metrics['rmse'] - mlp_metrics['rmse']) / anfis_metrics['rmse'] * 100)
]

axes[1].bar(metrics_names, improvements, color='green', alpha=0.7, edgecolor='darkgreen')
axes[1].set_xlabel('Metryka', fontsize=12)
axes[1].set_ylabel('Poprawa MLP (%)', fontsize=12)
axes[1].set_title('MLP: Poprawa względem ANFIS', fontsize=14, fontweight='bold')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[1].grid(True, alpha=0.3, axis='y')

# Dodaj wartości
for i, val in enumerate(improvements):
    axes[1].text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Wykres zapisany: {plot_path}")
plt.close()

# =============================================================================
# KROK 4: Wykres cech (ANFIS vs MLP)
# =============================================================================
print("\nKROK 4: Porównanie wykorzystanych cech")
print("-" * 80)

fig, ax = plt.subplots(figsize=(10, 6))

models = ['ANFIS', 'MLP']
features_count = [3, 5]
colors = ['skyblue', 'lightcoral']

bars = ax.bar(models, features_count, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Liczba cech wejściowych', fontsize=12)
ax.set_title('Porównanie liczby wykorzystanych cech', fontsize=14, fontweight='bold')
ax.set_ylim(0, 6)
ax.grid(True, alpha=0.3, axis='y')

# Dodaj etykiety
for bar, count, model in zip(bars, features_count, models):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
            f'{count} cechy' if count == 3 else f'{count} cech',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Dodaj listę cech
    if model == 'ANFIS':
        features_text = 'story, direction, acting'
    else:
        features_text = 'story, acting, visuals,\nsound, direction'

    ax.text(bar.get_x() + bar.get_width() / 2., 0.5,
            features_text,
            ha='center', va='bottom', fontsize=9, style='italic')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'features_comparison.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Wykres zapisany: {plot_path}")
plt.close()

# =============================================================================
# KROK 5: Tabela zalety/wady
# =============================================================================
print("\nROK 5: Analiza zalet i wad")
print("-" * 80)

advantages = {
    'Aspekt': [
        'Dokładność (MAE)',
        'Dokładność (RMSE)',
        'Interpretowalność',
        'Cold Start',
        'Wyjaśnialność',
        'Czas treningu',
        'Zaufanie użytkownika',
        'Debugowanie'
    ],
    'ANFIS': [
        f'0.40⭐',
        f'0.48⭐',
        '✅ Wysoka (reguły)',
        '✅ Działa (formularz)',
        '✅ Łatwe',
        '✅ Szybki (~10s)',
        '✅ Wyższe',
        '✅ Łatwe'
    ],
    'MLP': [
        f'0.22⭐ 🏆',
        f'0.29⭐ 🏆',
        '❌ Niska (black box)',
        '❌ Nie działa',
        '❌ Trudne',
        '⚠️ Średni (~20s)',
        '⚠️ Niższe',
        '❌ Trudne'
    ],
    'Zwycięzca': [
        '🏆 MLP',
        '🏆 MLP',
        '🏆 ANFIS',
        '🏆 ANFIS',
        '🏆 ANFIS',
        '🏆 ANFIS',
        '🏆 ANFIS',
        '🏆 ANFIS'
    ]
}

df_advantages = pd.DataFrame(advantages)
print("\n" + df_advantages.to_string(index=False))

# Zapisz
adv_path = os.path.join(OUTPUT_DIR, 'advantages_comparison.csv')
df_advantages.to_csv(adv_path, index=False, encoding='utf-8')
print(f"\nTabela zapisana: {adv_path}")

# =============================================================================
# KROK 6: Raport końcowy
# =============================================================================
print("\nKROK 6: Generowanie raportu końcowego")
print("-" * 80)

report = f"""
================================================================================
RAPORT PORÓWNAWCZY: ANFIS vs MLP
================================================================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. METRYKI ILOŚCIOWE
====================

{'Metryka':<20} {'ANFIS':>12} {'MLP':>12} {'Różnica':>15} {'Zwycięzca':>15}
{'-' * 80}
{'MAE (0-1)':<20} {anfis_metrics['mae']:>12.4f} {mlp_metrics['mae']:>12.4f} {((anfis_metrics['mae'] - mlp_metrics['mae']) / anfis_metrics['mae'] * 100):>14.1f}% {'🏆 MLP':>15}
{'RMSE (0-1)':<20} {anfis_metrics['rmse']:>12.4f} {mlp_metrics['rmse']:>12.4f} {((anfis_metrics['rmse'] - mlp_metrics['rmse']) / anfis_metrics['rmse'] * 100):>14.1f}% {'🏆 MLP':>15}
{'MAE (gwiazdki)':<20} {anfis_metrics['mae_denorm']:>11.2f}⭐ {mlp_metrics['mae_denorm']:>11.2f}⭐ {(anfis_metrics['mae_denorm'] - mlp_metrics['mae_denorm']):>14.2f}⭐ {'🏆 MLP':>15}
{'RMSE (gwiazdki)':<20} {anfis_metrics['rmse_denorm']:>11.2f}⭐ {mlp_metrics['rmse_denorm']:>11.2f}⭐ {(anfis_metrics['rmse_denorm'] - mlp_metrics['rmse_denorm']):>14.2f}⭐ {'🏆 MLP':>15}

INTERPRETACJA:
- MLP myli się średnio o {mlp_metrics['mae_denorm']:.2f} gwiazdki
- ANFIS myli się średnio o {anfis_metrics['mae_denorm']:.2f} gwiazdki
- MLP jest o {((anfis_metrics['mae'] - mlp_metrics['mae']) / anfis_metrics['mae'] * 100):.1f}% dokładniejszy

2. ANALIZA JAKOŚCIOWA
======================

ANFIS - ZALETY:
Interpretowalność: Reguły IF-THEN są czytelne dla człowieka
Cold Start: Działa dla nowych użytkowników (formularz onboarding)
Wyjaśnialność: Można pokazać "dlaczego polecamy ten film"
Zaufanie: Użytkownicy bardziej ufają zrozumiałym rekomendacjom
Debugowanie: Łatwo znaleźć źródło błędów

ANFIS - WADY:
Niższa dokładność: MAE o 82% wyższe niż MLP
Mniej cech: Tylko 3 aspekty (top 3 z EDA)
Brak uczenia: Statyczne reguły (nie adaptują się)

MLP - ZALETY:
Najwyższa dokładność: MAE = {mlp_metrics['mae_denorm']:.2f}⭐ (najlepsze)
Wszystkie cechy: 5 aspektów (kompletna informacja)
Prawdziwe uczenie: Backpropagation, gradient descent
Szybkie trenowanie: 45 epochs z early stopping

MLP - WADY:
Black box: Trudno wyjaśnić predykcje
Brak Cold Start: Potrzebuje ≥5 ocen od użytkownika
Niższe zaufanie: Użytkownicy nie rozumieją "dlaczego"

3. WNIOSKI
==========

W systemie produkcyjnym zalecamy HYBRYDOWE PODEJŚCIE:

1. COLD START (< 5 ocen):
   → Użyj ANFIS + formularz onboarding
   → Wyjaśnienie: "Polecamy bo lubisz filmy z wysoką fabułą"
   → Zaufanie: Użytkownik rozumie rekomendacje

2. NORMALNI UŻYTKOWNICY (≥ 5 ocen):
   → Użyj MLP dla dokładności predykcji
   → Wygeneruj rekomendacje MLP (najdokładniejsze)
   → Dodaj wyjaśnienia ANFIS (dla zrozumienia)

3. "DLACZEGO POLECAMY":
   → Użyj reguł ANFIS do generowania wyjaśnień
   → Przykład: "Ten film ma świetną fabułę (4.8⭐) 
               i reżyserię (4.7⭐), które cenisz"

To łączy:
Dokładność MLP ({mlp_metrics['mae_denorm']:.2f}⭐ błędu)
Interpretowalność ANFIS (reguły IF-THEN)
Cold Start ANFIS (formularz)

4. REKOMENDACJE IMPLEMENTACYJNE
================================

KROK 1: Endpoint /recommendations
```python
if user.ratings_count < 5:
    # Cold Start
    recommendations = anfis.predict(user)
    explanations = anfis.explain(user, movie)
else:
    # Normal
    recommendations = mlp.predict(user)  # Dokładność
    explanations = anfis.explain(user, movie)  # Wyjaśnienie
```

KROK 2: Dashboard
- Pokaż porównanie ANFIS vs MLP (wykresy)
- Dodaj przycisk "Dlaczego polecamy?" → ANFIS rules
- Wizualizuj top 3 aspekty użytkownika

KROK 3: A/B Testing
- Testuj ANFIS vs MLP vs Hybrid
- Metryki: Click-through rate, User satisfaction

5. PLIKI WYGENEROWANE
======================
- Tabela metryk: {table_path}
- Tabela zalet/wad: {adv_path}
- Wykresy: {OUTPUT_DIR}/

================================================================================
KONIEC RAPORTU
================================================================================
"""

# Zapisz raport
report_path = os.path.join(OUTPUT_DIR, 'comparison_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Raport zapisany: {report_path}")

# Wyświetl raport
print("\n" + report)

# =============================================================================
# ZAKOŃCZENIE
# =============================================================================
print("\n" + "=" * 80)
print("✅ PORÓWNANIE ZAKOŃCZONE POMYŚLNIE!")
print("=" * 80)

print(f"""
Pliki wygenerowane w folderze: {OUTPUT_DIR}/
   - comparison_table.csv
   - advantages_comparison.csv
   - comparison_report.txt
   - metrics_comparison.png
   - features_comparison.png

Co dalej:
   1. Sprawdź wykresy
   2. Przeczytaj raport porównawczy
   3. Dodaj do dokumentacji projektu
   4. Gotowe na prezentację!

Kluczowe wnioski:
   - MLP jest dokładniejszy ({mlp_metrics['mae_denorm']:.2f}⭐ vs {anfis_metrics['mae_denorm']:.2f}⭐)
   - ANFIS jest bardziej interpretowalny
   - Zalecamy hybrydowe podejście w produkcji
""")

print("=" * 80)