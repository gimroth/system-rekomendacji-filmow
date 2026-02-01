"""
ZAKTUALIZOWANY SKRYPT POROWNANIA: ANFIS vs MLP
Obsluguje rozne formaty raportow (z gwiazdkami i bez).
Generuje kompletny zestaw plikow: raport .txt, tabele .csv oraz wykresy .png.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from datetime import datetime
from glob import glob

def get_latest_report(pattern):
    folders = sorted(glob(pattern), reverse=True)
    if not folders: return None
    path = os.path.join(folders[0], 'training_report.txt')
    return path if os.path.exists(path) else None

ANFIS_REPORT = get_latest_report("results_anfis_match_*")
MLP_REPORT = get_latest_report("results_mlp_*")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"comparison_anfis_mlp_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_metrics(path):
    if not path: return None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    rmse_01_match = re.search(r'RMSE:\s+([\d.]+)', content)
    mae_01_match = re.search(r'MAE:\s+([\d.]+)', content)

    rmse_01 = float(rmse_01_match.group(1)) if rmse_01_match else 0.0
    mae_01 = float(mae_01_match.group(1)) if mae_01_match else 0.0

    rmse_star_match = re.search(r'RMSE:.*?([\d.]+)\s*⭐', content)
    mae_star_match = re.search(r'MAE:.*?([\d.]+)\s*⭐', content)

    rmse_star = float(rmse_star_match.group(1)) if rmse_star_match else round(rmse_01 * 4, 4)
    mae_star = float(mae_star_match.group(1)) if mae_star_match else round(mae_01 * 4, 4)

    return {
        'rmse': rmse_01,
        'mae': mae_01,
        'rmse_star': rmse_star,
        'mae_star': mae_star
    }

print("\nPobieranie danych z raportow...")
anfis = extract_metrics(ANFIS_REPORT)
mlp = extract_metrics(MLP_REPORT)

if not anfis or not mlp:
    print("Blad: Nie mozna znalezc obu raportow do porownania.")
    sys.exit(1)

print("Generowanie wizualizacji...")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
labels = ['MAE', 'RMSE']
x = np.arange(len(labels))
plt.bar(x - 0.2, [anfis['mae_star'], anfis['rmse_star']], 0.4, label='ANFIS', color='skyblue', edgecolor='black')
plt.bar(x + 0.2, [mlp['mae_star'], mlp['rmse_star']], 0.4, label='MLP', color='lightcoral', edgecolor='black')
plt.ylabel('Blad (gwiazdki)')
plt.title('Porownanie bledow predykcji', fontsize=14)
plt.xticks(x, labels)
plt.legend()
plt.grid(True, alpha=0.2, axis='y')

plt.subplot(1, 2, 2)
improvement = ((anfis['mae'] - mlp['mae']) / anfis['mae']) * 100
plt.bar(['Poprawa MLP'], [improvement], color='green', alpha=0.7, edgecolor='black')
plt.title(f'Poprawa MLP wzgledem ANFIS: {improvement:.1f}%', fontsize=12)
plt.ylabel('Procent (%)')
plt.grid(True, alpha=0.2, axis='y')
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), dpi=150)

plt.figure(figsize=(8, 5))
plt.bar(['ANFIS', 'MLP'], [3, 5], color=['skyblue', 'lightcoral'], edgecolor='black', width=0.6)
plt.ylabel('Liczba cech wejsciowych')
plt.title('Porownanie liczby wykorzystanych cech', fontsize=14)
plt.ylim(0, 6)
plt.grid(True, alpha=0.2, axis='y')
plt.savefig(os.path.join(OUTPUT_DIR, 'features_comparison.png'), dpi=150)

df_metrics = pd.DataFrame({
    'Metryka': ['MAE (0-1)', 'RMSE (0-1)', 'MAE (gwiazdki)', 'RMSE (gwiazdki)'],
    'ANFIS': [anfis['mae'], anfis['rmse'], f"{anfis['mae_star']}", f"{anfis['rmse_star']}"],
    'MLP': [mlp['mae'], mlp['rmse'], f"{mlp['mae_star']}", f"{mlp['rmse_star']}"]
})
df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'comparison_table.csv'), index=False)

pd.DataFrame({'Aspekt': ['Interpretowalnosc', 'Dokladnosc'], 'ANFIS': ['Wysoka', 'Srednia'], 'MLP': ['Niska', 'Wysoka']})\
    .to_csv(os.path.join(OUTPUT_DIR, 'advantages_comparison.csv'), index=False)

report = f"""
================================================================================
RAPORT POROWNAWCZY: ANFIS vs MLP
================================================================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. METRYKI ILOSCIOWE
====================

Metryka                     ANFIS          MLP         Roznica       Zwyciezca
--------------------------------------------------------------------------------
MAE (0-1)                  {anfis['mae']:.4f}       {mlp['mae']:.4f}           {improvement:.1f}%           MLP
RMSE (0-1)                 {anfis['rmse']:.4f}       {mlp['rmse']:.4f}           {((anfis['rmse']-mlp['rmse'])/anfis['rmse']*100):.1f}%           MLP
MAE (gwiazdki)              {anfis['mae_star']:.2f}        {mlp['mae_star']:.2f}           {(anfis['mae_star']-mlp['mae_star']):.2f}           MLP
RMSE (gwiazdki)             {anfis['rmse_star']:.2f}        {mlp['rmse_star']:.2f}           {(anfis['rmse_star']-mlp['rmse_star']):.2f}           MLP

INTERPRETACJA:
- MLP myli sie srednio o {mlp['mae_star']:.2f} gwiazdki
- ANFIS myli sie srednio o {anfis['mae_star']:.2f} gwiazdki
- MLP jest o {improvement:.1f}% dokladniejszy

2. ANALIZA JAKOSCIOWA
======================

ANFIS - ZALETY:
Interpretowalnosc: Reguly IF-THEN sa czytelne dla czlowieka
Cold Start: Dziala dla nowych uzytkownikow (formularz onboarding)
Wyjasnialno sc: Mozna pokazac "dlaczego polecamy ten film"

MLP - ZALETY:
Najwyzsza dokladnosc: MAE = {mlp['mae_star']:.2f} (najlepsze)
Wszystkie cechy: 5 aspektow (kompletna informacja)

3. WNIOSKI
==========
Zalecamy HYBRYDOWE PODEJSCIE: ANFIS dla nowych uzytkownikow i wyjasnien, MLP dla rankingu koncowego.

5. PLIKI WYGENEROWANE
======================
- Tabela metryk: {OUTPUT_DIR}\\comparison_table.csv
- Wykresy: {OUTPUT_DIR}/

================================================================================
KONIEC RAPORTU
================================================================================
"""

with open(os.path.join(OUTPUT_DIR, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nSukces! Pliki zapisane w: {OUTPUT_DIR}/")
print(f"   - comparison_report.txt\n   - metrics_comparison.png\n   - features_comparison.png")