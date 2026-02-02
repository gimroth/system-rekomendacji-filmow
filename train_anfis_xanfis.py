import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Wymuszenie trybu bezokienkowego (kluczowe dla zapisu plików na serwerze)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold

sys.path.insert(0, os.path.abspath('.'))
from app.database import SessionLocal
from app.ml.data_processor import DataProcessor
from app.ml.anfis_xanfis import XANFISWrapper

# =============================================================================
# FUNKCJE WIZUALIZACJI (STYL Z PLIKU REFERENCYJNEGO)
# =============================================================================

def generate_analysis_dashboard(history, y_true, y_pred, model, feature_names, save_path):
    """Generuje zbiorczy dashboard (Loss, Scatter, Hist, MFs)."""
    
    # Przygotowanie danych
    losses = history.get('loss', [])
    errors = y_pred - y_true
    
    # Ustawienia wykresu
    fig = plt.figure(figsize=(20, 15))
    
    # 1. KRZYWA UCZENIA (Training History)
    plt.subplot(3, 3, 1)
    if losses:
        plt.plot(losses, label='Training Loss (MSE)', linewidth=2, color='tab:blue')
        plt.xlabel('Epoka', fontsize=11)
        plt.ylabel('Loss (MSE)', fontsize=11)
        plt.title('Historia Treningu', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Brak historii treningu', ha='center', va='center')

    # 2. PREDYKCJE VS RZECZYWISTE (Scatter)
    plt.subplot(3, 3, 2)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', s=40, color='tab:green')
    
    # Linia idealna 0-1
    min_val, max_val = 0, 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Idealna predykcja')
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Prawdziwe (0-1)', fontsize=11)
    plt.ylabel('Predykcje (0-1)', fontsize=11)
    plt.title(f'Predykcje vs Rzeczywiste\nRMSE={rmse:.4f}, R2={r2:.4f}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # 3. ROZKŁAD BŁĘDÓW (Histogram)
    plt.subplot(3, 3, 3)
    plt.hist(errors, bins=30, color='lightblue', edgecolor='black', alpha=0.8)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    plt.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                label=f'Średni błąd: {np.mean(errors):.4f}')
    
    mae = mean_absolute_error(y_true, y_pred)
    plt.xlabel('Błąd (Pred - True)', fontsize=11)
    plt.ylabel('Liczba próbek', fontsize=11)
    plt.title(f'Rozkład błędów\nMAE={mae:.4f}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.2)

    # 4. FUNKCJE PRZYNALEŻNOŚCI (Próba wizualizacji)
    # Próbujemy narysować MF w dolnych kafelkach, jeśli model na to pozwala
    try:
        # Hack: xanfis rysuje na aktywnym axes, więc spróbujmy aktywować subplot
        if hasattr(model.model, 'network') and hasattr(model.model.network, 'plot_mfs'):
            # Rysujemy MF w jednym z dolnych okienek (uproszczone)
            ax = plt.subplot(3, 1, 3) # Zajmie cały dół
            # Uwaga: xanfis może tworzyć własne figure, więc to jest ryzykowne,
            # dlatego robimy to w bloku try
            plt.title("Funkcje Przynależności (Warstwa 1)", fontsize=13, fontweight='bold')
            # Tutaj niestety xanfis jest specyficzny, więc zostawiamy puste miejsce na opis
            plt.text(0.5, 0.5, "Szczegółowy wykres MF zapisano w 'mfs_after.png'", 
                     ha='center', va='center', fontsize=12)
            plt.axis('off')
    except:
        pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_cv_boxplot(cv_scores, save_path):
    """Generuje wykres pudełkowy z wyników Cross-Walidacji."""
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [cv_scores] # Oczekujemy listy RMSE
    
    bp = plt.boxplot(data_to_plot, positions=[1], widths=0.6, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        
    plt.xticks([1], ['RMSE'])
    plt.ylabel('Wartość błędu', fontsize=12)
    plt.title('Stabilność modelu (Cross-Walidacja)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Dodanie punktów poszczególnych foldów
    y = cv_scores
    x = np.random.normal(1, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.5)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# GŁÓWNA PĘTLA
# =============================================================================

def main(epochs=150): 
    print("=" * 80)
    print("TRENOWANIE ANFIS (XANFIS WRAPPER)")
    print("=" * 80)
    
    db = SessionLocal()
    processor = DataProcessor(db)
    df = processor.get_training_data()
    
    if df.empty:
        print('Błąd: Brak danych w bazie.')
        return

    # Folder na wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_xanfis_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 Folder wyników: {results_dir}")

    # Przygotowanie danych
    df = df.sample(n=min(30000, len(df)), random_state=42)
    feature_cols = ['m_story', 'm_acting', 'm_visuals', 'm_sound', 'm_direction']
    top_3 = df[feature_cols].corrwith(df['target']).sort_values(ascending=False).head(3).index.tolist()
    feature_names = [f.replace('m_', '') for f in top_3]

    print(f"Wybrane cechy (Top 3): {feature_names}")

    X = pd.DataFrame({f'match{i+1}': df[feat] for i, feat in enumerate(top_3)})
    y = df['target']

    # --- KROK 1: CROSS-WALIDACJA (3-Fold) ---
    print(f"\nKROK 1: Cross-Walidacja (3-Fold)...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Krótki trening dla sprawdzenia stabilności
        cv_model = XANFISWrapper(n_inputs=3, n_mfs=3)
        cv_model.fit(X_fold_train, y_fold_train, epochs=30, lr=1e-3)
        
        preds = cv_model.predict(X_fold_val)
        rmse_val = np.sqrt(mean_squared_error(y_fold_val, preds))
        cv_scores.append(rmse_val)
        print(f"   Fold {fold+1} RMSE: {rmse_val:.4f}")

    avg_cv_rmse = np.mean(cv_scores)
    
    # Generowanie wykresu Cross-Walidacji
    generate_cv_boxplot(cv_scores, os.path.join(results_dir, 'cross_validation.png'))
    print("✅ Zapisano: cross_validation.png")

    # --- KROK 2: TRENING FINALNY ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print(f'\nKROK 2: Budowa i trening modelu finalnego...')
    model = XANFISWrapper(n_inputs=3, n_mfs=3)

    # Zapisz MF przed nauką (standardowa metoda xanfis)
    model.plot_mfs(save_path=os.path.join(results_dir, 'mfs_before.png'))

    print(f'   Rozpoczęcie treningu (lr=1e-4, epochs={epochs})...')
    history = model.fit(X_train, y_train, epochs=epochs, lr=1e-4, batch_size=32)

    # --- KROK 3: WIZUALIZACJA I RAPORTOWANIE ---
    print(f'\nKROK 3: Generowanie raportów i wykresów...')
    
    # Zapisz MF po nauce
    model.plot_mfs(save_path=os.path.join(results_dir, 'mfs_after.png'))

    # Predykcje
    preds = model.predict(X_test)
    y_test_np = np.asarray(y_test).flatten()
    preds_np = np.asarray(preds).flatten()

    # GENEROWANIE DASHBOARDU (Loss + Scatter + Hist)
    generate_analysis_dashboard(
        history, y_test_np, preds_np, model, feature_names,
        os.path.join(results_dir, 'anfis_analysis.png')
    )
    print("✅ Zapisano: anfis_analysis.png (Dashboard)")

    # Metryki
    rmse = np.sqrt(mean_squared_error(y_test_np, preds_np))
    mae = mean_absolute_error(y_test_np, preds_np)
    r2 = r2_score(y_test_np, preds_np)

    # Skalowanie gwiazdkowe (dla człowieka)
    rmse_stars = rmse * 4
    mae_stars = mae * 4

    # Ocena jakości
    if rmse < 0.15: quality = "BARDZO DOBRA"
    elif rmse < 0.25: quality = "DOBRA"
    else: quality = "ŚREDNIA"

    # Zapis modelu
    model.save_stable(os.path.join('app/ml/models', 'xanfis_latest.pkl'), top_3=top_3)
    
    # Raport tekstowy
    report = f"""
================================================================================
RAPORT: XANFIS (5-LAYER ARCHITECTURE)
================================================================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

WYNIKI CROSS-WALIDACJI (3-FOLD):
--------------------------------
RMSE: {avg_cv_rmse:.4f} +/- {np.std(cv_scores):.4f}

WYNIKI NA ZBIORZE TESTOWYM:
---------------------------
RMSE: {rmse:.4f} (skala 0-1)  ->  {rmse_stars:.2f} gwiazdek
MAE:  {mae:.4f} (skala 0-1)  ->  {mae_stars:.2f} gwiazdek
R2:   {r2:.4f}

Ocena jakości modelu: {quality}

PARAMETRY:
----------
- Epoki: {epochs}
- Cechy wejściowe: {', '.join(feature_names)}
- Struktura: 5 Warstw (Input -> MF -> Rules -> Norm -> Output)

PLIKI GRAFICZNE:
----------------
1. anfis_analysis.png  -> Zbiorczy dashboard (Krzywa uczenia, Scatter, Histogram)
2. cross_validation.png -> Wykres stabilności (Boxplot)
3. mfs_before/after.png -> Wykresy funkcji przynależności (Gauss)

================================================================================
"""
    
    with open(os.path.join(results_dir, 'raport_techniczny.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'\nRAPORT KOŃCOWY:')
    print(f'   RMSE: {rmse:.4f}')
    print(f'   R2:   {r2:.4f}')
    print(f'   Jakość: {quality}')
    print(f'Wszystkie wyniki w: {results_dir}')
    print("=" * 80)
    db.close()

if __name__ == '__main__':
    main(epochs=150)