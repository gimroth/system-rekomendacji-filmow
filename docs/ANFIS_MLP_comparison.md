Instrukcja: Kompletny test porównawczy ANFIS vs MLP

Cel
- Wytrenować ANFIS i MLP na dwóch zbiorach (dane z bazy + syntetyczny) i wygenerować porównawcze wykresy i raporty.

Wymagania wstępne
- Python 3.8+ (zalecane 3.11)
- Aktywowane wirtualne środowisko w katalogu projektu
- Zainstalowane zależności: `pip install -r requirements.txt`

Pliki i skrypty
- `app/ml/anfis_model.py` — hybrydowy ANFIS (teraz z funkcjami gaussowskimi)
- `app/ml/generate_synthetic_dataset.py` — generator syntetycznego zbioru
- `train_compare_multi.py` — skrypt trenujący ANFIS i MLP na dwóch zbiorach i generujący porównanie

Kroki (szybkie)
1. Aktywuj venv (Windows PowerShell):

```powershell
& .\venv\Scripts\Activate.ps1
```

2. Zainstaluj wymagane paczki (jeśli jeszcze nie zainstalowane):

```powershell
python -m pip install -r requirements.txt
```

3. Uruchom skrypt porównawczy (trenuje i zapisuje wyniki):

```powershell
python train_compare_multi.py
```

4. Wyniki:
- Dla zbioru z bazy: folder `results_anfis_db_<timestamp>/` zawiera wykresy MF, CV metrics, pred_vs_actual.png, training_summary.json oraz zapisany model w `app/ml/models`.
- Dla zbioru syntetycznego: folder `results_anfis_synthetic_<timestamp>/` podobnie.
- Porównanie końcowe: folder `comparison_anfis_mlp_multi_<timestamp>/` z `rmse_comparison.png` i `comparison_report.json`.

Szczegóły implementacyjne
- W `app/ml/anfis_model.py` funkcje przynależności zostały zmienione na gaussowskie (fuzz.gaussmf). To wyrównuje bazę funkcji z implementacją PyTorch (gaussowskie MF) i ułatwia porównania.
- Skrypt `train_compare_multi.py`:
  - Tworzy syntetyczny zbiór `app/ml/datasets/anfis_synthetic.csv` jeśli go nie ma.
  - Dla każdego zbioru: oblicza top-3 cech (dla ANFIS), przeprowadza K-Fold CV i trenuje finalny model na całym zbiorze.
  - Dla MLP używa tych samych danych (5 aspektów) i zapisuje model oraz wykresy.
  - Na końcu tworzy wykres porównawczy RMSE (ANFIS vs MLP) dla obu zbiorów.

Jak interpretować wyniki
- `cv_metrics.png` — RMSE i R2 na poszczególnych foldach dla ANFIS.
- `pred_vs_actual.png` — rozrzut predykcji względem rzeczywistych wartości.
- `mlp_evaluation.png` — podobne wykresy dla MLP.
- `rmse_comparison.png` — prosty wykres porównujący średnie RMSE ANFIS i MLP na obu zbiorach.

Następne kroki (opcjonalne)
- Dostosować parametry MFs (sigma) w `app/ml/anfis_model.py` i powtórzyć testy.
- Uruchomić dłuższe treningi PyTorch‑ANFIS (jeśli chcesz gradientowy wariant).
- Dodać endpoint admina zwracający szczegóły reguł/firing strengths dla pojedynczych predykcji.

Kontakt
- Jeśli chcesz, mogę:
  - (A) dopracować parametry gaussowskie i powtórzyć porównania, lub
  - (B) dodać endpoint debugujący dla per-sample explainability.

