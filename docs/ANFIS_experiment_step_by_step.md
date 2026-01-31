ANFIS vs MLP — Eksperyment: krok po kroku

Cel
- Przeprowadzić porównawczy test jakości predykcji między hybrydowym ANFIS a MLP na dwóch zbiorach: dane z bazy (DB) i syntetyczny (synthetic).

Wyniki (przykładowe z ostatniego uruchomienia)
- Dataset `db`:
  - ANFIS: RMSE = 0.08896, R2 = 0.8850
  - MLP:   RMSE = 0.07348, R2 = 0.9202
  - Winner: MLP (niższe RMSE)
- Dataset `synthetic`:
  - ANFIS: RMSE = 0.07541, R2 = 0.7278
  - MLP:   RMSE = 0.05548, R2 = 0.8549
  - Winner: MLP

Gdzie znajdują się pliki wygenerowane przez skrypt
- Porównanie RMSE: `comparison_anfis_mlp_multi_<timestamp>/rmse_comparison.png`
- Foldery wyników (ANFIS/MLP) dla każdego zbioru: `results_anfis_db_<timestamp>/` i `results_anfis_synthetic_<timestamp>/`
  - `cv_metrics.png`, `pred_vs_actual.png`, `mlp_evaluation.png`, `mf_fold_*` (obrazy funkcji przynależności)

Interpretacja metryk
- RMSE (Root Mean Squared Error): średni błąd w tej samej skali co target (tu 0-1). Niższe = lepsze.
- MAE (Mean Absolute Error): średni bezwzględny błąd. Również niższe = lepsze.
- R2 (coefficient of determination): udział wariancji wyjaśnionej przez model (1.0 = idealnie dopasowany; 0 = model nie lepszy niż średnia).

Co oznaczają uzyskane różnice
- W tym eksperymencie MLP osiąga niższe RMSE i wyższe R2 na obu zbiorach, co oznacza, że MLP jest dokładniejszy w przewidywaniu znormalizowanych ocen niż hybrydowy ANFIS przy obecnych ustawieniach i danych.
- ANFIS oferuje interpretowalność (możliwość sprawdzenia MF, firing strengths i per-rule contributions), co jest widoczne w endpointzie explain i w folderze `mf_fold_*`.

Jak uruchomić test i zobaczyć różnice w terminalu
1. Aktywuj venv:

```powershell
& .\venv\Scripts\Activate.ps1
```

2. (Opcjonalnie) wygeneruj syntetyczny zbiór ręcznie:

```powershell
python -c "from app.ml.generate_synthetic_dataset import generate; generate('app/ml/datasets/anfis_synthetic.csv', n_samples=3000)"
```

3. Uruchom porównanie (trening + CV + zapisy):

```powershell
python train_compare_multi.py
```

4. Po zakończeniu sprawdź folder `comparison_anfis_mlp_multi_<timestamp>/` i uruchom podsumowanie:

```powershell
python scripts/summarize_comparisons.py comparison_anfis_mlp_multi_<timestamp>
```

To wypisze RMSE/R2 i wskaże "zwycięzcę".

Jak pokażesz, że ANFIS działa i jak debugować pojedyncze predykcje
- Endpoint explain (admin): `/admin/anfis/explain` — zwróci szczegóły dla pojedynczej próbki (możesz podać `match1,match2,match3` lub `user_id`+`movie_id`).
- Lokalny skrypt testowy: `scripts/example_explain.py` — uruchamia explain bez potrzeby auth i wypisuje JSON z membership/firing/weights/consequents/per_rule_contribution/prediction.

Przykład — zobaczenie wpływu nowych ocen
- Najprostszy sposób: modyfikuj syntetyczny zbiór (np. dodaj kilka próbek z innymi wagami/noise) i ponownie uruchom `train_compare_multi.py`.
- Alternatywa (produkcyjna): dodaj oceny do bazy (endpointy aplikacji) i ponownie uruchom trening na danych z DB.

Następne kroki, które mogę wykonać teraz
- (A) Dodać widok w panelu admina pokazujący explain JSON w ładnej tabeli + wykresiki (MF + bar per_rule_contribution).  
- (B) Dodać skrypt do „online testów” — wstawia nowe oceny do bazy i od razu uruchamia szybki trening/eval, aby pokazać jak rekomendacje się zmieniają.  

Powiedz, którą z tych opcji chcesz, żebym implementował dalej (A lub B), albo czy mam od razu wygenerować dokument `.doc` z instrukcją i wynikami.  
