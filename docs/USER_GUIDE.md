**Szybki przewodnik — uruchomienie eksperymentu ANFIS vs MLP**

Ten dokument pokazuje krok po kroku, jak przygotować środowisko, uruchomić treningi, wygenerować porównania i podglądnąć wykresy lokalnie.

Środowisko: Windows, PowerShell. Zakładam, że pracujesz w katalogu repo: `C:/Users/piotr/Documents/GitHub/system-rekomendacji-filmow`.

1. Utwórz i aktywuj virtualenv

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -U pip
```

2. Zainstaluj zależności

```powershell
pip install -r requirements.txt
```

3. (Opcjonalnie) jeśli brakuje pakietów, zainstaluj np. torch:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4. Wygeneruj (jeśli potrzebne) dane syntetyczne i uruchom porównanie (ANFIS vs MLP)

```powershell
python train_compare_multi.py
```

Skrypt utworzy dwa zestawy wyników: `results_anfis_db_<timestamp>/` i `results_anfis_synthetic_<timestamp>/` oraz katalog porównawczy `comparison_anfis_mlp_multi_<timestamp>/` z plikiem `rmse_comparison.png` i `comparison_report.json`.

5. Wygeneruj dodatkowe wykresy (residuals, preds overlay)

```powershell
python scripts/generate_additional_plots.py
```

Wyniki znajdziesz w katalogach `results_anfis_db_<timestamp>/` i `results_anfis_synthetic_<timestamp>/` (pliki: `pred_vs_actual_overlay.png`, `residuals_vs_predicted_*.png`, `residuals_hist_*.png`, wykresy MF w `mf_fold_*/`).

6. (Test „online”) Wstaw kilka ocen i sprawdź metryki przed/po

```powershell
python scripts/insert_and_test_ratings.py --user-id 1 --n 10 --force
```

Skrypt wykona szybką ewaluację przed i po upsercie ocen i zapisze raporty w `results_online_test_<timestamp>/report.json`.

7. Jak obejrzeć wykresy lokalnie (przeglądarka)

Uruchom prosty serwer HTTP w katalogu repo i otwórz pliki w przeglądarce (port 8008):

```powershell
.\\venv\\Scripts\\python.exe -m http.server 8008
# potem w przeglądarce otwórz adresy np:
http://127.0.0.1:8008/comparison_anfis_mlp_multi_<timestamp>/rmse_comparison.png
http://127.0.0.1:8008/results_anfis_db_<timestamp>/pred_vs_actual_overlay.png
```

8. Gdzie szukać plików wyników

- Porównanie RMSE: `comparison_anfis_mlp_multi_<timestamp>/rmse_comparison.png`
- Raport porównawczy JSON: `comparison_anfis_mlp_multi_<timestamp>/comparison_report.json`
- Wyniki ANFIS (DB): `results_anfis_db_<timestamp>/` (z MF, CV plots, pred_vs_actual.png)
- Wyniki synthetic: `results_anfis_synthetic_<timestamp>/`

9. Szybkie debugowanie

- Jeśli skrypt się zatrzymuje przy braku pakietów, przeczytaj komunikat i zainstaluj brakujący moduł.
- Jeśli wstawianie ocen zwraca 0 wstawień, spróbuj dodać `--force` (upsert) lub użyć istniejącego `--user-id`.

10. Chcesz automatyzację (GitHub Actions)?

Mogę dodać prosty workflow, który uruchomi `train_compare_multi.py` i zapisze artefakty. Powiedz, czy chcesz, żeby działało na runnerze z GPU/CPU.

---
Plik ten utworzyłem w `docs/USER_GUIDE.md` — jeśli chcesz, przygotuję skróconą wersję `README.md` do umieszczenia w repo.
