# MovieRec - System Rekomendacji Filmów

Inteligentna platforma rekomendacyjna wykorzystująca model hybrydowy **ANFIS** do personalizacji treści na podstawie wielokryterialnej oceny aspektów filmu.

---

## 1. Struktura Projektu

Projekt został zaprojektowany w oparciu o architekturę modułową, oddzielającą logikę uczenia maszynowego od interfejsu API i warstwy prezentacji.

### Główne Katalogi i Pliki:

* **`app/`** – Serce aplikacji FastAPI.
    * **`ml/`** – Moduły odpowiedzialne za sztuczną inteligencję.
        * `anfis_model.py`: Implementacja logiki rozmytej (27 reguł, 3 wejścia).
        * `data_processor.py`: Przygotowanie cech (Dynamic Match Logic) i przetwarzanie danych z bazy.
        * `models/`: Katalog przechowujący wytrenowane modele w formacie `.pkl`.
    * **`models/`** – Definicje tabel bazy danych SQLAlchemy (User, Movie, Rating, Comment, Preference).
    * **`routers/`** – Endpointy API podzielone tematycznie (auth, admin, recommendations, movies).
    * **`database/`** – Konfiguracja połączenia z bazą danych PostgreSQL.
* **`templates/`** – Szablony HTML (Jinja2) dla frontendu.
* **`static/`** – Pliki statyczne (CSS Tailwind, JavaScript frontendowy, obrazy).
* **`train_anfis.py`** – Skrypt przeprowadzający proces uczenia modelu rozmytego z automatycznym doborem Top 3 cech.
* **`train_mlp.py`** – Skrypt trenujący sieć neuronową jako punkt odniesienia (baseline).
* **`compare_models.py`** – Skrypt generujący raporty porównawcze i wykresy wydajności ANFIS vs MLP.

---

## 2. Implementacja modelu ANFIS

System wykorzystuje model **Adaptive Neuro-Fuzzy Inference System**, który łączy interpretowalność logiki rozmytej z możliwością uczenia się sieci neuronowych.

### Kluczowe Elementy Systemu:

1.  **Dynamiczne Wejścia:** Model analizuje Top 3 aspekty filmu, które wykazują najwyższą korelację z preferencjami użytkownika (wybierane spośród: Fabuła, Aktorstwo, Wizualia, Dźwięk, Reżyseria).

2.  **Logika Match (Dopasowanie):** Wejścia modelu są obliczane jako stopień podobieństwa między profilami:  
    $$Match = 1.0 - |u_{val} - m_{val}|$$

3.  **Fuzzification (Rozmywanie):** 3 wejścia są mapowane na zbiory rozmyte (*Niski, Średni, Wysoki*) przy użyciu trójkątnych funkcji przynależności.

4.  **Baza Reguł:** Zgodnie z architekturą $3^3$, system operuje na **27 regułach logicznych** typu *IF-THEN*. Pozwala to na pełną interpretowalność – system dokładnie "wie", dlaczego dany film otrzymał konkretną notę.



---

## 3. Funkcjonalności Aplikacji

### Panel Użytkownika
* **Inteligentny Onboarding:** Formularz definiujący wagi aspektów, rozwiązujący problem "zimnego startu".
* **Rekomendacje "Dla Ciebie":** Lista generowana przez ANFIS z wizualnym wskaźnikiem pewności (Confidence Badge) oraz tekstowym uzasadnieniem.
* **Rankingi Top 100:** Globalna lista filmów z progiem wiarygodności (min. 500 głosów) i filtrowaniem gatunkowym.

### Panel Administratora
* **Zarządzanie Treścią:** Wyszukiwarka TMDB zintegrowana z bazą lokalną (oznaczanie filmów już posiadanych).
* **Zaawansowana Moderacja:** Administator ma możliwość zarządzania, filmami, użytkownikami, oraz komentarzami i ocenami.
* **ML Dashboard:** Podgląd metryk błędu (RMSE, MAE) i porównanie skuteczności modeli w czasie rzeczywistym.

---

## 4. Instalacja na nowym komputerze

Aby uruchomić aplikację lokalnie, wykonaj poniższe kroki:

### KROK 1: Pobranie kodu
```bash
# Sklonuj repozytorium za pomocą Git
git clone [https://github.com/gimroth/system-rekomendacji-filmow.git](https://github.com/gimroth/system-rekomendacji-filmow.git)

# Wejdź do folderu projektu
cd system-rekomendacji-filmow
```

### KROK 2: Przygotowanie środowiska
```bash
# Stworzenie środowiska wirtualnego
python -m venv .venv

# Aktywacja (Windows)
.venv\Scripts\activate

# Instalacja bibliotek
pip install -r requirements.txt
```

### KROK 3: Konfiguracja zmiennych (.env)
Stwórz plik .env w głównym folderze i uzupełnij:

```bash
DATABASE_URL=postgresql://uzytkownik:haslo@localhost:5432/nazwa_bazy
TMDB_API_KEY=twoj_klucz_api_z_tmdb
SECRET_KEY=twoj_klucz_jwt
ALGORITHM=HS256
```

### KROK 4: Inicjalizacja bazy i modeli

```bash
# 1. Import danych z CSV i wzbogacenie danymi z TMDB
python -m app.utils.data_management

# 2. Wytrenowanie modeli ML
python train_anfis.py
python train_mlp.py
python compare_models.py
```

### KROK 5: Uruchomienie aplikacji

```bash
uvicorn app.main:app --reload
```
Aplikacja będzie dostępna pod adresem: http://127.0.0.1:8000
