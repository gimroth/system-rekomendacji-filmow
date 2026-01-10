#  Struktura Systemu ANFIS

---

## 1. Analiza Statystyczna (EDA) i Przygotowanie Danych
Przed budową modelu przeprowadzono kompleksową analizę EDA, która potwierdziła gotowość zbioru danych do procesów uczenia maszynowego.

* **Czyszczenie danych:** Dane z tabeli `ratings` zostały znormalizowane do zakresu [0, 1], aby zapewnić kompatybilność z funkcjami przynależności ANFIS.
* **Korelacja cech:** Analiza wykazała silną i równomierną korelację wszystkich aspektów (story, acting, visuals, sound, direction) z oceną ogólną na poziomie **0.84**. 
* **Wnioski z EDA:** Wysoka korelacja wewnętrzna między aspektami (0.72) potwierdza, że cechy te niosą podobną informację statystyczną, co uzasadnia redukcję wymiarowości wejść bez utraty jakości predykcji.
* **Rozwiązanie Cold Start:** Wykorzystanie wag z `user_preferences` jest merytorycznie uzasadnione, ponieważ wagi te bezpośrednio odpowiadają aspektom o udowodnionym silnym wpływie na ocenę końcową.

---

## 2. Dobór Liczby Reguł i Parametrów
Zdecydowano się na model oparty na **3 dynamicznych wejściach** i **27 regułach**.

### Uzasadnienie:
* **Optymalizacja statystyczna:** EDA wykazało, że dominujące gatunki w bazie to Dramat, Komedia i Thriller. Skupienie się na 3 kluczowych cechach pozwala modelowi na lepszą generalizację w tych najliczniejszych grupach.
* **Uniknięcie Klątwy Wymiarowości:** Ograniczenie liczby reguł z potencjalnych 243 (dla 5 cech) do 27 zapobiega zjawisku overfittingu, szczególnie istotnemu przy obserwowanym rozkładzie ocen, gdzie dominują wartości wysokie (4.0).
* **Personalizacja Dynamiczna:** `DataProcessor` wybiera dla użytkownika 3 aspekty o najwyższych wagach z onboardingu. Ponieważ korelacja każdego aspektu z oceną jest identyczna (0.84), system zachowuje tę samą precyzję niezależnie od wybranych cech.

---

## 3. Baza Reguł Rozmytych (Fuzzy Rule Base)
Poniższa tabela definiuje logikę "IF... THEN..." modelu. 
*Wejścia (X1, X2, X3) posiadają 3 funkcje przynależności: Niski (L), Średni (M), Wysoki (H).*

| ID | IF Cecha 1 (X1) | AND Cecha 2 (X2) | AND Cecha 3 (X3) | THEN Predykcja (Y) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | Wysoki (H) | Wysoki (H) | Wysoki (H) | **5.0** |
| 2 | Wysoki (H) | Wysoki (H) | Średni (M) | **4.5** |
| 3 | Wysoki (H) | Wysoki (H) | Niski (L) | **4.0** |
| 4 | Wysoki (H) | Średni (M) | Wysoki (H) | **4.5** |
| 5 | Wysoki (H) | Średni (M) | Średni (M) | **4.0** |
| 6 | Wysoki (H) | Średni (M) | Niski (L) | **3.5** |
| 7 | Wysoki (H) | Niski (L) | Wysoki (H) | **3.5** |
| 8 | Wysoki (H) | Niski (L) | Średni (M) | **3.0** |
| 9 | Wysoki (H) | Niski (L) | Niski (L) | **2.5** |
| 10 | Średni (M) | Wysoki (H) | Wysoki (H) | **4.0** |
| 11 | Średni (M) | Wysoki (H) | Średni (M) | **3.5** |
| 12 | Średni (M) | Wysoki (H) | Niski (L) | **3.0** |
| 13 | Średni (M) | Średni (M) | Wysoki (H) | **3.5** |
| 14 | Średni (M) | Średni (M) | Średni (M) | **3.0** |
| 15 | Średni (M) | Średni (M) | Niski (L) | **2.5** |
| 16 | Średni (M) | Niski (L) | Wysoki (H) | **3.0** |
| 17 | Średni (M) | Niski (L) | Średni (M) | **2.5** |
| 18 | Średni (M) | Niski (L) | Niski (L) | **2.0** |
| 19 | Niski (L) | Wysoki (H) | Wysoki (H) | **3.0** |
| 20 | Niski (L) | Wysoki (H) | Średni (M) | **2.5** |
| 21 | Niski (L) | Wysoki (H) | Niski (L) | **2.0** |
| 22 | Niski (L) | Średni (M) | Wysoki (H) | **2.5** |
| 23 | Niski (L) | Średni (M) | Średni (M) | **2.0** |
| 24 | Niski (L) | Średni (M) | Niski (L) | **1.5** |
| 25 | Niski (L) | Niski (L) | Wysoki (H) | **2.0** |
| 26 | Niski (L) | Niski (L) | Średni (M) | **1.5** |
| 27 | Niski (L) | Niski (L) | Niski (L) | **1.0** |

---

## 4. Wpływ na jakość rekomendacji

Zastosowana struktura zapewnia balans między precyzją (RMSE) a interpretowalnością modelu.
* **Uzasadnienie merytoryczne:** Analiza korelacji (0.84) oraz rozkładu gatunków (dominacja Dramatu i Komedii) pozwoliła na zaprojektowanie reguł, które najlepiej odzwierciedlają rzeczywiste zachowania ocenowe użytkowników w tej konkretnej bazie danych.