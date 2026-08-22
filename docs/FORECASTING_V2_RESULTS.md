# Financial Decision Support System — wyniki Forecasting V2

## Status i decyzja wykonawcza

Etap 1 został ukończony eksperymentalnie na danych rozwojowych. Powstał
powtarzalny, chronologiczny benchmark bez dostępu do zbioru testowego V1 oraz
walidacja ekonomiczna uwzględniająca koszt transakcyjny.

**Decyzja:** klasyfikator logistyczny dla horyzontu pięciu sesji zachowujemy
wyłącznie jako eksperymentalny sygnał informacyjny wspomagający interpretację.
Nie zastępuje on modeli V1, nie wyznacza wag portfela, nie zmienia mechanizmu
alokacji i nie jest dowodem rentowności strategii.

## Odpowiedź na pytanie o poprawę predykcji

Tak — uzyskano powtarzalną poprawę zdolności rozpoznawania klas kierunku w
horyzoncie pięciu sesji. Mediana balanced accuracy wzrosła z 0,333333 dla
baseline'u opartego na częstości historycznej do
0.394369. Mediana
poprawy wyniosła
0.061036
punktu, a model wygrał w
5/5
chronologicznych foldach. Bootstrapowa różnica balanced accuracy wyniosła
0.060745
z 95-procentowym przedziałem
[0.048758,
0.073772], a
prawdopodobieństwo przewagi nad baseline'em wyniosło
1.000.

Nie uzyskano natomiast stabilnej poprawy regresji zwrotu ani przewagi
ekonomicznej po kosztach. Lepsza predykcja klas nie jest więc utożsamiana z
lepszą decyzją inwestycyjną.

## Metodyka i ochrona przed leakage

- wykorzystano wyłącznie splity `training` i `validation`;
- wykonano 5 chronologicznych foldów expanding-window;
- cechy i modele dopasowywano wyłącznie na części treningowej każdego foldu;
- target nie przekraczał granicy części treningowej;
- test V1 odczytany: **nie**;
- znany wynik testu V1 wykorzystany do selekcji:
  **nie**;
- analiza ekonomiczna używa niepokrywających się okresów pięciosesyjnych;
- koszt zmiany pozycji wynosił
  0.0010;
- wszystkie decyzje mają `final_test_claim=false`.

## Audyt stabilności modeli V1

| Model | Średnie MAE | Mediana MAE | Średnia poprawa MAE wobec zero-return | Prawdopodobieństwo przewagi w bootstrapie |
|---|---:|---:|---:|---:|
| Zero-return | 0.01100529 | 0.00984422 | 0,000% | — |
| Random Forest V1 | 0.01102360 | 0.00985044 | -0.155% | 0.238 |

Audyt nie potwierdził stabilnej przewagi Random Forest V1 nad naiwną prognozą
zerowego zwrotu. Uzasadniało to zmianę horyzontu i zadania, ale nie usunięcie
prostych baseline'ów.

## Wyniki kandydatów V2

| Kandydat | Główna metryka | Mediana | Poprawa mediany | Wygrane foldy | Bootstrap P(lepszy) | Decyzja |
|---|---|---:|---:|---:|---:|---|
| Ridge, 1 sesja | MAE | 0.00984644 | -0.511% | 1/5 | 0.034 | Odrzucony |
| Ridge, 5 sesji | MAE | 0.02155744 | 0.348% | 3/5 | 0.115 | Nie promować |
| Logistic, 5 sesji | Balanced accuracy | 0.394369 | +0.061036 pkt | 5/5 | 1.000 | Tylko sygnał informacyjny |

Klasyfikator miał średnie coverage
30.88%, lecz jego
średnia screening utility była ujemna
(-0.000739). Ponadto
średni multiclass Brier score klasyfikatora
(0.662171) był
gorszy od baseline'u prior
(0.638318), dlatego
prawdopodobieństw nie należy przedstawiać jako skalibrowanego confidence.

## Walidacja ekonomiczna

Sprawdzono prostą politykę long/cash: pozycja długa tylko wtedy, gdy klasa `up`
ma najwyższe prawdopodobieństwo i przekracza stały próg; w przeciwnym razie
portfel pozostaje w gotówce. Nie stosowano krótkiej sprzedaży.

Dla domyślnego progu 0.45:

- średnie coverage: 10.34%;
- średni zwrot polityki: 0.000054;
- średni zwrot buy-and-hold: 0.053601;
- średnia nadwyżka: -0.053547;
- wygrane grupy fold-ticker: 12/40;
- bootstrap P(lepszy od buy-and-hold):
  0.026;
- 95-procentowy przedział nadwyżki:
  [-0.109135,
  0.000119];
- średnia redukcja maximum drawdown:
  10.41%.

Żaden z 6 sprawdzonych progów nie osiągnął dodatniej
średniej nadwyżki względem buy-and-hold (liczba dodatnich progów:
0). Redukcja obsunięcia wynikała głównie z niskiej
ekspozycji na rynek i nie rekompensowała utraconego zwrotu.

## Ostateczne decyzje

1. Random Forest V1 pozostaje historycznym benchmarkiem, ale nie ma podstaw do
   deklarowania jego stabilnej przewagi nad zero-return.
2. Ridge V2 dla jednego dnia zostaje odrzucony.
3. Ridge V2 dla pięciu sesji nie przechodzi bramki niepewności i nie jest
   integrowany.
4. Logistic V2 dla pięciu sesji wykazuje powtarzalny sygnał klasyfikacyjny,
   dlatego może później zasilać opis kontekstu kierunkowego z możliwością
   `abstain`.
5. Logistic V2 nie może wybierać wag, zastępować Decision Engine, zmieniać
   optymalizacji portfela ani być prezentowany jako rentowna strategia.
6. Nie wykonano jednorazowego testu na nowych, rzeczywiście niewidzianych
   danych; raport nie zawiera finalnego twierdzenia o przewadze poza okresem
   rozwojowym.

## Ograniczenia

- badanie obejmuje osiem instrumentów i dane dzienne;
- klasyfikacja rozwojowa wykorzystuje pięciosesyjne cele, a osobna walidacja
  ekonomiczna dopiero usuwa nakładanie okresów;
- rozkład klas, reżimy i relacje między aktywami mogą zmieniać się w czasie;
- nie potwierdzono kalibracji prawdopodobieństw;
- strategia long/cash jest celowo prosta i nie stanowi pełnego backtestu
  portfelowego;
- nie uwzględniono wszystkich realnych ograniczeń wykonania zleceń, podatków i
  poślizgu;
- końcowy test V1 był wcześniej znany, dlatego nie mógł być testem V2;
- wiarygodne twierdzenie o generalizacji wymaga późniejszego, zamrożonego testu
  na nowych danych.

## Reprodukcja

Raport powstaje wyłącznie z wersjonowanych artefaktów:

- `benchmark_results/forecast_v2/stability_audit/`;
- `benchmark_results/forecast_v2/model_benchmark/`;
- `benchmark_results/forecast_v2/economic_validation/`.

Polecenie:

```bash
python3 -m backend.src.forecast.forecast_v2_report
```

Wynik: `docs/FORECASTING_V2_RESULTS.md`.
