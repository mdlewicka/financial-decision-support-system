# Podsumowanie końcowej ewaluacji

## Financial Decision Support System

Niniejszy dokument przedstawia wyniki jednorazowej końcowej ewaluacji przeprowadzonej na wcześniej zamkniętym zbiorze testowym danych rynkowych.

Końcowa ewaluacja została wykonana dopiero po zamrożeniu wyboru modeli prognostycznych, strategii portfelowych oraz konfiguracji Reinforcement Learning.

Po otwarciu zbioru testowego nie wykonywano dodatkowej selekcji modeli, strojenia hiperparametrów, ponownej optymalizacji portfeli ani ponownego treningu modelu RL.

## Protokół ewaluacji

Projekt wykorzystuje chronologiczny podział danych na trzy części:

- `training` — uczenie modeli i konstrukcja strategii portfelowych;
- `validation` — rozwój systemu, porównanie modeli i podejmowanie decyzji projektowych;
- `test` — jednorazowa końcowa ewaluacja po zamrożeniu wszystkich decyzji.

Przed otwarciem zbioru testowego:

- cały projekt przeszedł kontrolę Ruff;
- zestaw testów non-RL zakończył się wynikiem 499 testów zaliczonych;
- zestaw testów RL i testów integracyjnych RL zakończył się wynikiem 59 testów zaliczonych;
- testy modułu finalnej ewaluacji zakończyły się powodzeniem;
- preflight finalnej ewaluacji zwrócił status `READY`;
- zamrożone artefakty analityczne zostały zapisane wraz z sumami kontrolnymi SHA256;
- przygotowano trwały mechanizm zabezpieczający przed wielokrotnym użyciem zbioru testowego.

Przed uzyskaniem dostępu do zbioru testowego utworzono manifest:

```text
benchmark_results/final_evaluation_frozen_manifest.json
```

z następującymi informacjami:

```text
status: frozen
test_split_used: false
frozen_at_utc: 2026-08-12T14:00:27.611264+00:00
```

Następnie końcowa ewaluacja została uruchomiona dokładnie jeden raz z jawnym potwierdzeniem:

```text
OPEN_FINAL_TEST_ONCE
```

Proces zakończył się poprawnie i utworzył pliki:

```text
benchmark_results/final_test/final_test_usage.json
benchmark_results/final_test/final_evaluation_result.json
```

Nie został utworzony plik `final_evaluation_failure.json`, co oznacza, że finalna ewaluacja zakończyła się bez błędu wykonania.

## Wyniki prognozowania

Wszystkie wcześniej zamrożone modele prognostyczne zostały ocenione na końcowym zbiorze testowym.

| Model | Liczba obserwacji | MAE | RMSE | R² | Trafność kierunku |
|---|---:|---:|---:|---:|---:|
| Random Forest | 4786 | 0.009504 | 0.014632 | 0.007772 | 54.53% |
| ARIMA walk-forward | 4786 | 0.009518 | 0.014706 | -0.002362 | 53.36% |
| Naiwny baseline zero-return | 4786 | 0.009534 | 0.014707 | -0.002547 | 0.42% |

Random Forest osiągnął najlepszy ogólny wynik spośród ocenianych modeli prognostycznych.

Przewaga modelu pozostaje jednak niewielka. W szczególności wartość R² na poziomie około `0.0078` oznacza, że model wyjaśnia jedynie bardzo małą część zmienności przyszłych jednodniowych stóp zwrotu.

Końcowy rezultat potwierdza więc konserwatywną interpretację przyjętą już na etapie walidacji: prognozowanie krótkoterminowych stóp zwrotu na rynkach finansowych pozostaje zadaniem trudnym, a generowane prognozy powinny być traktowane jako sygnał wspierający decyzję, a nie jako samodzielna podstawa działania.

## Wyniki klasycznych strategii portfelowych

Pięć wcześniej zdefiniowanych strategii portfelowych zostało ocenionych z wykorzystaniem wag zamrożonych przed otwarciem zbioru testowego.

Portfele optymalizowane nie były ponownie optymalizowane po uzyskaniu dostępu do danych testowych.

| Strategia | Annualizowany zwrot | Annualizowana zmienność | Sharpe | Sortino | VaR | CVaR | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Defensive | 16.63% | 10.56% | 1.388 | 2.044 | 1.04% | 1.54% | 8.01% |
| Equal Weight | 19.78% | 15.34% | 1.160 | 1.744 | 1.39% | 2.10% | 11.84% |
| Growth | 19.67% | 15.87% | 1.115 | 1.676 | 1.42% | 2.18% | 14.51% |
| Maximum Sharpe | 15.98% | 14.27% | 0.981 | 1.464 | 1.41% | 1.93% | 11.38% |
| Minimum Volatility | 17.13% | 10.65% | 1.422 | 2.037 | 1.06% | 1.62% | 9.11% |

Najważniejsze obserwacje z końcowej ewaluacji są następujące:

- `equal_weight` osiągnął najwyższy annualizowany zwrot wśród klasycznych strategii;
- `minimum_volatility` osiągnął najwyższy współczynnik Sharpe;
- `defensive` uzyskał najniższą zrealizowaną zmienność;
- `defensive` osiągnął również najmniejszy maximum drawdown;
- `maximum_sharpe`, mimo że został zoptymalizowany pod kątem współczynnika Sharpe na danych treningowych, nie osiągnął najwyższego Sharpe na końcowym zbiorze testowym.

Wyniki pokazują ograniczenia historycznej optymalizacji portfela oraz zależność rankingu strategii od warunków panujących w danym okresie rynkowym.

## Wynik Reinforcement Learning

Wcześniej wytrenowana polityka PPO została wczytana z zamrożonego artefaktu:

```text
benchmark_results/rl/ppo/ppo_allocation_agent.zip
```

Model nie był ponownie trenowany po otwarciu zbioru testowego.

Końcowe wyniki PPO przedstawiają się następująco:

| Metryka | Wynik |
|---|---:|
| Liczba obserwacji | 565 |
| Annualizowany średni zwrot | 28.93% |
| Annualizowana zmienność | 22.25% |
| Sharpe ratio | 1.211 |
| Sortino ratio | 2.649 |
| Value at Risk | 1.38% |
| Conditional Value at Risk | 2.07% |
| Maximum drawdown | 11.63% |
| Skumulowany zwrot | 81.67% |
| Końcowa wartość portfela | 1.8167 |
| Łączny koszt transakcyjny | 0.005061 |
| Średni turnover | 0.008958 |
| Łączna nagroda | -2.569404 |

PPO osiągnęło najwyższy annualizowany zwrot spośród ocenianych metod alokacji portfela oraz najwyższy współczynnik Sortino.

Jednocześnie PPO nie osiągnęło najwyższego współczynnika Sharpe oraz nie zapewniło najniższej zmienności ani najmniejszego obsunięcia kapitału.

Wynik nie stanowi więc dowodu na uniwersalną przewagę Reinforcement Learning nad klasycznymi metodami konstrukcji portfela.

Pokazuje natomiast, że zamrożona polityka PPO osiągnęła bardzo dobry wynik w końcowym okresie testowym, akceptując przy tym wyższy poziom zmienności.

Ujemna wartość `total_reward` nie oznacza ujemnej stopy zwrotu portfela. Funkcja nagrody w środowisku RL jest specyficznym sygnałem optymalizacyjnym, który uwzględnia również kary i ograniczenia środowiska, dlatego nie jest równoważna stopie zwrotu z inwestycji.

## Interpretacja końcowa

Końcowa ewaluacja potwierdza kilka głównych wniosków sformułowanych podczas rozwoju systemu.

Skuteczność prognozowania pozostaje ograniczona w wartościach bezwzględnych. Random Forest osiąga najlepszy wynik spośród testowanych modeli prognostycznych, jednak jego zdolność wyjaśniania zmienności przyszłych zwrotów pozostaje bardzo niewielka.

Klasyczne strategie portfelowe nadal wykazują istotne różnice pod względem relacji zwrotu do ryzyka. Proste i interpretowalne rozwiązania pozostają konkurencyjne, a cele optymalizacyjne ustalone na danych treningowych nie muszą prowadzić do najlepszych wyników na niewidzianych danych.

Polityka PPO osiągnęła bardzo dobry wynik w końcowym okresie testowym, szczególnie pod względem zwrotu oraz miar uwzględniających downside risk. Wspiera to traktowanie Reinforcement Learning jako obiecującego rozszerzenia eksperymentalnego systemu.

Nie uzasadnia to jednak zastąpienia klasycznych metod alokacji wyłącznie na podstawie dobrego wyniku PPO w pojedynczym końcowym okresie testowym.

## Wnioski dotyczące governance

Końcowy zbiór testowy został już wykorzystany i od tego momentu musi być traktowany jako zużyty.

Wyniki finalnego testu nie mogą być wykorzystywane do:

- dodatkowej selekcji modeli;
- strojenia hiperparametrów;
- ponownego wyboru rzędów ARIMA;
- ponownej optymalizacji portfeli;
- ponownego treningu PPO;
- przeprojektowania funkcji nagrody na podstawie wyniku testowego;
- wyboru strategii, a następnie ponownego przedstawiania jej wyniku jako niezależnej oceny testowej.

Każda przyszła zmiana metodologiczna wymagałaby nowego protokołu ewaluacji oraz nowych, rzeczywiście niewidzianych wcześniej danych.

Zapisane artefakty finalnego testu stanowią końcową ewaluację aktualnie zamrożonej konfiguracji projektu.
