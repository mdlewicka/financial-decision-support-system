# Financial Decision Support System V1 — zamrożony baseline

## Status

Wersja V1 jest ukończonym punktem odniesienia dla dalszego rozwoju Financial Decision Support System.
Jej modele, konfiguracje i końcowe wyniki nie mogą być zmieniane w celu poprawienia
wyników po obejrzeniu końcowego zbioru testowego.

- status manifestu: `frozen`;
- zamrożenie decyzji: `2026-08-12T14:00:27.611264+00:00`;
- jednorazowe otwarcie testu: `2026-08-12T14:11:21.966165+00:00`;
- zakres końcowej ewaluacji: `test`;
- definicja celu prognozy: zwrot prostej stopy zwrotu następnej sesji.

Źródłami prawdy są:

- `benchmark_results/final_evaluation_frozen_manifest.json`;
- `benchmark_results/final_test/final_evaluation_result.json`;
- `benchmark_results/final_test/final_test_usage.json`.

## Wyniki prognozowania V1 na teście końcowym

| Model | MAE | RMSE | R² | Trafność kierunku |
|---|---:|---:|---:|---:|
| Random Forest | 0,009504 | 0,014632 | 0,007772 | 54,53% |
| ARIMA walk-forward | 0,009518 | 0,014706 | -0,002362 | 53,36% |
| Naiwny | 0,009534 | 0,014707 | -0,002547 | 0,42%* |

\* Model naiwny prognozuje dokładnie zero. Jego metryka kierunku nie jest
porównywalnym klasyfikatorem wzrost/spadek, ponieważ prognozowany kierunek jest
oznaczany jako `flat`.

Random Forest osiągnął najlepszy wynik spośród trzech zamrożonych modeli, lecz
jego przewaga jest niewielka. Dodatnie R² nie jest dowodem trwałej przewagi
rynkowej. Wynik wymaga analizy stabilności na wielu oknach czasowych.

## Integralność zamrożonych artefaktów

Podczas audytu Etapu 0 zweryfikowano SHA-256 wszystkich ośmiu artefaktów
wymienionych w zamrożonym manifeście. Każdy plik był obecny i zgodny z zapisanym
hashem.

Chronione artefakty obejmują:

- prognozy naive, Random Forest i ARIMA;
- wybór rzędu ARIMA;
- wyniki i wagi portfeli;
- wynik PPO i zapisany model agenta.

## Reguły ochrony V1

1. Nie wolno używać testu V1 do wyboru cech, celu, modelu ani hiperparametrów V2.
2. Nie wolno nadpisywać katalogu `benchmark_results/final_test/` wynikami V2.
3. Nie wolno przedstawiać testu V1 jako niewidzianej ewaluacji modeli V2.
4. Każda zmiana metodologiczna należy do V2 i otrzymuje osobne artefakty.
5. Wyniki V1 pozostają raportowane także wtedy, gdy V2 okaże się lepsze.
6. Brak poprawy V2 jest prawidłowym wynikiem badawczym.

## Przeznaczenie baseline'u

V1 służy do:

- udokumentowania ukończonego projektu;
- porównania zakresu funkcjonalnego z V2;
- pokazania uczciwego wyniku końcowego osiągniętego przed rozpoczęciem nowych
  eksperymentów;
- ochrony projektu przed nieświadomym dostrajaniem do testu.
