# Forecasting V2 — protokół eksperymentów

## Cel

Celem jest sprawdzenie, czy przewaga predykcyjna Financial Decision Support System może zostać
zwiększona w sposób powtarzalny, odporny na zmianę okresu i użyteczny po
uwzględnieniu kosztów. Celem nie jest znalezienie najwyższej pojedynczej liczby.

## Hipotezy

### H1 — cechy czasowe

Opóźnione zwroty, rolling volatility, rolling drawdown, trend i siła relatywna
zawierają stabilniejszy sygnał niż wyłącznie bieżący przekrój cech V1.

### H2 — horyzont

Skumulowany zwrot pięciosesyjny może mieć wyższy stosunek sygnału do szumu niż
zwrot następnej sesji.

### H3 — decyzja progowa

Klasyfikacja ruchu przekraczającego próg ekonomiczny oraz możliwość `abstain`
mogą być użyteczniejsze niż bezwarunkowa regresja każdego dziennego zwrotu.

### H4 — indukcyjne ograniczenie modelu

Regularizowany model liniowy lub umiarkowany gradient boosting może być
stabilniejszy od obecnego Random Forest w rolling validation.

## Zasada separacji danych

Końcowy test V1 został już otwarty i jest znany. Nie jest zbiorem testowym V2.

Zabronione jest użycie wyników z `benchmark_results/final_test/` do:

- wyboru cech;
- wyboru horyzontu;
- wyboru modelu;
- ustawienia hiperparametrów;
- wyboru progu decyzyjnego;
- wyboru tickera lub reżimu do raportowania;
- odrzucenia eksperymentu, który wygląda gorzej na teście V1.

Rozwój V2 wykorzystuje wyłącznie dane historycznie dostępne przed nowym,
zamrożonym okresem ewaluacyjnym. Wyniki V1 mogą być pokazane jedynie jako
historyczny punkt odniesienia.

## Dane i wersjonowanie

Każde uruchomienie zapisuje:

- wersję schematu eksperymentu;
- zakres dat;
- tickery;
- definicję celu;
- listę cech;
- granice foldów;
- konfigurację modelu;
- random seed;
- wersję kodu;
- fingerprint danych;
- metryki per fold i ticker;
- czas wykonania;
- artefakty predykcji.

## Etap 1 — audyt V1

Przed dodaniem modeli należy:

1. odtworzyć wyniki V1 na danych rozwojowych;
2. dodać baseline średniej historycznej i prostego momentum/last-sign;
3. zbadać rozkład celu, imbalance kierunku i autokorelację;
4. sprawdzić wyniki per ticker i okres;
5. obliczyć bootstrapowe przedziały różnic względem baseline;
6. wskazać reżimy trendu i zmienności bez wykorzystywania przyszłości.

## Etap 2 — cele

Eksperymenty obejmują:

1. regresję `next_session_simple_return` jako zgodność z V1;
2. regresję skumulowanego zwrotu pięciosesyjnego;
3. klasyfikację kierunku przekraczającego z góry ustalony próg;
4. regułę `abstain`, gdy prawdopodobieństwo albo oczekiwany ruch nie osiąga
   progu.

Próg musi wynikać z jawnego kosztu lub zostać ustalony wyłącznie na danych
rozwojowych.

## Etap 3 — cechy

Kandydaci:

- zwroty opóźnione o 1, 2, 5, 10 i 20 sesji;
- momentum i średnie dla kilku okien;
- rolling volatility;
- rolling drawdown;
- rolling volume z relacją do własnej historii;
- odległość od średnich kroczących;
- intraday range;
- siła relatywna instrumentu wobec rynku;
- opóźnione zwroty innych klas aktywów;
- deterministyczne, historycznie dostępne oznaczenie reżimu.

Każda cecha musi mieć test potwierdzający brak dostępu do przyszłości.

## Etap 4 — modele

Modele V1:

- naive;
- ARIMA walk-forward;
- Random Forest.

Maksymalnie dwa nowe podejścia w pierwszej iteracji:

- regularizowany model liniowy/logistyczny;
- gradient boosting albo klasyfikator kierunku.

Opcjonalny ensemble może powstać dopiero po ocenie różnorodności błędów.

Deep learning nie jest częścią pierwszej iteracji V2.

## Walidacja

Stosujemy expanding-window albo rolling-origin evaluation z co najmniej pięcioma
chronologicznymi foldami, jeśli zakres danych na to pozwala.

Dla każdego foldu:

- trening kończy się przed walidacją;
- transformacje są dopasowane tylko na treningu;
- cechy nie przekraczają granicy czasu;
- target nie może przekraczać granicy splitu;
- próg i hiperparametry nie są dobierane na ocenianym foldzie.

## Metryki

Regresja:

- MAE;
- RMSE;
- R²;
- korelacja Spearmana i information coefficient;
- błąd względem baseline.

Klasyfikacja:

- balanced accuracy;
- precision i recall dla ruchów użytecznych;
- ROC-AUC, gdy obie klasy są obecne;
- Brier score i kalibracja prawdopodobieństwa;
- coverage dla polityki `abstain`.

Użyteczność ekonomiczna:

- zwrot po kosztach;
- Sharpe i Sortino;
- maximum drawdown;
- turnover;
- stabilność per fold i ticker.

Directional accuracy modelu prognozującego zawsze zero nie jest traktowana jako
pełnoprawna metryka klasyfikacyjna.

## Niepewność i stabilność

Raportujemy:

- średnią i medianę per fold;
- odchylenie i najgorszy fold;
- przedziały bootstrapowe różnic względem baseline;
- liczbę foldów, w których kandydat wygrywa;
- wyniki per ticker;
- wyniki per reżim;
- wpływ kosztów.

## Bramki akceptacji kandydata

Kandydat może przejść do integracji tylko wtedy, gdy łącznie:

1. nie narusza testów leakage;
2. poprawa nie pochodzi z jednego folda lub jednego tickera;
3. mediana głównej metryki jest lepsza od odpowiedniego baseline;
4. kandydat wygrywa w większości foldów albo wykazuje wyraźną, uzasadnioną
   specjalizację;
5. wynik po kosztach nie znika całkowicie;
6. złożoność jest proporcjonalna do poprawy;
7. wyniki i artefakty są w pełni odtwarzalne.

Spełnienie bramek rozwojowych nie oznacza jeszcze ogłoszenia przewagi na nowych
danych.

## Nowa końcowa ewaluacja

Nowy test V2 może zostać utworzony wyłącznie z danych rzeczywiście niewidzianych
podczas rozwoju V2, np. z późniejszego okresu dostępnego po zamrożeniu modelu.

Przed jego otwarciem należy:

- zamrozić kod, cechy, modele, progi i konfigurację;
- zapisać hashe artefaktów;
- zapisać kryteria sukcesu;
- uruchomić test jednokrotnie;
- opublikować także wynik negatywny.

## Kryterium ukończenia Etapu 1

Etap Forecasting V2 jest ukończony, gdy:

- wykonano zaplanowane eksperymenty bez leakage;
- powstał reprodukowalny raport stabilności;
- świadomie zaakceptowano najlepszy stabilny kandydat albo zachowano V1;
- nie użyto poznanego testu V1 do rozwoju;
- ograniczenia zostały opisane.
