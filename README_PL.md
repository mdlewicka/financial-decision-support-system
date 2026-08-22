# Financial Decision Support System

Financial Decision Support System to aplikacja oparta na AI/ML, przeznaczona do reprodukowalnej analizy finansowej i wspomagania decyzji.

System integruje prognozowanie finansowe, explainable machine learning, analizę ryzyka, optymalizację portfela, Retrieval-Augmented Generation, wyspecjalizowanych agentów AI oraz eksperymentalny komponent alokacji wykorzystujący Reinforcement Learning w ramach jednej architektury wspomagania decyzji.

Aplikacja została zaprojektowana do wspierania interpretacji i porównywania scenariuszy finansowych. Nie udziela spersonalizowanych porad inwestycyjnych i nie traktuje wyników modeli jako deterministycznych rekomendacji.


Dokumentacja projektu:

- [Architektura](docs/ARCHITECTURE.md)
- [Bezpieczeństwo i koszty](docs/SECURITY_AND_COSTS.md)
- [Instrukcja kontenerowa](docs/CONTAINER_RUNBOOK.md)
- [Instrukcja CI](docs/CI_RUNBOOK.md)
- [Instrukcja odporności](docs/RESILIENCE_RUNBOOK.md)
- [Protokół eksperymentu Forecasting V2](docs/FORECASTING_V2_EXPERIMENT_PROTOCOL.md)
- [Wyniki Forecasting V2](docs/FORECASTING_V2_RESULTS.md)
- [Baseline V1](docs/V1_BASELINE.md)

## Szybkie uruchomienie przez Docker

Docker jest rekomendowanym sposobem uruchamiania projektu na innym komputerze, ponieważ odtwarza środowisko aplikacji bez zależności od lokalnego środowiska wirtualnego autora projektu.

Wymagania:

- Docker Desktop lub kompatybilny Docker Engine;
- Docker Compose v2 (`docker compose`).

W katalogu głównym projektu zbuduj obrazy i uruchom oba serwisy:

```bash
docker compose build
docker compose up --detach
docker compose ps
```

Otwórz aplikację:

- frontend Streamlit: `http://127.0.0.1:8501`
- backend FastAPI: `http://127.0.0.1:8000`

Opcjonalna kontrola zdrowia usług:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/system/readiness
curl -fsS http://127.0.0.1:8501/_stcore/health
```

Poprawna odpowiedź readiness backendu zawiera `"status":"ready"`, a endpoint zdrowia Streamlit zwraca `ok`.

Zatrzymaj i usuń lokalne kontenery oraz sieć Compose poleceniem:

```bash
docker compose down
```

Podstawowy workflow analityczny nie wymaga Ollama. Gdy opcjonalna funkcjonalność lokalnego LLM jest niedostępna, aplikacja korzysta z zaimplementowanego trybu zdegradowanego/fallback.

Dodatkowe informacje o kontenerach znajdują się w [`docs/CONTAINER_RUNBOOK.md`](docs/CONTAINER_RUNBOOK.md).

## Cel projektu

Celem projektu jest zbudowanie reprodukowalnego i interpretowalnego systemu wspomagania decyzji finansowych, który łączy analizę ilościową z wyjaśnieniami wspieranymi przez AI.

System został zaprojektowany zgodnie z następującymi zasadami:

- reprodukowalne przetwarzanie danych rynkowych;
- chronologiczne rozdzielenie danych treningowych, walidacyjnych i testowych;
- jawna ochrona przed leakage temporalnym;
- porównanie z transparentnymi modelami bazowymi;
- interpretowalne wyniki modeli;
- analiza portfela uwzględniająca ryzyko;
- odpowiedzi RAG oparte na możliwych do zweryfikowania źródłach;
- ograniczony zakres odpowiedzialności agentów AI;
- kontrolowane eksperymentalne wykorzystanie Reinforcement Learning;
- jawne przedstawianie ograniczeń modeli i pochodzenia wyników;
- rozdzielenie obliczeń analitycznych od warstwy prezentacji.

Końcowy zbiór testowy pozostawał zamknięty przez cały okres rozwoju i selekcji modeli. Został otwarty dokładnie jeden raz dopiero po zamrożeniu decyzji analitycznych i aplikacyjnych.

Końcowa ewaluacja została udokumentowana w plikach:

- `benchmark_results/final_test/final_evaluation_summary_en.md`;
- `benchmark_results/final_test/final_evaluation_summary_pl.md`.

## Architektura systemu

Aplikacja webowa wykorzystuje architekturę warstwową:

```text
Frontend Streamlit
        │
        ▼
HTTP / JSON
        │
        ▼
Aplikacja FastAPI
        │
        ▼
Usługi aplikacyjne, analityczne i dowodowe
        │
        ├── Prognozowanie
        ├── Explainability
        ├── Analiza ryzyka
        ├── Optymalizacja portfela
        ├── Deterministyczny Decision Engine
        ├── Retrieval-Augmented Generation
        ├── Agenci lokalni z guardrails
        ├── Market Intelligence
        ├── Audytowalne raportowanie
        ├── Obserwowalność i runtime status
        └── Eksperymentalny Reinforcement Learning
```

Frontend nie implementuje ponownie obliczeń finansowych ani machine-learningowych. Wykorzystuje ustrukturyzowane wyniki udostępniane przez backend API.

Takie rozdzielenie zapewnia reprodukowalność warstwy analitycznej i pozwala interfejsowi użytkownika koncentrować się na prezentacji, interpretacji oraz interakcji.

Pełne diagramy komponentów, wdrożenia i przepływu decyzji znajdują się w
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Aplikacja webowa

Interfejs Streamlit jest zorganizowany w dziewięć głównych modułów analitycznych.

### Dashboard

Dashboard zapewnia syntetyczny przegląd bieżącego stanu analitycznego systemu.

Prezentuje:

- wskaźniki jakości prognoz;
- miary ryzyka i optymalizacji portfela;
- podsumowania portfela;
- dostępność komponentów analitycznych;
- ostrzeżenia dotyczące jakości modeli;
- informacje o zbiorze ewaluacyjnym;
- alerty związane ze słabą skutecznością prognoz lub istotnym ryzykiem portfela.

Dashboard korzysta z istniejących wyników API i nie wykonuje dodatkowych obliczeń finansowych.

### Prognozowanie

Moduł Forecasting prezentuje prognozy stopy zwrotu dla kolejnej sesji.

Aktualny benchmark obejmuje:

- naiwny baseline zero-return;
- ARIMA;
- Random Forest.

Ewaluacja obejmuje:

- MAE;
- RMSE;
- R²;
- trafność kierunku;
- zbiorcze wyniki walidacyjne;
- wyniki dla poszczególnych tickerów;
- informacje o prognozach na poziomie pojedynczych obserwacji.

Zmienną prognozowaną jest prosta stopa zwrotu dla kolejnej sesji.

Rozwój i porównanie modeli odbywały się z wykorzystaniem zbioru walidacyjnego.

Po zamrożeniu decyzji projektowych wybrane konfiguracje modeli zostały ocenione dokładnie jeden raz na końcowym zbiorze testowym. Wyniki testowe są traktowane jako końcowa ewaluacja i nie są wykorzystywane do dalszej selekcji ani strojenia modeli.

### Ryzyko i scenariusze

Moduł Risk and Scenarios prezentuje informacje o ryzyku rynkowym i portfelowym.

Zaimplementowana warstwa analityczna obejmuje:

- annualizowaną stopę zwrotu;
- annualizowaną zmienność;
- Value at Risk;
- Conditional Value at Risk;
- maximum drawdown;
- analizę korelacji;
- miary koncentracji;
- efektywną liczbę aktywów;
- miary dywersyfikacji;
- predefiniowane scenariusze portfelowe.

Interfejs umożliwia porównywanie alternatywnych profili portfela oraz analizę szczegółowych charakterystyk ryzyka.

### Optymalizacja

Moduł Optimization prezentuje wyniki konstrukcji i alokacji portfela.

Dostępne strategie obejmują:

- defensive allocation;
- equal-weight allocation;
- growth allocation;
- maximum-Sharpe allocation;
- minimum-volatility allocation.

Interfejs prezentuje:

- metryki walidacyjne;
- wagi portfela;
- wkład poszczególnych aktywów w stopę zwrotu;
- wkład poszczególnych aktywów w zmienność;
- skumulowane stopy zwrotu portfela;
- porównanie wyników training-versus-validation.

Wizualizacje portfelowe wykorzystują Plotly do tworzenia interaktywnych wykresów analitycznych.

Strategie optymalizowane są konstruowane na danych treningowych i oceniane na danych walidacyjnych.

W końcowej ewaluacji wykorzystano wcześniej zamrożone wagi portfeli bez ponownej optymalizacji.

### Decision Workspace

Human-in-the-loop Decision Workspace przekształca typowany profil użytkownika i
jawne założenia w deterministyczny ranking zamrożonych scenariuszy portfelowych.
Pokazuje naruszenia ograniczeń, znormalizowane składowe wyniku, kompromisy,
provenance oraz porównanie przed i po zmianie założeń.

Opcjonalny lokalny LLM wyjaśnia już policzony wynik. Nie może wybierać wag,
zmieniać faktów analitycznych ani zastępować finalnej decyzji użytkownika. Przy
niedostępności Ollama workflow zachowuje deterministyczny fallback.

### Market Intelligence

Market Intelligence udostępnia mały snapshot bieżącego kontekstu: ostatnią
dostępną cenę, dzienną zmianę, timestamp UTC, źródło, strefę rynku oraz status
świeżości/cache. Obsługuje ręczne odświeżanie i jawny tryb offline. Nie jest
streamingiem ani usługą wykonywania zleceń.

### AI Insights

Moduł AI Insights łączy Retrieval-Augmented Generation oraz wyspecjalizowanych agentów analitycznych.

Komponent RAG dostarcza odpowiedzi oparte na kontrolowanej finansowej bazie wiedzy.

Warstwa agentowa udostępnia wyspecjalizowane workflow dla:

- interpretacji prognoz;
- interpretacji ryzyka;
- interpretacji alokacji portfela.

Odpowiedzi agentów zawierają ustrukturyzowane sekcje, takie jak:

- summary;
- findings;
- risks;
- recommendations;
- limitations;
- evidence.

Zapytania generatywne wykorzystują dłuższy timeout frontendu niż zwykłe zapytania analityczne API, ponieważ lokalna inferencja LLM może wymagać więcej czasu.

### Explainability and Governance

Moduł Explainability and Governance zapewnia interpretację modeli oraz kontrolę metodologiczną.

Aktualny artefakt XAI koncentruje się na modelu Random Forest i obejmuje:

- globalną ważność cech SHAP;
- lokalne wyjaśnienia SHAP;
- Partial Dependence;
- Accumulated Local Effects;
- reprezentatywne obserwacje lokalne;
- metadane modelu i artefaktów;
- jawne ograniczenia interpretacyjne.

Interaktywne wykresy Plotly są wykorzystywane do wizualizacji SHAP, PDP i ALE.

Moduł jawnie rozróżnia wyjaśnienie predykcyjne od wnioskowania przyczynowego.

SHAP, PDP i ALE opisują zachowanie modelu i nie mogą być interpretowane jako dowód zależności przyczynowych na rynku.

### Raporty

Moduł Reports generuje reprodukowalne podsumowania analityczne na podstawie istniejących wyników API.

Raporty zawierają informacje z obszarów:

- metadanych systemu;
- prognozowania;
- explainability;
- analizy ryzyka;
- optymalizacji portfela;
- RAG;
- agentów AI.

Wygenerowane raporty można pobrać w formatach:

- Markdown;
- JSON.

Generowanie raportów nie powoduje ponownego trenowania modeli ani przeliczania wyników analitycznych. Moduł agreguje istniejące ustrukturyzowane wyniki backendu.

## Pipeline danych rynkowych

Workflow danych rynkowych obsługuje obecnie osiem instrumentów:

| Ticker | Reprezentacja |
|---|---|
| `EPOL` | polskie akcje |
| `VGK` | europejskie akcje |
| `SPY` | amerykańskie akcje large-cap |
| `VT` | globalne akcje |
| `QQQ` | Nasdaq-100 |
| `GLD` | złoto |
| `BTC-USD` | Bitcoin |
| `AGG` | amerykańskie obligacje agregatowe |

Pipeline obejmuje:

- pobieranie historycznych danych przez Yahoo Finance;
- standaryzację schematu;
- walidację danych;
- sortowanie chronologiczne;
- feature engineering;
- walidację minimalnej liczby obserwacji;
- jawne daty targetu;
- konstrukcję stopy zwrotu dla kolejnej sesji;
- chronologiczny podział na training, validation i test;
- ochronę przed temporal leakage;
- metadane reprodukowalności.

Workflow danych został zaprojektowany tak, aby daty targetu nie przekraczały granic pomiędzy splitami.

## Metodologia prognozowania

Problem prognozowania został zdefiniowany jako przewidywanie prostej stopy zwrotu dla kolejnej sesji.

Porównanie obejmuje trzy rodziny modeli:

```text
Naive baseline
ARIMA
Random Forest
```

Model naiwny prognozuje zerową stopę zwrotu i pełni rolę transparentnego punktu odniesienia.

Random Forest jest również wykorzystywany przez warstwę XAI.

Proces ewaluacji wykorzystuje dane walidacyjne do rozwoju i porównywania modeli.

Końcowy zbiór testowy został celowo wyłączony z selekcji modeli i rozwoju interfejsu. Po zamrożeniu konfiguracji modeli został wykorzystany dokładnie jeden raz do końcowej ewaluacji out-of-sample.

Wyniki finalnego testu nie mogą być wykorzystywane do dodatkowej selekcji modeli, strojenia hiperparametrów ani ponownego wyboru rzędów ARIMA.

### Końcowe wyniki prognozowania

Zamrożone konfiguracje modeli zostały ocenione dokładnie jeden raz na końcowym zbiorze testowym.

| Model | Obserwacje | MAE | RMSE | R² | Trafność kierunku |
|---|---:|---:|---:|---:|---:|
| Random Forest | 4786 | 0.009504 | 0.014632 | 0.007772 | 54.53% |
| ARIMA walk-forward | 4786 | 0.009518 | 0.014706 | -0.002362 | 53.36% |
| Naiwny baseline zero-return | 4786 | 0.009534 | 0.014707 | -0.002547 | 0.42% |

Random Forest osiągnął najlepszy ogólny wynik spośród ocenionych modeli prognostycznych.

Jego przewaga pozostaje jednak niewielka. Wartość R² na zbiorze testowym pozostaje bliska zeru, co wspiera konserwatywną interpretację skuteczności prognozowania krótkoterminowych zwrotów finansowych.

Prognozy powinny być zatem traktowane jako pomocnicze sygnały analityczne, a nie jako samodzielne reguły inwestycyjne.

## Explainable AI

Aktywny workflow explainability wykorzystuje:

```text
SHAP
Partial Dependence
Accumulated Local Effects
```

Globalne wyjaśnienia identyfikują cechy najsilniej związane z wynikiem modelu.

Lokalne wyjaśnienia pokazują, w jaki sposób poszczególne cechy wpływają na jedną reprezentatywną prognozę.

Partial Dependence i ALE służą do analizy reakcji modelu na zmiany wartości cech.

Obowiązują następujące ograniczenia metodologiczne:

- wartości SHAP opisują zachowanie modelu, a nie przyczynowość;
- PDP może być zniekształcone przy silnej korelacji predyktorów;
- ALE ogranicza część problemów związanych z korelacją, ale nadal pozostaje techniką interpretacji modelu;
- dobra jakość wyjaśnień nie kompensuje słabej jakości predykcyjnej;
- wyjaśnienia walidacyjne nie gwarantują stabilności modelu w przyszłości.

Workflow XAI bazuje na wcześniej wybranej konfiguracji Random Forest i nie wykorzystuje końcowego zbioru testowego do selekcji modelu.

## Analiza portfela i ryzyka

Ewaluacja portfela obejmuje strategie regułowe oraz optymalizowane.

Framework analityczny obejmuje:

```text
oczekiwane stopy zwrotu
estymację macierzy kowariancji
optymalizację w stylu Markowitza
wagi portfela
annualizowane ryzyko i zwrot
Sharpe ratio
Sortino ratio
maximum drawdown
VaR
CVaR
koncentrację
efektywną liczbę aktywów
diversification ratio
wkłady poszczególnych aktywów
skumulowane stopy zwrotu portfela
```

Parametry portfeli optymalizowanych są wyznaczane na danych treningowych.

Główne wyniki aplikacji są raportowane dla zbioru walidacyjnego.

Takie rozdzielenie umożliwia sprawdzenie, czy strategia pozostaje stabilna poza okresem wykorzystanym do jej konstrukcji.

W jednorazowej końcowej ewaluacji wagi portfeli skonstruowane przed dostępem do testu zostały zamrożone i zastosowane bezpośrednio do okresu testowego bez ponownej optymalizacji.

### Końcowe wyniki klasycznych strategii portfelowych

| Strategia | Annualizowany zwrot | Annualizowana zmienność | Sharpe | Sortino | VaR | CVaR | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Defensive | 16.63% | 10.56% | 1.388 | 2.044 | 1.04% | 1.54% | 8.01% |
| Equal Weight | 19.78% | 15.34% | 1.160 | 1.744 | 1.39% | 2.10% | 11.84% |
| Growth | 19.67% | 15.87% | 1.115 | 1.676 | 1.42% | 2.18% | 14.51% |
| Maximum Sharpe | 15.98% | 14.27% | 0.981 | 1.464 | 1.41% | 1.93% | 11.38% |
| Minimum Volatility | 17.13% | 10.65% | 1.422 | 2.037 | 1.06% | 1.62% | 9.11% |

Końcowe wyniki pokazują, że:

- `equal_weight` osiągnął najwyższy annualizowany zwrot spośród strategii klasycznych;
- `minimum_volatility` osiągnął najwyższy współczynnik Sharpe;
- `defensive` osiągnął najniższą zrealizowaną zmienność;
- `defensive` osiągnął również najmniejszy maximum drawdown;
- `maximum_sharpe` nie utrzymał najwyższego współczynnika Sharpe na niewidzianych danych testowych.

Wyniki ilustrują ograniczenia historycznej optymalizacji portfela oraz wrażliwość rankingu strategii na zmieniające się warunki rynkowe.

## Retrieval-Augmented Generation

Kompaktowa implementacja RAG obsługuje:

- deterministyczne przetwarzanie dokumentów;
- embeddingi Sentence Transformer;
- wyszukiwanie wektorowe FAISS;
- wyszukiwanie leksykalne BM25;
- hybrid retrieval;
- reranking;
- lokalne generowanie odpowiedzi przez Ollama;
- cytowane evidence;
- benchmark evaluation;
- metryki retrieval;
- artefakty benchmarkowe w formatach maszynowych.

Aktualny korpus rozwojowy obejmuje niewielki kontrolowany zbiór dokumentów finansowych oraz ręcznie przygotowany golden dataset.

Obecny benchmark służy przede wszystkim do rozwoju systemu i porównania konfiguracji.

Oceniane strategie retrieval to:

```text
Vector
Hybrid
Reranked
```

Na obecnym benchmarku wszystkie trzy konfiguracje osiągnęły taki sam zbiorczy wynik jakości retrieval. Bardziej złożone konfiguracje nie wykazały mierzalnej przewagi jakościowej na niewielkim kontrolowanym korpusie.

Aktualny benchmark RAG pozostaje ewaluacją rozwojową opartą na kontrolowanym golden dataset. Nie był częścią jednorazowej ewaluacji market test split wykorzystywanej do oceny prognozowania, portfela i RL.

Szersza ewaluacja RAG wymagałaby rozszerzonego i niezależnie przygotowanego korpusu ewaluacyjnego.

Warstwa generatywna została skonfigurowana tak, aby odpowiadać w języku pytania użytkownika i opierać odpowiedź na odnalezionych źródłach.

## Agenci AI

System zawiera kontrolowaną warstwę agentową opartą na wyspecjalizowanych zadaniach analitycznych.

Aktualne workflow agentowe obejmują:

```text
Forecast Agent
Risk Agent
Allocation Agent
```

Każdy agent otrzymuje ograniczone zadanie analityczne i zwraca ustrukturyzowany wynik.

Wspólny kontrakt odpowiedzi obejmuje:

```text
agent_name
question
summary
findings
risks
recommendations
limitations
evidence
```

Architektura celowo unika nieograniczonego autonomicznego agent swarm.

Agenci mają interpretować istniejące dowody analityczne, a nie zastępować deterministyczne obliczenia finansowe.

Zapisane artefakty agentowe zostały wygenerowane na podstawie danych walidacyjnych i jawnie wskazują, że końcowy test split nie został wykorzystany do rozwoju agentów.

## Rozszerzenie Reinforcement Learning

Repozytorium zawiera eksperymentalne rozszerzenie alokacji portfela oparte na Proximal Policy Optimization.

Środowisko RL obsługuje:

- dynamiczną alokację portfela;
- koszty transakcyjne;
- śledzenie turnover;
- ograniczenia maksymalnej wagi aktywa;
- kary za drawdown;
- śledzenie wartości portfela;
- reprodukowalne trajektorie ewaluacyjne.

Polityka PPO jest trenowana na danych treningowych i została początkowo oceniona na danych walidacyjnych.

Rozszerzenie RL jest traktowane jako eksperymentalny komponent alokacji portfela, a nie jako zamiennik głównych deterministycznych workflow portfelowych i ryzyka.

### Końcowy wynik PPO

Model PPO został zamrożony przed dostępem do finalnego zbioru testowego i wczytany z wcześniej zapisanego artefaktu bez ponownego treningu.

Końcowe metryki PPO:

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

PPO osiągnęło najwyższy annualizowany zwrot spośród ocenianych podejść do alokacji oraz najwyższy współczynnik Sortino.

Nie osiągnęło jednak najwyższego współczynnika Sharpe ani najniższej zmienności czy najmniejszego maximum drawdown.

Wynik wspiera więc traktowanie PPO jako obiecującego rozszerzenia eksperymentalnego, ale nie stanowi dowodu jego uniwersalnej przewagi nad prostszymi i bardziej interpretowalnymi klasycznymi metodami portfelowymi.

Ujemna wartość `total_reward` nie oznacza ujemnej stopy zwrotu portfela. Reward w RL jest specyficznym dla środowiska sygnałem optymalizacyjnym, który uwzględnia kary i nie jest równoważny stopie zwrotu z inwestycji.

## Końcowa ewaluacja

Projekt wykorzystuje kontrolowany workflow jednorazowej finalnej ewaluacji.

Przed otwarciem testu:

- zamrożono modele i strategie;
- finalne artefakty benchmarkowe zostały zabezpieczone fingerprintami SHA256;
- utworzono frozen manifest;
- cały projekt przeszedł Ruff;
- zestaw testów non-RL zakończył się wynikiem 499 passed;
- zestaw testów RL i testów integracyjnych RL zakończył się wynikiem 59 passed;
- testy finalnej ewaluacji zakończyły się powodzeniem;
- dry-run preflight zwrócił status `READY`.

Frozen manifest znajduje się w:

```text
benchmark_results/final_evaluation_frozen_manifest.json
```

Następnie finalny market test split został otwarty dokładnie jeden raz przy użyciu kontrolowanego executora.

Poprawnie zakończone uruchomienie utworzyło:

```text
benchmark_results/final_test/final_test_usage.json
benchmark_results/final_test/final_evaluation_result.json
benchmark_results/final_test/forecast/
benchmark_results/final_test/portfolio/
benchmark_results/final_test/rl/
```

Plik `final_evaluation_failure.json` nie został utworzony.

Podsumowania przeznaczone do odczytu przez człowieka są dostępne w:

```text
benchmark_results/final_test/final_evaluation_summary_en.md
benchmark_results/final_test/final_evaluation_summary_pl.md
```

Końcowy zbiór testowy jest obecnie traktowany jako zużyty.

Jego wyniki nie mogą być wykorzystywane do:

- dalszej selekcji modeli;
- strojenia hiperparametrów;
- selekcji rzędów ARIMA;
- ponownej optymalizacji portfela;
- ponownego treningu PPO;
- przeprojektowywania funkcji nagrody;
- wyboru strategii, a następnie przedstawiania tego samego wyniku testowego jako niezależnego potwierdzenia.

Każda przyszła zmiana metodologiczna wymaga nowego protokołu ewaluacji i rzeczywiście niewidzianych wcześniej danych.

## Notebooki analityczne

Projekt zawiera sześć notebooków analitycznych:

```text
01_data_exploration.ipynb
02_forecasting.ipynb
03_risk_and_portfolio.ipynb
04_explainable_ai.ipynb
05_llm_rag_agents.ipynb
06_reinforcement_learning.ipynb
```

Notebooki dokumentują główne komponenty analityczne projektu.

Ich role:

- `01_data_exploration.ipynb` — eksploracja danych i struktura datasetu;
- `02_forecasting.ipynb` — modele bazowe i porównanie prognoz;
- `03_risk_and_portfolio.ipynb` — analiza ryzyka i alokacji portfela;
- `04_explainable_ai.ipynb` — interpretacja SHAP, PDP i ALE;
- `05_llm_rag_agents.ipynb` — benchmark RAG i architektura agentowa;
- `06_reinforcement_learning.ipynb` — eksperyment PPO i porównanie ze strategiami klasycznymi.

Notebooki przede wszystkim analizują reprodukowalne artefakty backendu zamiast ponownie implementować logikę analityczną.

## API

Aplikacja FastAPI udostępnia możliwości analityczne przez wersjonowane endpointy.

Główne grupy endpointów obejmują:

```text
/api/v1/forecast
/api/v1/xai
/api/v1/risk
/api/v1/portfolio
/api/v1/rag
/api/v1/agents
/api/v1/decision
/api/v1/reports
/api/v1/market-intelligence
/api/v1/system
```

Tytuł aplikacji:

```text
Financial Decision Support API
```

FastAPI stanowi granicę pomiędzy analitycznym backendem a interfejsem Streamlit.

## Stack technologiczny

Projekt wykorzystuje:

```text
Python 3.12
FastAPI
Streamlit
Pandas
NumPy
scikit-learn
statsmodels
SHAP
Plotly
FAISS
Sentence Transformers
BM25
Ollama
PyTorch
Gymnasium
Stable-Baselines3
Pytest
Ruff
Docker Compose
GitHub Actions
```

## Struktura projektu

```text
project-root/
├── backend/
│   └── src/
│       ├── agents/
│       ├── api/
│       ├── decision/
│       ├── domain/
│       ├── final_evaluation/
│       ├── forecast/
│       ├── market_intelligence/
│       ├── models/
│       ├── observability/
│       ├── pipelines/
│       ├── portfolio/
│       ├── providers/
│       ├── rag/
│       ├── reporting/
│       ├── risk/
│       ├── rl/
│       ├── services/
│       ├── storage/
│       ├── xai/
│       ├── config.py
│       ├── data_cli.py
│       └── main.py
├── frontend/
│   ├── ai_insights_view.py
│   ├── api_client.py
│   ├── app.py
│   ├── branding.py
│   ├── dashboard_view.py
│   ├── decision_workspace_view.py
│   ├── forecast_view.py
│   ├── governance_view.py
│   ├── i18n.py
│   ├── market_intelligence_view.py
│   ├── optimization_view.py
│   ├── reports_view.py
│   └── risk_view.py
├── benchmark_results/
│   ├── agents/
│   ├── final_test/
│   ├── forecast/
│   ├── rag/
│   ├── risk/
│   ├── rl/
│   └── xai/
├── data/
├── docs/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_forecasting.ipynb
│   ├── 03_risk_and_portfolio.ipynb
│   ├── 04_explainable_ai.ipynb
│   ├── 05_llm_rag_agents.ipynb
│   └── 06_reinforcement_learning.ipynb
├── tests/
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Wygenerowane datasety i artefakty benchmarkowe są traktowane jako reprodukowalne wyniki analityczne.

Artefakty finalnego testu pełnią dodatkowo funkcję metodologicznego zapisu końcowej ewaluacji i nie powinny być ponownie generowane na podstawie zużytego już zbioru testowego.

## Lokalna konfiguracja

Projekt wymaga Python 3.12.

Utwórz i aktywuj środowisko wirtualne:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Zainstaluj zależności runtime:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Zainstaluj zależności developerskie:

```bash
python3 -m pip install -r requirements-dev.txt
```

Sprawdź środowisko:

```bash
python3 -m pip check
```

Niektóre workflow RAG i agentowe wymagają również lokalnej instalacji Ollama oraz skonfigurowanego lokalnego modelu językowego.

## Ponowne zbudowanie datasetów rynkowych

Workflow danych rynkowych można uruchomić przez data CLI.

Data końcowa jest wyłączna i powinna być podana jawnie w celu zachowania reprodukowalności.

Przykład:

```bash
python3 -m backend.src.data_cli \
  --start-date 2010-01-01 \
  --end-date 2026-08-01 \
  --minimum-observations 1000
```

Wygenerowane dane raw, processed i model datasets są lokalnymi artefaktami analitycznymi.

Ponowne zbudowanie datasetów po terminalnej finalnej ewaluacji tworzy nowy stan metodologiczny i nie powinno być używane do twierdzenia, że wynik jest bezpośrednią kontynuacją zapisanej finalnej ewaluacji.

## Uruchamianie workflow prognozowania

Funkcjonalność benchmarków prognozowania jest dostępna przez pakiet backend forecast.

Przykład:

```bash
python3 -m backend.src.forecast.cli
```

Artefakty benchmarków mogą zawierać prognozy na poziomie obserwacji, zagregowane metryki oraz wyniki ewaluacji w formatach maszynowych.

Standardowe workflow prognozowania wykorzystują dane walidacyjne do celów rozwojowych.

## Uruchamianie benchmarku RAG

Przykład:

```bash
python3 -m backend.src.rag.cli \
  --corpus data/rag/evaluation/reference_corpus.json \
  --golden-dataset data/rag/evaluation/golden_dataset_expanded.json \
  --output benchmark_results/rag
```

Uruchomienie RAG może wymagać lokalnie dostępnych modeli embeddingowych, rerankingowych oraz Ollama, zależnie od wybranej konfiguracji.

## Uruchamianie aplikacji webowej bez Dockera

Uruchom backend FastAPI z katalogu głównego projektu:

```bash
python3 -m uvicorn \
  backend.src.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

W drugim terminalu aktywuj to samo środowisko wirtualne i uruchom Streamlit:

```bash
cd /path/to/project-root
source .venv/bin/activate

PYTHONPATH="$PWD" python3 -m streamlit run frontend/app.py
```

Domyślnie Streamlit jest dostępny lokalnie na porcie `8501`.

## Uruchamianie testów frontendu

Uruchom zestaw testów frontendu:

```bash
python3 -m pytest \
  tests/frontend \
  -q
```

## Kontrola jakości projektu

Uruchom Ruff:

```bash
python3 -m ruff check .
```

Sprawdź formatowanie:

```bash
python3 -m ruff format --check .
```

## Testy regresyjne

Projekt zawiera zarówno standardowe testy analityczne, jak i testy Reinforcement Learning.

Ze względu na możliwość problemów z jednoczesną inicjalizacją natywnych komponentów PyTorch i FAISS w jednym procesie Python w niektórych środowiskach, testy regresyjne są wykonywane w dwóch kontrolowanych grupach.

Uruchom zestaw non-RL:

```bash
python3 -m pytest tests -q \
  --ignore=tests/rl \
  --ignore=tests/integration/test_rl_allocation_workflow_integration.py \
  --ignore=tests/integration/test_portfolio_allocation_env_integration.py
```

Następnie osobno uruchom testy RL:

```bash
python3 -m pytest \
  tests/rl \
  tests/integration/test_rl_allocation_workflow_integration.py \
  tests/integration/test_portfolio_allocation_env_integration.py \
  -q
```

Aktualny stan testów regresyjnych przy zamykaniu V2:

```text
Non-RL: 793 passed
RL:      59 passed
Łącznie: 852 passed
```

Zestaw non-RL zgłaszał również cztery nieblokujące warningi deprecation pochodzące z zewnętrznych zależności.

Podział testów jest zabezpieczeniem środowiska wykonawczego i nie zmienia logicznego zakresu testowanego systemu.

## Zasady metodologiczne

Projekt wykorzystuje następujący protokół ewaluacji:

```text
Dane treningowe
    ↓
uczenie modeli / konstrukcja parametrów

Dane walidacyjne
    ↓
rozwój, porównanie i analiza interfejsu

Dane testowe
    ↓
jedna końcowa ewaluacja po zamrożeniu decyzji projektowych
```

Protokół ten był przestrzegany podczas rozwoju projektu.

Końcowy test split pozostawał zamknięty do momentu zamrożenia wszystkich istotnych decyzji dotyczących prognozowania, portfela i Reinforcement Learning.

Następnie został otwarty dokładnie jeden raz przez kontrolowany workflow finalnej ewaluacji.

Trwały marker użycia zapisuje fakt zużycia test splitu:

```text
benchmark_results/final_test/final_test_usage.json
```

Zbiór testowy nie może być traktowany jako niezależnie wielokrotnie wykorzystywany dataset ewaluacyjny.

Pozostałe zasady metodologiczne obejmują:

- chronologiczny zamiast losowego podział danych finansowych;
- jawne daty targetu;
- zapobieganie target leakage pomiędzy granicami datasetów;
- transparentne porównanie z naiwnym baseline;
- raportowanie negatywnych lub słabych wyników modeli;
- rozdzielenie jakości predykcyjnej od interpretowalności;
- ocenę zachowania portfela poza okresem wykorzystanym do konstrukcji;
- grounding odpowiedzi generatywnych w możliwych do odnalezienia źródłach;
- jawne przedstawienie ograniczeń analitycznych;
- reprodukowalne artefakty benchmarkowe;
- rozdzielenie obliczeń backendu od prezentacji frontendu;
- jawne oddzielenie selekcji modeli od finalnej ewaluacji;
- zachowanie artefaktów governance związanych z końcowym testem.

## Interpretacja wyników

System jest aplikacją wspomagania decyzji, a nie automatycznym doradcą inwestycyjnym.

Dokładność prognoz może być niska nawet wtedy, gdy wyjaśnienia modelu są technicznie poprawne.

Historyczna optymalizacja portfela nie gwarantuje przyszłej skuteczności.

Miary ryzyka zależą od historycznych obserwacji i przyjętych założeń.

Wyniki RAG zależą od jakości i zakresu dostępnej bazy wiedzy.

Wyniki agentów zależą zarówno od ustrukturyzowanych dowodów analitycznych, jak i zachowania lokalnego modelu językowego.

Wyniki Reinforcement Learning mogą silnie zależeć od procesu treningowego, konstrukcji funkcji nagrody, reżimu rynkowego i konfiguracji środowiska.

Dobry wynik w jednym końcowym okresie testowym nie oznacza uniwersalnej przewagi w przyszłości.

Wszystkie wyniki powinny być interpretowane wraz z ich provenance, użytym splitem ewaluacyjnym oraz jawnymi ograniczeniami.

## Aktualne ograniczenia

System posiada kilka celowych ograniczeń.

Problem predykcyjny jest trudny, a modele prognostyczne zapewniają jedynie ograniczoną przewagę predykcyjną, mimo że Random Forest osiągnął najlepszy wynik spośród ocenianych modeli w finalnym teście.

Korpus ewaluacyjny RAG jest nadal niewielki i pozostaje przede wszystkim zbiorem rozwojowym.

Końcowy dataset testowy rynku został już wykorzystany w jednorazowej terminalnej ewaluacji i nie może być ponownie używany do rozwoju ani strojenia modeli.

Optymalizacja portfela opiera się na historycznych relacjach stóp zwrotu i kowariancji, które mogą zmieniać się w czasie.

Wynik PPO został uzyskany w jednym końcowym okresie testowym i nie powinien być interpretowany jako dowód uniwersalnej przewagi nad metodami klasycznymi.

Aplikacja nie modeluje obecnie wszystkich indywidualnych ograniczeń inwestora, podatków ani pełnego rzeczywistego modelu kosztów transakcyjnych.

Lokalne uruchamianie LLM wprowadza opóźnienia i zależy od dostępnego lokalnie modelu.

Aplikacja jest przeznaczona do celów analitycznych, edukacyjnych i badawczych i nie stanowi porady inwestycyjnej.

## Status ukończenia projektu

Główne etapy inżynieryjne i analityczne zostały ukończone:

```text
1. Baseline analityczny V1 i ewaluacja terminalna              COMPLETE
2. Forecasting V2 i ekonomiczne bramki promocji                COMPLETE
3. Deterministyczny Decision Engine i workspace HITL           COMPLETE
4. Grounded AI, XAI, RAG i audytowalne raportowanie            COMPLETE
5. Kontekst rynkowy, lokalizacja i dostępność                   COMPLETE
6. Kontenery, CI, obserwowalność i odporność                    COMPLETE
7. Dokumentacja projektu                                       COMPLETE
8. Weryfikacja finalnej paczki i test uruchomienia Docker       COMPLETE
```

Końcowy test split został wykorzystany w jednorazowej terminalnej ewaluacji i nie jest już dostępny do niezależnego rozwoju modeli.

Każda przyszła zmiana metodologiczna wymaga nowego protokołu ewaluacji oraz rzeczywiście niewidzianych wcześniej danych.

## Kontekst akademicki i inżynieryjny

Projekt demonstruje zarówno metodologię Data Science, jak i praktyki AI/ML Engineering.

Główne wkłady inżynieryjne i metodologiczne obejmują:

- reprodukowalne workflow danych;
- typowane usługi analityczne;
- jawne kontrakty API;
- automatyczne testy jednostkowe i integracyjne;
- temporal evaluation świadomą ryzyka leakage;
- porównanie modeli z baseline;
- explainability modeli;
- ilościową analizę ryzyka;
- optymalizację portfela;
- ewaluację retrieval;
- generowanie odpowiedzi opartych na evidence;
- kontrolowaną orkiestrację agentów;
- eksperymentalne Reinforcement Learning;
- interaktywne wizualizacje analityczne;
- reprodukowalne artefakty benchmarkowe;
- fingerprinting zamrożonych artefaktów;
- kontrolowaną jednorazową końcową ewaluację;
- jawny governance test splitu;
- wyraźne rozdzielenie warstw aplikacji;
- jawne przedstawienie ograniczeń i zasad governance.

Celem projektu nie jest maksymalizacja złożoności architektury, lecz zbudowanie spójnego, testowalnego i interpretowalnego systemu wspomagania decyzji finansowych.
