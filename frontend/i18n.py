"""Small explicit Polish/English translation catalog for the Streamlit UI."""

from typing import Final, Literal

LanguageCode = Literal["pl", "en"]

DEFAULT_LANGUAGE: Final[LanguageCode] = "pl"
SUPPORTED_LANGUAGES: Final[tuple[LanguageCode, ...]] = ("pl", "en")
LANGUAGE_NAMES: Final[dict[LanguageCode, str]] = {
    "pl": "Polski",
    "en": "English",
}

_TRANSLATIONS: Final[dict[LanguageCode, dict[str, str]]] = {
    "pl": {
        "preferences.heading": "Ustawienia prezentacji",
        "preferences.language": "Język / Language",
        "preferences.timezone": "Strefa czasowa",
        "accessibility.chart_description": "Opis wykresu: {description}",
        "accessibility.optimization.weights": (
            "Wykres słupkowy porównuje udział każdego instrumentu w wybranym portfelu."
        ),
        "accessibility.optimization.return_contribution": (
            "Wykres słupkowy pokazuje wkład każdego instrumentu w roczny zwrot portfela."
        ),
        "accessibility.optimization.volatility_contribution": (
            "Wykres słupkowy pokazuje wkład każdego instrumentu w roczną zmienność portfela."
        ),
        "accessibility.optimization.cumulative": (
            "Wykres liniowy przedstawia skumulowany wynik portfela w kolejnych datach walidacji."
        ),
        "accessibility.governance.global": (
            "Wykres słupkowy porządkuje cechy według średniej bezwzględnej wartości SHAP."
        ),
        "accessibility.governance.pdp": (
            "Wykres liniowy przedstawia średnią prognozę modelu dla kolejnych wartości wybranej cechy."
        ),
        "accessibility.governance.ale": (
            "Wykres liniowy przedstawia skumulowany lokalny efekt wybranej cechy."
        ),
        "accessibility.governance.local": (
            "Wykres słupkowy pokazuje dodatni lub ujemny wkład SHAP cech dla jednej obserwacji."
        ),
        "navigation.heading": "Moduły",
        "navigation.caption": "Wybierz obszar analizy",
        "navigation.about": "Opis aplikacji",
        "navigation.dashboard": "Dashboard",
        "navigation.forecasting": "Prognozowanie",
        "navigation.risk": "Ryzyko i scenariusze",
        "navigation.optimization": "Optymalizacja",
        "navigation.decision": "Przestrzeń decyzyjna",
        "navigation.market": "Informacje rynkowe",
        "navigation.ai": "Wnioski AI",
        "navigation.governance": "Wyjaśnialność i governance",
        "navigation.reports": "Raporty",
        "about.title": "Opis aplikacji",
        "about.caption": "Krótki przewodnik po systemie wspomagania decyzji finansowych.",
        "about.disclaimer": (
            "Aplikacja ma charakter analityczny i edukacyjny. Nie stanowi "
            "indywidualnej rekomendacji inwestycyjnej ani gwarancji przyszłych wyników."
        ),
        "about.purpose.title": "Do czego służy?",
        "about.purpose.body": (
            "System łączy prognozowanie, analizę ryzyka, porównywanie portfeli, "
            "wyjaśnialność modeli oraz lokalne mechanizmy AI. Pomaga uporządkować "
            "informacje i porównać dostępne warianty przy jawnych założeniach."
        ),
        "about.data.title": "Na jakich danych się opiera?",
        "about.data.body": (
            "Analizy wykorzystują historyczne dzienne dane rynkowe instrumentów "
            "AGG, BTC-USD, EPOL, GLD, QQQ, SPY, VGK i VT pobrane z Yahoo Finance. "
            "Bieżący widok informacji rynkowych pokazuje ostatnie dostępne ceny, "
            "a nie notowania czasu rzeczywistego."
        ),
        "about.results.title": "Co otrzymuje użytkownik?",
        "about.results.body": (
            "Użytkownik może przeglądać prognozy i ich błędy, miary ryzyka, wagi "
            "oraz wyniki portfeli, ranking wariantów, wyjaśnienia modelu Random "
            "Forest, informacje rynkowe, odpowiedzi RAG i gotowe raporty."
        ),
        "about.modules.title": "Przewodnik po modułach",
        "about.modules.caption": "Każda zakładka odpowiada innemu etapowi analizy.",
        "about.module.dashboard.title": "Dashboard",
        "about.module.dashboard.body": "Syntetyczny przegląd najważniejszych wyników i statusów systemu.",
        "about.module.forecasting.title": "Prognozowanie",
        "about.module.forecasting.body": "Porównanie prognoz modeli naiwnych, ARIMA i Random Forest na danych walidacyjnych.",
        "about.module.risk.title": "Ryzyko i scenariusze",
        "about.module.risk.body": "Miary ryzyka oraz wyniki zdefiniowanych scenariuszy portfelowych.",
        "about.module.optimization.title": "Optymalizacja",
        "about.module.optimization.body": "Wagi, efektywność, ryzyko i przebieg wartości porównywanych portfeli.",
        "about.module.decision.title": "Przestrzeń decyzyjna",
        "about.module.decision.body": "Deterministyczny ranking wariantów według profilu, priorytetów i ograniczeń użytkownika.",
        "about.module.market.title": "Informacje rynkowe",
        "about.module.market.body": "Ostatnie dostępne ceny, zmiany dzienne i jawna informacja o świeżości danych.",
        "about.module.ai.title": "Wnioski AI",
        "about.module.ai.body": "Wyszukiwanie RAG i agenci porządkujący informacje z zapisanych artefaktów analitycznych.",
        "about.module.governance.title": "Wyjaśnialność i governance",
        "about.module.governance.body": "Wyjaśnienia SHAP, PDP i ALE oraz informacje o pochodzeniu i ograniczeniach wyników.",
        "about.module.reports.title": "Raporty",
        "about.module.reports.body": "Generowanie podsumowań wyników wraz z zakresem, źródłami i ograniczeniami analizy.",
        "about.start.title": "Jak rozpocząć?",
        "about.start.body": (
            "Rozpocznij od **Dashboardu**, aby zobaczyć przekrojowe podsumowanie. "
            "Następnie wybierz moduł odpowiadający interesującej Cię analizie. "
            "Do świadomego porównania wariantów przejdź do **Przestrzeni decyzyjnej**."
        ),
        "system.heading": "Status systemu",
        "system.api_available": "API dostępne",
        "system.api_unavailable": "API niedostępne",
        "system.backend_status": "Stan backendu: {status}",
        "system.api_version": "Wersja API: {version}",
        "system.backend": "Backend: {url}",
        "system.http_boundary": (
            "Warstwa prezentacji komunikuje się z backendem wyłącznie przez HTTP/JSON."
        ),
        "page.unknown": "Nieznany moduł aplikacji.",
        "market.title": "Informacje rynkowe",
        "market.caption": (
            "Ostatnie dostępne dzienne ceny z Yahoo Finance. Moduł nie "
            "prezentuje notowań czasu rzeczywistego."
        ),
        "market.refresh": "Odśwież dane z Yahoo Finance",
        "market.refresh_help": (
            "Pomija świeży cache i podejmuje nową próbę pobrania danych."
        ),
        "market.load_error": "Nie udało się pobrać informacji rynkowych.",
        "market.refresh_done": "Wykonano ręczną próbę odświeżenia danych.",
        "market.status_error": (
            "Brak dostępnych cen. Sprawdź połączenie z siecią i spróbuj ponownie."
        ),
        "market.status_partial": (
            "Dane częściowe: nieaktualne {stale}, niedostępne {unavailable}."
        ),
        "market.status_success": (
            "Wszystkie ceny są dostępne i mieszczą się w limicie cache."
        ),
        "market.metric_available": "Dostępne instrumenty",
        "market.metric_stale": "Nieaktualne instrumenty",
        "market.metric_cache": "Limit świeżego cache",
        "market.empty": "API nie zwróciło żadnych instrumentów.",
        "market.time_note": (
            "Timestamp jest przechowywany w UTC i prezentowany w wybranej "
            "strefie czasowej. Strefa rynku jest pokazana oddzielnie."
        ),
        "market.disclaimer": (
            "Dane mają charakter informacyjny, nie są notowaniami czasu "
            "rzeczywistego ani rekomendacją inwestycyjną."
        ),
        "market.freshness_help": "Jak interpretować świeżość danych?",
        "market.freshness_markdown": (
            "- **Świeże**: pobrane w granicach skonfigurowanego czasu cache.\n"
            "- **Nieaktualne**: starsza kopia użyta po błędzie sieci.\n"
            "- **Niedostępne**: brak odpowiedzi Yahoo Finance i brak cache.\n"
            "- **Cache offline**: ciągłość działania bez ukrywania wieku danych."
        ),
        "market.fresh": "Świeże",
        "market.stale": "Nieaktualne",
        "market.unavailable": "Niedostępne",
        "market.live": "Yahoo Finance",
        "market.cache": "Pamięć podręczna",
        "market.offline_cache": "Cache offline",
        "market.no_data": "Brak danych",
        "market.missing": "Brak",
        "market.column.ticker": "Ticker",
        "market.column.price": "Cena",
        "market.column.currency": "Waluta",
        "market.column.change": "Zmiana dzienna",
        "market.column.change_percent": "Zmiana dzienna %",
        "market.column.market_date": "Data rynku",
        "market.column.freshness": "Świeżość",
        "market.column.age": "Wiek danych",
        "market.column.delivery": "Dostarczenie",
        "market.column.presented_time": "Czas prezentacji",
        "market.column.market_timezone": "Strefa rynku",
        "decision.profile.conservative": "Konserwatywny",
        "decision.profile.balanced": "Zrównoważony",
        "decision.profile.growth": "Wzrostowy",
        "decision.scenario.none": "Brak rekomendacji",
        "decision.scenario.defensive": "Defensywny",
        "decision.scenario.equal_weight": "Równe wagi",
        "decision.scenario.growth": "Wzrostowy",
        "decision.scenario.maximum_sharpe": "Maksymalny Sharpe",
        "decision.scenario.minimum_volatility": "Minimalna zmienność",
        "decision.component.return": "Zwrot",
        "decision.component.volatility": "Zmienność",
        "decision.component.drawdown": "Obsunięcie",
        "decision.component.diversification": "Dywersyfikacja",
        "decision.yes": "Tak",
        "decision.no": "Nie",
        "decision.title": "Przestrzeń decyzyjna",
        "decision.caption": (
            "Świadome porównywanie wariantów alokacji przy jawnych priorytetach "
            "i ograniczeniach."
        ),
        "decision.local_info": (
            "Obliczenia są lokalne i deterministyczne. LLM nie wybiera wariantu, "
            "nie zmienia wag i nie generuje kosztów API."
        ),
        "decision.profile.active": "Założenia aktywnego profilu",
        "decision.profile.constraints": "Ograniczenia profilu",
        "decision.profile.selector": "Profil decyzyjny",
        "decision.liquidity.outside": (
            "Kapitał pozostający poza analizowanym portfelem"
        ),
        "decision.liquidity.help": (
            "Ta wartość reprezentuje płynną rezerwę utrzymywaną poza "
            "porównywanymi wariantami portfela."
        ),
        "decision.calculate": "Przelicz ranking",
        "decision.custom.title": "Własne założenia — analiza what-if",
        "decision.custom.caption": (
            "Punktem odniesienia pozostaje wybrany profil bazowy. Zmiany wpływają "
            "wyłącznie na ranking istniejących wariantów; nie zmieniają ich wag."
        ),
        "decision.custom.priorities": "Priorytety — suma musi wynosić 100%",
        "decision.custom.priority_sum": "Suma priorytetów: {value}%",
        "decision.custom.constraints": "Twarde ograniczenia",
        "decision.custom.max_volatility": "Maksymalna zmienność roczna (%)",
        "decision.custom.max_drawdown": "Maksymalne obsunięcie (%)",
        "decision.custom.max_concentration": "Maksymalna koncentracja (%)",
        "decision.custom.min_assets": "Minimalna liczba efektywnych aktywów",
        "decision.custom.min_diversification": ("Minimalny wskaźnik dywersyfikacji"),
        "decision.custom.min_cash": "Minimalna rezerwa płynnościowa (%)",
        "decision.custom.liquidity_need": "Deklarowana potrzeba płynności",
        "decision.custom.horizon": "Horyzont inwestycyjny (lata)",
        "decision.custom.submit": "Oceń własne założenia",
        "decision.custom.sum_error": (
            "Nie można przeliczyć rankingu: suma czterech priorytetów musi "
            "wynosić dokładnie 100%."
        ),
        "decision.liquidity.low": "Niska",
        "decision.liquidity.medium": "Średnia",
        "decision.liquidity.high": "Wysoka",
        "decision.comparison.title": "Wpływ zmiany założeń",
        "decision.comparison.before": "Rekomendacja przed zmianą",
        "decision.comparison.after": "Rekomendacja po zmianie",
        "decision.comparison.caption": (
            "Porównanie opisuje obserwowane skutki przeliczenia. Nie jest miarą "
            "pewności ani gwarancją wyniku inwestycyjnego."
        ),
        "decision.metric.recommended": "Rekomendowany wariant",
        "decision.metric.admissible": "Dopuszczalne",
        "decision.metric.rejected": "Odrzucone",
        "decision.metric.source": "Źródło oceny",
        "decision.no_admissible": (
            "Żaden wariant nie spełnia wszystkich ograniczeń. Zmień założenia "
            "świadomie zamiast wymuszać rekomendację."
        ),
        "decision.success": (
            "Rekomendacja wynika z jawnego rankingu metryk walidacyjnych: {name}."
        ),
        "decision.ranking": "Ranking i odrzucenia",
        "decision.components": "Składowe rekomendacji",
        "decision.governance": "Provenance i zasady governance",
        "decision.governance.deterministic": "Ranking deterministyczny:",
        "decision.governance.llm": "LLM użyty do rankingu:",
        "decision.governance.weights": "Wagi portfeli zmodyfikowane:",
        "decision.narrative.title": "Opcjonalne wyjaśnienie lokalnego AI",
        "decision.narrative.caption": (
            "Agent tylko objaśnia ponownie obliczony wynik Decision Engine. Nie "
            "wybiera wariantu i nie zmienia wag. Ollama jest opcjonalna; bez niej "
            "backend zwraca deterministyczne wyjaśnienie bez kosztów API."
        ),
        "decision.narrative.question": "Co chcesz wyjaśnić?",
        "decision.narrative.default_question": (
            "Wyjaśnij rekomendację, odrzucone warianty i najważniejsze ograniczenia."
        ),
        "decision.narrative.submit": "Wyjaśnij wynik",
        "decision.narrative.error": "Nie udało się przygotować wyjaśnienia wyniku.",
        "decision.narrative.result": "Wyjaśnienie AI z kontrolą faktów",
        "decision.narrative.findings": "Najważniejsze ustalenia",
        "decision.narrative.risks": "Ryzyka",
        "decision.narrative.steps": "Dalsze kroki",
        "decision.narrative.limitations": "Ograniczenia",
        "decision.narrative.sources": "Źródła narracji",
        "decision.report.title": "Audytowalny raport decyzji",
        "decision.report.caption": (
            "Backend ponownie oblicza ranking i tworzy równoważne raporty JSON, "
            "Markdown i PDF. Każdy plik jest sprawdzany skrótem SHA-256."
        ),
        "decision.report.scope": "Zakres raportu",
        "decision.report.timezone": "Strefa czasowa raportu",
        "decision.report.submit": "Generuj raport",
        "decision.report.error": "Nie udało się przygotować bezpiecznego raportu.",
        "decision.report.integrity_error": (
            "Zapisany raport nie przeszedł kontroli integralności."
        ),
        "decision.report.ready": "Raport został wygenerowany i zweryfikowany.",
        "decision.report.download_json": "Pobierz JSON",
        "decision.report.download_markdown": "Pobierz Markdown",
        "decision.report.download_pdf": "Pobierz PDF",
        "decision.report.download_manifest": "Pobierz manifest",
        "decision.error.profiles": (
            "Nie udało się pobrać profili decyzyjnych z backendu."
        ),
        "decision.error.no_profiles": "API nie zwróciło dostępnych profili decyzyjnych.",
        "decision.error.ranking": "Nie udało się przeliczyć rankingu decyzyjnego.",
        "decision.error.no_result": "Brak wyniku Decision Engine.",
        "decision.error.custom": "Nie udało się ocenić własnych założeń.",
        "decision.custom.result": "Wynik dla własnych założeń",
        "decision.column.rank": "Pozycja",
        "decision.column.scenario": "Scenariusz",
        "decision.column.admissible": "Dopuszczalny",
        "decision.column.score": "Wynik",
        "decision.column.violations": "Naruszenia",
        "decision.column.criterion": "Kryterium",
        "decision.column.normalized": "Wartość znormalizowana",
        "decision.column.contribution": "Wkład do wyniku",
        "decision.column.before_admissible": "Dopuszczalny — przed",
        "decision.column.before_rank": "Pozycja — przed",
        "decision.column.before_score": "Wynik — przed",
        "decision.column.after_admissible": "Dopuszczalny — po",
        "decision.column.after_rank": "Pozycja — po",
        "decision.column.after_score": "Wynik — po",
        "decision.change.unchanged": (
            "Rekomendowany wariant pozostał bez zmian: {name}."
        ),
        "decision.change.changed": "Rekomendacja zmieniła się: {before} → {after}.",
        "decision.change.rejected": (
            "Wariant „{name}” został odrzucony po zmianie ograniczeń."
        ),
        "decision.change.admitted": (
            "Wariant „{name}” stał się dopuszczalny po zmianie ograniczeń."
        ),
        "decision.change.warning": (
            "Rekomendacja zmieniła się po zmianie założeń: {before} → {after}."
        ),
        "decision.report.metadata": (
            "Źródło: {source}; test użyty: {test}; czas UTC: {time}"
        ),
        "decision.violation.maximum_annualized_volatility": (
            "przekroczona maksymalna zmienność"
        ),
        "decision.violation.maximum_drawdown": "przekroczone maksymalne obsunięcie",
        "decision.violation.maximum_concentration_index": "przekroczona koncentracja",
        "decision.violation.minimum_effective_assets": (
            "zbyt mała liczba efektywnych aktywów"
        ),
        "decision.violation.minimum_diversification_ratio": (
            "zbyt niski poziom dywersyfikacji"
        ),
        "decision.violation.minimum_cash_reserve": (
            "niewystarczająca rezerwa płynnościowa"
        ),
        "decision.constraint.maximum_annualized_volatility": (
            "Maksymalna zmienność roczna"
        ),
        "decision.constraint.maximum_drawdown": "Maksymalne obsunięcie",
        "decision.constraint.maximum_concentration_index": "Maksymalna koncentracja",
        "decision.constraint.minimum_effective_assets": (
            "Minimalna liczba efektywnych aktywów"
        ),
        "decision.constraint.minimum_diversification_ratio": (
            "Minimalny wskaźnik dywersyfikacji"
        ),
        "decision.constraint.minimum_cash_reserve": ("Minimalna rezerwa płynnościowa"),
        "dashboard.title": "Dashboard",
        "dashboard.caption": (
            "Syntetyczny przegląd najważniejszych wyników systemu wspomagania "
            "decyzji finansowych."
        ),
        "dashboard.executive.heading": "Podsumowanie decyzji",
        "dashboard.executive.caption": (
            "Deterministyczny wynik dla zrównoważonego profilu odniesienia i "
            "10% płynności pozostającej poza portfelem."
        ),
        "dashboard.executive.empty": (
            "Decision Engine nie zwrócił dopuszczalnego scenariusza. Przejdź do "
            "Przestrzeni decyzyjnej, aby sprawdzić ograniczenia."
        ),
        "dashboard.executive.recommendation": (
            "Scenariusz {scenario} zajmuje pierwsze miejsce dla zrównoważonych założeń."
        ),
        "dashboard.executive.scenario": "Rekomendowany scenariusz",
        "dashboard.executive.return": "Zwrot roczny — walidacja",
        "dashboard.executive.volatility": "Zmienność — walidacja",
        "dashboard.executive.sharpe": "Sharpe — walidacja",
        "dashboard.executive.drawdown": "Maksymalne obsunięcie",
        "dashboard.executive.forecast": "Trafność kierunku RF",
        "dashboard.executive.validation_notice": (
            "Wyniki historyczne ze zbioru walidacyjnego. Nie są prognozą przyszłych "
            "stóp zwrotu ani indywidualną rekomendacją inwestycyjną."
        ),
        "dashboard.actions.heading": "Przejdź do szczegółów",
        "dashboard.actions.decision": "Dostosuj założenia",
        "dashboard.actions.forecast": "Sprawdź prognozy",
        "dashboard.actions.market": "Zobacz rynek",
        "dashboard.actions.reports": "Otwórz raporty",
        "dashboard.technical.heading": "Status techniczny i zakres",
        "dashboard.kpi.heading": "Kluczowe wskaźniki",
        "dashboard.kpi.direction": "RF trafność kierunku",
        "dashboard.kpi.max_sharpe": "Sharpe – max",
        "dashboard.kpi.lowest_volatility": "Najniższa zmienność",
        "dashboard.kpi.validation_observations": "Obserwacje walidacyjne",
        "dashboard.portfolio.heading": "Snapshot portfela",
        "dashboard.portfolio.empty": "Brak danych portfelowych do prezentacji.",
        "dashboard.portfolio.caption": (
            "Porównanie wykorzystuje wyniki walidacyjne. Strategie optymalizowane "
            "zostały skonstruowane na zbiorze treningowym."
        ),
        "dashboard.column.strategy": "Strategia",
        "dashboard.column.return": "Zwrot roczny",
        "dashboard.column.volatility": "Zmienność",
        "dashboard.column.sharpe": "Sharpe",
        "dashboard.column.drawdown": "Max drawdown",
        "dashboard.column.effective_assets": "Efektywna liczba aktywów",
        "dashboard.capabilities.heading": "Dostępność komponentów",
        "dashboard.column.component": "Komponent",
        "dashboard.column.status": "Status",
        "dashboard.capability.rag_corpus": "Korpus RAG",
        "dashboard.capability.rag_benchmark": "Benchmark RAG",
        "dashboard.capability.forecast_agent": "Agent Forecast",
        "dashboard.capability.risk_agent": "Agent Risk",
        "dashboard.capability.allocation_agent": "Agent Allocation",
        "dashboard.capability.agent_runtime": "Runtime agentów",
        "dashboard.status.available": "dostępny",
        "dashboard.status.ready": "gotowy",
        "dashboard.status.missing": "brak",
        "dashboard.status.no_data": "brak danych",
        "dashboard.alerts.heading": "Uwagi analityczne",
        "dashboard.alerts.direction": (
            "Trafność kierunku modelu Random Forest znajduje się blisko poziomu "
            "50%, dlatego wyniki prognostyczne należy interpretować ostrożnie."
        ),
        "dashboard.alerts.r2": (
            "R² modelu Random Forest jest bliskie zeru, co wskazuje na ograniczoną "
            "zdolność wyjaśniania zmienności przyszłych zwrotów."
        ),
        "dashboard.alerts.drawdown": (
            "Nawet strategia o minimalnej zmienności wykazuje istotny historyczny "
            "drawdown. Niska zmienność nie oznacza braku ryzyka strat."
        ),
        "dashboard.alerts.test_closed": (
            "Końcowy zbiór testowy pozostaje zamknięty. Dashboard prezentuje "
            "wyniki walidacyjne."
        ),
        "dashboard.error": "Nie udało się pobrać danych wymaganych przez Dashboard.",
        "dashboard.scope.heading": "Zakres i interpretacja Dashboardu",
        "dashboard.scope.body": (
            "- Dashboard agreguje dane już udostępnione przez istniejące endpointy API.\n"
            "- Frontend nie wykonuje ponownie trenowania modeli ani obliczeń finansowych.\n"
            "- Prezentowane KPI dotyczą zbioru walidacyjnego.\n"
            "- Wyniki prognostyczne, ryzyko i wyniki portfela należy interpretować łącznie.\n"
            "- Dostępność komponentu oznacza dostępność wymaganych artefaktów lub danych runtime, a nie gwarancję poprawności przyszłych odpowiedzi modelu.\n"
            "- Końcowy zbiór testowy pozostaje zamknięty do etapu finalnej ewaluacji.\n"
            "- Dashboard ma charakter analityczny i nie stanowi indywidualnej rekomendacji inwestycyjnej."
        ),
        "common.split.training": "treningowy",
        "common.split.validation": "walidacyjny",
        "forecast.title": "Prognozowanie",
        "forecast.caption": (
            "Ocena jakości modeli prognostycznych na historycznym zbiorze walidacyjnym."
        ),
        "forecast.info": (
            "Prezentowane wyniki pochodzą z zapisanych artefaktów benchmarkowych. "
            "Końcowy zbiór testowy nie jest wykorzystywany na tym etapie."
        ),
        "forecast.error.load": "Nie udało się pobrać danych Forecast z backendu.",
        "forecast.models.empty": ("API nie zwróciło informacji o dostępnych modelach."),
        "forecast.models.heading": "Dostępność modeli",
        "forecast.models.available": "dostępny",
        "forecast.models.missing": "brak",
        "forecast.rf.empty": ("Brak wyników Random Forest dla wybranego instrumentu."),
        "forecast.rf.metrics_empty": "Brak metryk Random Forest.",
        "forecast.column.model": "Model",
        "forecast.column.observations": "Obserwacje",
        "forecast.column.direction": "Trafność kierunku",
        "forecast.ticker.metrics_empty": (
            "Brak metryk modeli dla wybranego instrumentu."
        ),
        "forecast.ticker.comparison_note": (
            "Trafność kierunku modelu Naive zero-return nie jest bezpośrednio "
            "porównywalna z modelami generującymi prognozę znaku zwrotu, "
            "ponieważ model naiwny prognozuje zero."
        ),
        "forecast.global.heading": "Porównanie globalne",
        "forecast.global.empty": "Brak zbiorczych wyników walidacyjnych.",
        "forecast.global.caption": (
            "Metryki globalne agregują obserwacje dla wszystkich instrumentów "
            "uwzględnionych w benchmarku."
        ),
        "forecast.metric.split": "Zbiór oceny",
        "forecast.metric.assets": "Instrumenty",
        "forecast.metric.target": "Zmienna celu",
        "forecast.ticker.heading": "Analiza instrumentu",
        "forecast.ticker.empty": ("API nie zwróciło listy obsługiwanych instrumentów."),
        "forecast.ticker.select": "Instrument finansowy",
        "forecast.ticker.error": (
            "Nie udało się pobrać wyników dla wybranego instrumentu."
        ),
        "forecast.ticker.context": "Instrument: {ticker} · zbiór: {split} · cel: {target}",
        "forecast.ticker.models_heading": "Porównanie modeli",
        "forecast.interpretation.heading": "Interpretacja wyników",
        "forecast.interpretation.body": (
            "- MAE i RMSE opisują wielkość błędu prognozy zwrotu.\n"
            "- Niższe MAE i RMSE oznaczają mniejszy błąd predykcyjny.\n"
            "- R² bliskie zeru oznacza bardzo ograniczoną zdolność modelu do wyjaśniania zmienności przyszłych zwrotów.\n"
            "- Trafność kierunku bliska 50% nie wskazuje na silną przewagę w przewidywaniu znaku przyszłego zwrotu.\n"
            "- Wyniki walidacyjne służą do porównania modeli i nie są gwarancją przyszłej skuteczności.\n"
            "- Interpretacja cech modelu została wydzielona do modułu Explainability and Governance."
        ),
        "forecast.warning": (
            "Moduł prezentuje ocenę modeli, a nie rekomendację inwestycyjną. "
            "Wyniki prognostyczne powinny być interpretowane razem z analizą "
            "ryzyka i ograniczeniami modelu."
        ),
        "risk.title": "Ryzyko i scenariusze",
        "risk.caption": (
            "Analiza ryzyka portfelowego i porównanie scenariuszy na historycznym "
            "zbiorze walidacyjnym."
        ),
        "risk.info": (
            "Prezentowane wyniki pochodzą z zapisanych artefaktów benchmarkowych. "
            "Końcowy zbiór testowy pozostaje zamknięty."
        ),
        "risk.error.load": "Nie udało się pobrać danych Risk z backendu.",
        "risk.scenarios.empty": "API nie zwróciło scenariuszy ryzyka.",
        "risk.comparison.heading": "Porównanie scenariuszy",
        "risk.comparison.caption": (
            "Scenariusze reprezentują wcześniej zdefiniowane warianty konstrukcji "
            "portfela. Porównanie dotyczy wyników walidacyjnych."
        ),
        "risk.scenario.select": "Scenariusz",
        "risk.scenario.construction": "Konstrukcja scenariusza: {construction}",
        "risk.scenario.error": "Nie udało się pobrać szczegółów scenariusza.",
        "risk.profile.heading": "Profil ryzyka",
        "risk.split.heading": "Trening a walidacja",
        "risk.split.caption": (
            "Wyniki treningowe pokazano wyłącznie w celu oceny stabilności "
            "scenariusza. Główna interpretacja dotyczy zbioru walidacyjnego."
        ),
        "risk.market.heading": "Kontekst rynkowy",
        "risk.market.observations": "Obserwacje",
        "risk.market.assets": "Aktywa",
        "risk.market.periods": "Okresy / rok",
        "risk.market.range": "Zakres danych",
        "risk.market.instruments": "Instrumenty: {tickers}",
        "risk.market.return_definition": "Definicja zwrotu: {definition}",
        "risk.market.correlation": "Macierz korelacji",
        "risk.metric.return": "Zwrot roczny",
        "risk.metric.volatility": "Zmienność roczna",
        "risk.metric.sharpe": "Sharpe",
        "risk.metric.drawdown": "Maksymalne obsunięcie",
        "risk.metric.effective_assets": "Efektywna liczba aktywów",
        "risk.metric.diversification": "Wskaźnik dywersyfikacji",
        "risk.column.scenario": "Scenariusz",
        "risk.column.return": "Zwrot roczny",
        "risk.column.volatility": "Zmienność roczna",
        "risk.column.sharpe": "Sharpe",
        "risk.column.sortino": "Sortino",
        "risk.column.var": "VaR",
        "risk.column.cvar": "CVaR",
        "risk.column.drawdown": "Maksymalne obsunięcie",
        "risk.column.concentration": "Koncentracja",
        "risk.column.effective_assets": "Efektywna liczba aktywów",
        "risk.column.diversification": "Wskaźnik dywersyfikacji",
        "risk.column.split": "Zbiór",
        "risk.column.observations": "Obserwacje",
        "risk.interpretation.heading": "Interpretacja miar ryzyka",
        "risk.interpretation.body": (
            "- **Zmienność** opisuje skalę historycznych wahań zwrotów.\n"
            "- **VaR** wskazuje próg straty przy zadanym poziomie ufności.\n"
            "- **CVaR** opisuje średnią stratę po przekroczeniu progu VaR.\n"
            "- **Maksymalne obsunięcie** mierzy największy historyczny spadek od lokalnego maksimum do minimum.\n"
            "- **Sharpe** zestawia nadwyżkowy zwrot z całkowitą zmiennością.\n"
            "- **Sortino** koncentruje się na niekorzystnej zmienności.\n"
            "- **Wskaźnik koncentracji** opisuje koncentrację wag portfela.\n"
            "- **Efektywna liczba aktywów** przybliża rzeczywisty poziom dywersyfikacji.\n"
            "- **Wskaźnik dywersyfikacji** porównuje ryzyko składników z ryzykiem całego portfela."
        ),
        "risk.limitations.heading": "Ograniczenia analizy",
        "risk.limitations.body": (
            "- Miary ryzyka są estymowane na podstawie danych historycznych.\n"
            "- Wyniki mogą nie odzwierciedlać przyszłych reżimów rynkowych.\n"
            "- VaR i CVaR zależą od poziomu ufności i obserwowanego rozkładu zwrotów.\n"
            "- Korelacje historyczne mogą zmieniać się w okresach stresu rynkowego.\n"
            "- Scenariusze są wcześniej zdefiniowanymi wariantami, a nie interaktywną symulacją przyszłości.\n"
            "- Wagi strategii optymalizowanych wyznaczono na zbiorze treningowym.\n"
            "- Końcowy zbiór testowy nie jest wykorzystywany w tym widoku.\n"
            "- Moduł nie stanowi indywidualnej rekomendacji inwestycyjnej."
        ),
        "optimization.title": "Optymalizacja",
        "optimization.caption": (
            "Porównanie strategii alokacji i wyników konstrukcji portfela."
        ),
        "optimization.info": (
            "Strategie optymalizowane wyznaczono na zbiorze treningowym, a "
            "prezentowane wyniki główne dotyczą zbioru walidacyjnego. Końcowy "
            "zbiór testowy pozostaje zamknięty."
        ),
        "optimization.error.load": (
            "Nie udało się pobrać danych Portfolio z backendu."
        ),
        "optimization.empty": "API nie zwróciło strategii portfelowych.",
        "optimization.metric.split": "Zbiór oceny",
        "optimization.metric.strategies": "Strategie",
        "optimization.metric.assets": "Instrumenty",
        "optimization.return_definition": "Definicja zwrotu: {definition}",
        "optimization.comparison.heading": "Porównanie strategii",
        "optimization.comparison.caption": (
            "Porównanie obejmuje strategie regułowe oraz portfele wyznaczone "
            "na podstawie danych treningowych."
        ),
        "optimization.scenario.select": "Strategia",
        "optimization.scenario.construction": "Metoda konstrukcji: {construction}",
        "optimization.scenario.error": (
            "Nie udało się pobrać szczegółów wybranej strategii."
        ),
        "optimization.validation.heading": "Wyniki walidacyjne",
        "optimization.weights.heading": "Struktura portfela",
        "optimization.weights.empty": "Brak danych o wagach portfela.",
        "optimization.contributions.heading": "Wkład aktywów",
        "optimization.contributions.empty": (
            "Brak danych o wkładzie poszczególnych aktywów."
        ),
        "optimization.contributions.return": "Wkład w zwrot",
        "optimization.contributions.volatility": "Wkład w zmienność",
        "optimization.contributions.data": "Dane o wkładzie aktywów",
        "optimization.cumulative.heading": "Wynik skumulowany",
        "optimization.cumulative.empty": ("Brak szeregu czasowego wyników portfela."),
        "optimization.cumulative.final": "Końcowy zwrot skumulowany",
        "optimization.cumulative.caption": (
            "Wykres przedstawia wynik na zbiorze walidacyjnym, a nie na końcowym "
            "zbiorze testowym."
        ),
        "optimization.split.heading": "Trening a walidacja",
        "optimization.split.caption": (
            "Porównanie zbiorów pomaga ocenić, czy parametry strategii zachowują "
            "stabilność poza okresem użytym do ich konstrukcji."
        ),
        "optimization.interpretation.heading": "Interpretacja optymalizacji",
        "optimization.interpretation.body": (
            "- **Maximum Sharpe** wybiera wagi maksymalizujące historyczną relację nadwyżkowego zwrotu do ryzyka na zbiorze treningowym.\n"
            "- **Minimum Volatility** minimalizuje historyczną zmienność portfela na zbiorze treningowym.\n"
            "- **Equal Weight** stanowi prosty punkt odniesienia bez optymalizacji wag.\n"
            "- Pozostałe scenariusze reprezentują zdefiniowane wcześniej profile konstrukcji portfela.\n"
            "- Wysoka wartość Sharpe w treningu nie gwarantuje utrzymania wyniku poza próbą.\n"
            "- Porównanie trening–walidacja pomaga ocenić ryzyko przeuczenia strategii."
        ),
        "optimization.limitations.heading": "Ograniczenia",
        "optimization.limitations.body": (
            "- Optymalizacja wykorzystuje historyczne zależności pomiędzy stopami zwrotu aktywów.\n"
            "- Oczekiwane zwroty, zmienności i korelacje mogą zmieniać się w przyszłości.\n"
            "- Wyniki są wrażliwe na okres danych wykorzystany do konstrukcji strategii.\n"
            "- Matematycznie optymalny portfel nie musi być optymalny dla konkretnego inwestora.\n"
            "- Widok nie uwzględnia indywidualnych ograniczeń inwestora, podatków ani pełnego modelu kosztów transakcyjnych.\n"
            "- Końcowy zbiór testowy nie jest wykorzystywany podczas bieżącego projektowania aplikacji.\n"
            "- Wyniki nie stanowią indywidualnej rekomendacji inwestycyjnej."
        ),
        "optimization.column.strategy": "Strategia",
        "optimization.column.asset_count": "Liczba aktywów",
        "optimization.column.instrument": "Instrument",
        "optimization.column.weight": "Waga",
        "optimization.column.return_contribution": "Wkład w zwrot roczny",
        "optimization.column.volatility_contribution": "Wkład w zmienność roczną",
        "optimization.column.date": "Data",
        "optimization.column.portfolio_return": "Zwrot portfela",
        "optimization.column.cumulative_return": "Zwrot skumulowany",
        "optimization.chart.weight": "Waga portfela",
        "optimization.chart.contribution": "Wkład",
        "governance.title": "Wyjaśnialność i nadzór nad modelem",
        "governance.caption": (
            "Interpretacja działania modeli oraz kontrola pochodzenia danych, "
            "ograniczeń i procesu ewaluacji."
        ),
        "governance.error.load": "Nie udało się pobrać danych XAI z backendu.",
        "governance.artifact.missing": "Artefakt XAI jest niedostępny.",
        "governance.metric.model": "Model",
        "governance.metric.split": "Zbiór wyjaśnień",
        "governance.metric.observations": "Obserwacje SHAP",
        "governance.metric.artifact": "Artefakt XAI",
        "governance.status.available": "dostępny",
        "governance.status.missing": "brak",
        "governance.target": "Definicja zmiennej celu: {target}",
        "governance.methods": "Metody interpretacji: {methods}",
        "governance.global.heading": "Globalna wyjaśnialność",
        "governance.global.empty": "Brak globalnych wyników XAI.",
        "governance.global.warning": (
            "Wartości SHAP opisują sposób wykorzystania cech przez model. Nie "
            "stanowią dowodu zależności przyczynowej pomiędzy cechą a przyszłym zwrotem."
        ),
        "governance.effect.heading": "Efekt wybranej cechy",
        "governance.effect.empty": "Brak cech z dostępnymi analizami PDP i ALE.",
        "governance.effect.select": "Cecha",
        "governance.effect.error": "Nie udało się pobrać analizy efektu cechy.",
        "governance.effect.pdp_empty": "Brak danych PDP.",
        "governance.effect.ale_empty": "Brak danych ALE.",
        "governance.effect.source": "Dane źródłowe PDP i ALE",
        "governance.local.heading": "Lokalne wyjaśnienie",
        "governance.local.caption": (
            "Reprezentatywna obserwacja lokalna zapisana w artefakcie XAI."
        ),
        "governance.local.context": (
            "**Instrument:** {ticker}  \n**Data cech:** {feature_date}  "
            "\n**Data celu:** {target_date}"
        ),
        "governance.local.actual": "Rzeczywisty zwrot",
        "governance.local.predicted": "Prognozowany zwrot",
        "governance.local.base": "Wartość bazowa",
        "governance.local.reconstructed": "Rekonstrukcja SHAP",
        "governance.local.empty": "Brak lokalnych wartości SHAP.",
        "governance.local.error": ("Nie udało się pobrać lokalnego wyjaśnienia XAI."),
        "governance.section.heading": "Nadzór nad modelem",
        "governance.section.body": (
            "**Zasady obowiązujące w obecnej fazie projektu:**\n\n"
            "- wyjaśnienia dotyczą modelu Random Forest ocenianego na zbiorze walidacyjnym,\n"
            "- interpretacja modelu jest oddzielona od oceny końcowej,\n"
            "- końcowy zbiór testowy pozostaje zamknięty,\n"
            "- artefakty benchmarkowe są źródłem danych prezentowanych przez API,\n"
            "- frontend nie wykonuje ponownie obliczeń SHAP, PDP ani ALE,\n"
            "- wyniki interpretacji modelu nie są traktowane jako dowód zależności przyczynowej."
        ),
        "governance.section.valid": (
            "Wyjaśnienia wykorzystują zbiór walidacyjny. Końcowy zbiór testowy "
            "nie został otwarty."
        ),
        "governance.section.invalid": (
            "Zbiór użyty do wyjaśnień nie jest oznaczony jako validation. "
            "Wymagana jest weryfikacja pochodzenia danych."
        ),
        "governance.limitations.heading": "Ograniczenia interpretacji",
        "governance.limitations.body": (
            "- SHAP opisuje zachowanie modelu, a nie mechanizm przyczynowy rynku.\n"
            "- PDP może być zniekształcony w obecności silnie skorelowanych cech.\n"
            "- ALE ogranicza część problemów korelacji, lecz nadal jest metodą interpretacji modelu.\n"
            "- Lokalna interpretacja dotyczy pojedynczej reprezentatywnej obserwacji.\n"
            "- Wyniki modelu mogą zmieniać się wraz ze zmianą reżimu rynkowego.\n"
            "- Wyjaśnialność nie kompensuje słabej jakości prognostycznej modelu.\n"
            "- Wyniki walidacyjne nie są gwarancją przyszłej skuteczności.\n"
            "- Moduł służy kontroli i interpretacji systemu, a nie rekomendacji inwestycyjnej."
        ),
        "governance.column.feature": "Cecha",
        "governance.column.identifier": "Identyfikator",
        "governance.column.mean_shap": "Średnia |SHAP|",
        "governance.column.effect": "PDP/ALE",
        "governance.column.feature_value": "Wartość cechy",
        "governance.yes": "tak",
        "governance.no": "nie",
        "governance.feature.ticker": "Identyfikator {ticker}",
        "governance.feature.adjusted_close_to_ma_20": "Cena względem MA(20)",
        "governance.feature.momentum_20": "Momentum 20",
        "governance.feature.adjusted_close_to_ma_5": "Cena względem MA(5)",
        "governance.feature.open_to_close_return": "Zwrot open-to-close",
        "governance.feature.daily_return": "Dzienny zwrot",
        "governance.feature.volatility_5": "Zmienność 5",
        "governance.feature.volatility_20": "Zmienność 20",
        "governance.feature.intraday_range": "Dzienny zakres ceny",
        "governance.feature.momentum_5": "Momentum 5",
        "governance.feature.volume_change": "Zmiana wolumenu",
        "governance.chart.mean_shap": "Średnia bezwzględna wartość SHAP",
        "governance.chart.feature_value": "Wartość cechy",
        "governance.chart.average_prediction": "Średnia prognoza",
        "governance.chart.bin_center": "Środek przedziału",
        "governance.chart.ale_effect": "Efekt ALE",
        "governance.chart.shap_value": "Wartość SHAP",
        "ai.title": "Wnioski AI",
        "ai.caption": (
            "Synteza informacji źródłowych oraz ustrukturyzowane wsparcie analizy decyzyjnej."
        ),
        "ai.info": (
            "Moduł łączy odpowiedzi oparte na kontrolowanym korpusie z agentami "
            "interpretującymi wyniki pozostałych komponentów systemu."
        ),
        "ai.tab.rag": "Zapytanie do źródeł",
        "ai.tab.agents": "Analiza agentowa",
        "ai.available": "dostępny",
        "ai.missing": "brak",
        "ai.no_data": "brak danych",
        "ai.ready": "gotowy",
        "ai.used": "użyty",
        "ai.closed": "zamknięty",
        "ai.strategy.vector": "Wyszukiwanie wektorowe",
        "ai.strategy.hybrid": "Wyszukiwanie hybrydowe",
        "ai.strategy.reranked": "Wyszukiwanie z rerankingiem",
        "ai.agent_type.forecast": "Analiza prognoz",
        "ai.agent_type.risk": "Analiza ryzyka",
        "ai.agent_type.allocation": "Analiza alokacji",
        "ai.agent_name.forecast_analysis_agent": "Agent analizy prognoz",
        "ai.agent_name.risk_analysis_agent": "Agent analizy ryzyka",
        "ai.agent_name.allocation_recommendation_agent": "Agent analizy alokacji",
        "ai.risk.defensive": "Defensywna",
        "ai.risk.balanced": "Zrównoważona",
        "ai.risk.growth": "Wzrostowa",
        "ai.timeout": "Przekroczono czas oczekiwania na {operation}.",
        "ai.timeout.help": (
            "Operacje wykorzystujące lokalny model językowy mają limit {seconds} s. "
            "Jeżeli model jest właśnie inicjalizowany, odczekaj chwilę i spróbuj ponownie."
        ),
        "ai.rag.heading": "Zapytanie do źródeł",
        "ai.rag.caption": (
            "Generowanie odpowiedzi na podstawie kontrolowanego korpusu z "
            "wykorzystaniem Retrieval-Augmented Generation."
        ),
        "ai.rag.status_error": "Nie udało się pobrać statusu modułu RAG.",
        "ai.rag.metric.corpus": "Korpus",
        "ai.rag.metric.evaluation": "Zbiór ewaluacyjny",
        "ai.rag.metric.benchmark": "Benchmark",
        "ai.rag.metric.model": "Model LLM",
        "ai.rag.documents": "Dokumenty korpusu: {count}",
        "ai.rag.queries": "Zapytania ewaluacyjne: {count}",
        "ai.rag.question": "Zapytanie",
        "ai.rag.placeholder": (
            "Wprowadź zapytanie dotyczące informacji dostępnych w bazie wiedzy..."
        ),
        "ai.rag.strategy": "Strategia wyszukiwania",
        "ai.rag.submit": "Generuj odpowiedź",
        "ai.rag.question_empty": "Wprowadź zapytanie przed uruchomieniem wyszukiwania.",
        "ai.rag.spinner": "Wyszukiwanie źródeł i generowanie odpowiedzi...",
        "ai.rag.operation": "wygenerowanie odpowiedzi",
        "ai.rag.error": "Nie udało się uzyskać odpowiedzi z modułu RAG.",
        "ai.rag.answer": "Odpowiedź",
        "ai.rag.answer_empty": "Brak odpowiedzi.",
        "ai.rag.sources": "Źródła",
        "ai.rag.response_strategy": "Strategia: {strategy}",
        "ai.rag.chunk_count": "Liczba pobranych fragmentów: {count}",
        "ai.rag.warning": (
            "Odpowiedź modelu językowego należy interpretować razem ze źródłami. "
            "Nie stanowi indywidualnej porady finansowej."
        ),
        "ai.rag.citations_empty": "Odpowiedź nie zawiera jawnych odwołań do źródeł.",
        "ai.rag.column.document": "Dokument",
        "ai.rag.column.title": "Tytuł",
        "ai.rag.column.source": "Źródło",
        "ai.rag.column.page": "Strona",
        "ai.rag.benchmark": "Benchmark wyszukiwania",
        "ai.rag.benchmark_current": "Konfiguracja wskazana w benchmarku: {strategy}",
        "ai.rag.benchmark_empty": "Brak szczegółowej tabeli benchmarku wyszukiwania.",
        "ai.rag.benchmark_unavailable": "Benchmark jest obecnie niedostępny.",
        "ai.rag.benchmark_note": (
            "Konfiguracja domyślna nie oznacza przewagi jakościowej, jeżeli "
            "strategie osiągają ten sam wynik."
        ),
        "ai.rag.column.strategy": "Strategia",
        "ai.rag.column.duration": "Czas [s]",
        "ai.rag.limitations": "Ograniczenia RAG",
        "ai.rag.limitations_body": (
            "- Odpowiedź zależy od informacji dostępnych w kontrolowanym korpusie.\n"
            "- Retrieval może zwrócić fragmenty jedynie częściowo związane z zapytaniem.\n"
            "- Model językowy może błędnie zinterpretować poprawnie odnalezione źródło.\n"
            "- Źródła powinny służyć do weryfikacji wygenerowanej odpowiedzi.\n"
            "- Materiał źródłowy może zostać zsyntetyzowany w języku użytkownika.\n"
            "- Wyniki benchmarku nie gwarantują jakości każdego przyszłego zapytania."
        ),
        "ai.agents.heading": "Analiza agentowa",
        "ai.agents.caption": (
            "Ustrukturyzowana analiza wykorzystująca artefakty prognoz, ryzyka, "
            "portfela, uczenia ze wzmocnieniem i kontekst RAG."
        ),
        "ai.agents.status_error": "Nie udało się pobrać statusu modułu agentowego.",
        "ai.agents.runtime_missing": (
            "Nie wszystkie dane wymagane przez runtime agentów są dostępne."
        ),
        "ai.agents.metric.runtime": "Runtime",
        "ai.agents.metric.model": "Model LLM",
        "ai.agents.metric.retrieval": "Retrieval",
        "ai.agents.metric.test": "Final test",
        "ai.agents.evaluation_basis": "Podstawa ewaluacji: {basis}",
        "ai.agents.validation_ok": (
            "Analiza agentowa korzysta z artefaktów walidacyjnych. Końcowy zbiór "
            "testowy pozostaje zamknięty."
        ),
        "ai.agents.test_warning": (
            "Metadane wskazują wykorzystanie końcowego zbioru testowego."
        ),
        "ai.agents.type": "Typ analizy",
        "ai.agents.question": "Zapytanie analityczne",
        "ai.agents.placeholder.forecast": (
            "Wprowadź zapytanie dotyczące wyników prognoz lub jakości modeli..."
        ),
        "ai.agents.placeholder.risk": (
            "Wprowadź zapytanie dotyczące ryzyka lub porównania scenariuszy..."
        ),
        "ai.agents.placeholder.allocation": (
            "Wprowadź zapytanie dotyczące alokacji lub porównania strategii..."
        ),
        "ai.agents.ticker_placeholder": "Opcjonalnie, np. SPY",
        "ai.agents.ticker_help": (
            "Pozostaw puste, aby analiza nie była ograniczona do jednego instrumentu."
        ),
        "ai.agents.risk_preference": "Preferencja ryzyka",
        "ai.agents.submit": "Uruchom analizę",
        "ai.agents.question_empty": "Wprowadź zapytanie przed uruchomieniem analizy.",
        "ai.agents.spinner": "Agent analizuje dostępne artefakty...",
        "ai.agents.operation": "wykonanie analizy agentowej",
        "ai.agents.error": "Nie udało się wykonać analizy agentowej.",
        "ai.agents.workflow": "Zapisany workflow walidacyjny",
        "ai.agents.workflow_empty": "Brak zapisanych wyników workflow agentowego.",
        "ai.agents.workflow_unavailable": (
            "Zapisane wyniki agentów są obecnie niedostępne."
        ),
        "ai.agents.workflow_metadata": (
            "Model: {model} · retrieval: {retrieval} · podstawa: {basis}"
        ),
        "ai.agents.question_label": "Zapytanie: {question}",
        "ai.agents.summary": "Podsumowanie",
        "ai.agents.summary_empty": "Brak podsumowania.",
        "ai.agents.findings": "Najważniejsze ustalenia",
        "ai.agents.risks": "Ryzyka",
        "ai.agents.actions": "Rekomendowane działania",
        "ai.agents.limitations_title": "Ograniczenia",
        "ai.agents.evidence": "Materiał dowodowy",
        "ai.agents.no_items": "Brak danych.",
        "ai.agents.evidence_empty": "Brak jawnych referencji do materiału dowodowego.",
        "ai.agents.column.source_type": "Typ źródła",
        "ai.agents.column.identifier": "Identyfikator",
        "ai.agents.column.description": "Opis",
        "ai.agents.column.role": "Rola",
        "ai.agents.column.agent": "Agent",
        "ai.agents.column.summary": "Podsumowanie",
        "ai.agents.warning": (
            "Wynik agenta stanowi ustrukturyzowane wsparcie analityczne. Nie jest "
            "indywidualną rekomendacją inwestycyjną."
        ),
        "ai.agents.limitations": "Ograniczenia analizy agentowej",
        "ai.agents.limitations_body": (
            "- Agenci interpretują istniejące artefakty zamiast ponownie trenować modele.\n"
            "- Odpowiedzi zależą od jakości prognoz, ryzyka, optymalizacji i kontekstu RAG.\n"
            "- Agent może błędnie zsyntetyzować poprawne dane źródłowe.\n"
            "- Wyniki interaktywne mogą różnić się między uruchomieniami modelu.\n"
            "- Preferencja ryzyka nie jest pełnym profilem inwestora.\n"
            "- Końcowy zbiór testowy pozostaje zamknięty podczas projektowania interfejsu.\n"
            "- Wyniki nie stanowią indywidualnej rekomendacji inwestycyjnej."
        ),
        "reports.title": "Raporty",
        "reports.caption": "Generowanie i eksport reprodukowalnego raportu z wyników systemu.",
        "reports.info": (
            "Raport jest składany wyłącznie z danych zwracanych przez API. "
            "Generowanie nie uruchamia ponownego trenowania modeli."
        ),
        "reports.generate": "Generuj raport",
        "reports.spinner": "Pobieranie danych i generowanie raportu...",
        "reports.error": "Nie udało się wygenerować raportu.",
        "reports.empty": "Wybierz „Generuj raport”, aby przygotować aktualny raport walidacyjny.",
        "reports.summary": "Podsumowanie raportu",
        "reports.export": "Eksport",
        "reports.preview": "Podgląd",
        "reports.preview_markdown": "Pokaż raport Markdown",
        "reports.scope": "Zakres",
        "reports.final_test": "Final test",
        "reports.ai_model": "Model AI",
        "reports.download_markdown": "Pobierz raport Markdown",
        "reports.download_json": "Pobierz dane raportu JSON",
        "reports.warning": (
            "Raport przedstawia wyniki analityczne i walidacyjne. Nie stanowi "
            "indywidualnej rekomendacji inwestycyjnej."
        ),
    },
    "en": {
        "preferences.heading": "Presentation settings",
        "preferences.language": "Language / Język",
        "preferences.timezone": "Time zone",
        "accessibility.chart_description": "Chart description: {description}",
        "accessibility.optimization.weights": (
            "The bar chart compares each instrument's share of the selected portfolio."
        ),
        "accessibility.optimization.return_contribution": (
            "The bar chart shows each instrument's contribution to annualized portfolio return."
        ),
        "accessibility.optimization.volatility_contribution": (
            "The bar chart shows each instrument's contribution to annualized portfolio volatility."
        ),
        "accessibility.optimization.cumulative": (
            "The line chart shows cumulative portfolio performance across validation dates."
        ),
        "accessibility.governance.global": (
            "The bar chart ranks features by their mean absolute SHAP value."
        ),
        "accessibility.governance.pdp": (
            "The line chart shows the model's average prediction across values of the selected feature."
        ),
        "accessibility.governance.ale": (
            "The line chart shows the accumulated local effect of the selected feature."
        ),
        "accessibility.governance.local": (
            "The bar chart shows positive or negative SHAP contributions for one observation."
        ),
        "navigation.heading": "Modules",
        "navigation.caption": "Select an analysis area",
        "navigation.about": "Application overview",
        "navigation.dashboard": "Dashboard",
        "navigation.forecasting": "Forecasting",
        "navigation.risk": "Risk and scenarios",
        "navigation.optimization": "Optimization",
        "navigation.decision": "Decision Workspace",
        "navigation.market": "Market Intelligence",
        "navigation.ai": "AI Insights",
        "navigation.governance": "Explainability and governance",
        "navigation.reports": "Reports",
        "about.title": "Application overview",
        "about.caption": "A short guide to the Financial Decision Support System.",
        "about.disclaimer": (
            "The application is intended for analytical and educational use. It does "
            "not provide individualized investment advice or guarantee future results."
        ),
        "about.purpose.title": "What is it for?",
        "about.purpose.body": (
            "The system combines forecasting, risk analysis, portfolio comparison, "
            "model explainability and local AI mechanisms. It helps organize evidence "
            "and compare available alternatives under explicit assumptions."
        ),
        "about.data.title": "What data does it use?",
        "about.data.body": (
            "The analyses use historical daily market data for AGG, BTC-USD, EPOL, "
            "GLD, QQQ, SPY, VGK and VT obtained from Yahoo Finance. The market "
            "intelligence view shows the latest available prices, not real-time quotes."
        ),
        "about.results.title": "What can the user obtain?",
        "about.results.body": (
            "Users can inspect forecasts and errors, risk measures, portfolio weights "
            "and performance, ranked alternatives, Random Forest explanations, market "
            "information, RAG answers and downloadable reports."
        ),
        "about.modules.title": "Module guide",
        "about.modules.caption": "Each page supports a different stage of the analysis.",
        "about.module.dashboard.title": "Dashboard",
        "about.module.dashboard.body": "A concise overview of the system's main results and statuses.",
        "about.module.forecasting.title": "Forecasting",
        "about.module.forecasting.body": "Comparison of naive, ARIMA and Random Forest forecasts on validation data.",
        "about.module.risk.title": "Risk and scenarios",
        "about.module.risk.body": "Risk measures and results for predefined portfolio scenarios.",
        "about.module.optimization.title": "Optimization",
        "about.module.optimization.body": "Weights, performance, risk and value paths of the compared portfolios.",
        "about.module.decision.title": "Decision Workspace",
        "about.module.decision.body": "A deterministic ranking based on the user's profile, priorities and constraints.",
        "about.module.market.title": "Market Intelligence",
        "about.module.market.body": "Latest available prices, daily changes and explicit data-freshness information.",
        "about.module.ai.title": "AI Insights",
        "about.module.ai.body": "RAG retrieval and agents that organize information from stored analytical artifacts.",
        "about.module.governance.title": "Explainability and governance",
        "about.module.governance.body": "SHAP, PDP and ALE explanations plus result provenance and limitations.",
        "about.module.reports.title": "Reports",
        "about.module.reports.body": "Result summaries with the scope, sources and limitations of the analysis.",
        "about.start.title": "How to begin",
        "about.start.body": (
            "Start with the **Dashboard** for a cross-system summary. Then choose the "
            "module that matches your analytical question. Use the **Decision "
            "Workspace** to compare alternatives under explicit assumptions."
        ),
        "system.heading": "System status",
        "system.api_available": "API available",
        "system.api_unavailable": "API unavailable",
        "system.backend_status": "Backend status: {status}",
        "system.api_version": "API version: {version}",
        "system.backend": "Backend: {url}",
        "system.http_boundary": (
            "The presentation layer communicates with the backend only via HTTP/JSON."
        ),
        "page.unknown": "Unknown application module.",
        "market.title": "Market Intelligence",
        "market.caption": (
            "Latest available daily prices from Yahoo Finance. This module "
            "does not present real-time quotes."
        ),
        "market.refresh": "Refresh data from Yahoo Finance",
        "market.refresh_help": (
            "Bypass a fresh cache and make a new data retrieval attempt."
        ),
        "market.load_error": "Market information could not be loaded.",
        "market.refresh_done": "A manual data refresh was attempted.",
        "market.status_error": (
            "No prices are available. Check the network connection and try again."
        ),
        "market.status_partial": (
            "Partial data: stale {stale}, unavailable {unavailable}."
        ),
        "market.status_success": (
            "All prices are available and within the configured cache limit."
        ),
        "market.metric_available": "Available instruments",
        "market.metric_stale": "Stale instruments",
        "market.metric_cache": "Fresh cache limit",
        "market.empty": "The API returned no instruments.",
        "market.time_note": (
            "The timestamp is stored in UTC and displayed in the selected time "
            "zone. The market time zone is shown separately."
        ),
        "market.disclaimer": (
            "The data is informational, is not a real-time quote and does not "
            "constitute investment advice."
        ),
        "market.freshness_help": "How should data freshness be interpreted?",
        "market.freshness_markdown": (
            "- **Fresh**: retrieved within the configured cache lifetime.\n"
            "- **Stale**: an older copy used after a network failure.\n"
            "- **Unavailable**: no Yahoo Finance response and no cache.\n"
            "- **Offline cache**: continuity without hiding the age of the data."
        ),
        "market.fresh": "Fresh",
        "market.stale": "Stale",
        "market.unavailable": "Unavailable",
        "market.live": "Yahoo Finance",
        "market.cache": "Cache",
        "market.offline_cache": "Offline cache",
        "market.no_data": "No data",
        "market.missing": "Missing",
        "market.column.ticker": "Ticker",
        "market.column.price": "Price",
        "market.column.currency": "Currency",
        "market.column.change": "Daily change",
        "market.column.change_percent": "Daily change %",
        "market.column.market_date": "Market date",
        "market.column.freshness": "Freshness",
        "market.column.age": "Data age",
        "market.column.delivery": "Delivery",
        "market.column.presented_time": "Presented time",
        "market.column.market_timezone": "Market time zone",
        "decision.profile.conservative": "Conservative",
        "decision.profile.balanced": "Balanced",
        "decision.profile.growth": "Growth",
        "decision.scenario.none": "No recommendation",
        "decision.scenario.defensive": "Defensive",
        "decision.scenario.equal_weight": "Equal weight",
        "decision.scenario.growth": "Growth",
        "decision.scenario.maximum_sharpe": "Maximum Sharpe",
        "decision.scenario.minimum_volatility": "Minimum volatility",
        "decision.component.return": "Return",
        "decision.component.volatility": "Volatility",
        "decision.component.drawdown": "Drawdown",
        "decision.component.diversification": "Diversification",
        "decision.yes": "Yes",
        "decision.no": "No",
        "decision.title": "Decision Workspace",
        "decision.caption": (
            "Compare allocation variants consciously with explicit priorities "
            "and constraints."
        ),
        "decision.local_info": (
            "Calculations are local and deterministic. The LLM does not select "
            "a variant, change weights or generate API costs."
        ),
        "decision.profile.active": "Active profile assumptions",
        "decision.profile.constraints": "Profile constraints",
        "decision.profile.selector": "Decision profile",
        "decision.liquidity.outside": "Capital held outside the analyzed portfolio",
        "decision.liquidity.help": (
            "This value represents a liquid reserve held outside the compared "
            "portfolio variants."
        ),
        "decision.calculate": "Recalculate ranking",
        "decision.custom.title": "Custom assumptions — what-if analysis",
        "decision.custom.caption": (
            "The selected preset remains the baseline. Changes affect only the "
            "ranking of existing variants; they do not change their weights."
        ),
        "decision.custom.priorities": "Priorities — the sum must equal 100%",
        "decision.custom.priority_sum": "Priority total: {value}%",
        "decision.custom.constraints": "Hard constraints",
        "decision.custom.max_volatility": "Maximum annualized volatility (%)",
        "decision.custom.max_drawdown": "Maximum drawdown (%)",
        "decision.custom.max_concentration": "Maximum concentration (%)",
        "decision.custom.min_assets": "Minimum effective assets",
        "decision.custom.min_diversification": "Minimum diversification ratio",
        "decision.custom.min_cash": "Minimum liquidity reserve (%)",
        "decision.custom.liquidity_need": "Declared liquidity need",
        "decision.custom.horizon": "Investment horizon (years)",
        "decision.custom.submit": "Evaluate custom assumptions",
        "decision.custom.sum_error": (
            "The ranking cannot be recalculated: the four priorities must sum "
            "to exactly 100%."
        ),
        "decision.liquidity.low": "Low",
        "decision.liquidity.medium": "Medium",
        "decision.liquidity.high": "High",
        "decision.comparison.title": "Impact of changed assumptions",
        "decision.comparison.before": "Recommendation before",
        "decision.comparison.after": "Recommendation after",
        "decision.comparison.caption": (
            "The comparison describes observed recalculation effects. It is not "
            "a confidence measure or a guarantee of investment performance."
        ),
        "decision.metric.recommended": "Recommended variant",
        "decision.metric.admissible": "Admissible",
        "decision.metric.rejected": "Rejected",
        "decision.metric.source": "Evaluation source",
        "decision.no_admissible": (
            "No variant satisfies every constraint. Change assumptions explicitly "
            "instead of forcing a recommendation."
        ),
        "decision.success": (
            "The recommendation follows from an explicit ranking of validation "
            "metrics: {name}."
        ),
        "decision.ranking": "Ranking and rejections",
        "decision.components": "Recommendation components",
        "decision.governance": "Provenance and governance rules",
        "decision.governance.deterministic": "Deterministic ranking:",
        "decision.governance.llm": "LLM used for ranking:",
        "decision.governance.weights": "Portfolio weights modified:",
        "decision.narrative.title": "Optional local AI explanation",
        "decision.narrative.caption": (
            "The agent only explains a server-recomputed Decision Engine result. "
            "It does not select a variant or change weights. Ollama is optional; "
            "without it, the backend returns a deterministic no-cost explanation."
        ),
        "decision.narrative.question": "What would you like to explain?",
        "decision.narrative.default_question": (
            "Explain the recommendation, rejected variants and key constraints."
        ),
        "decision.narrative.submit": "Explain result",
        "decision.narrative.error": "The result explanation could not be prepared.",
        "decision.narrative.result": "Fact-checked AI explanation",
        "decision.narrative.findings": "Key findings",
        "decision.narrative.risks": "Risks",
        "decision.narrative.steps": "Next steps",
        "decision.narrative.limitations": "Limitations",
        "decision.narrative.sources": "Narrative sources",
        "decision.report.title": "Auditable decision report",
        "decision.report.caption": (
            "The backend recalculates the ranking and creates equivalent JSON, "
            "Markdown and PDF reports. Every file is verified with SHA-256."
        ),
        "decision.report.scope": "Report scope",
        "decision.report.timezone": "Report time zone",
        "decision.report.submit": "Generate report",
        "decision.report.error": "A secure report could not be prepared.",
        "decision.report.integrity_error": (
            "The stored report failed its integrity check."
        ),
        "decision.report.ready": "The report was generated and verified.",
        "decision.report.download_json": "Download JSON",
        "decision.report.download_markdown": "Download Markdown",
        "decision.report.download_pdf": "Download PDF",
        "decision.report.download_manifest": "Download manifest",
        "decision.error.profiles": (
            "Decision profiles could not be loaded from the backend."
        ),
        "decision.error.no_profiles": "The API returned no decision profiles.",
        "decision.error.ranking": "The decision ranking could not be recalculated.",
        "decision.error.no_result": "No Decision Engine result is available.",
        "decision.error.custom": "Custom assumptions could not be evaluated.",
        "decision.custom.result": "Result for custom assumptions",
        "decision.column.rank": "Rank",
        "decision.column.scenario": "Scenario",
        "decision.column.admissible": "Admissible",
        "decision.column.score": "Score",
        "decision.column.violations": "Violations",
        "decision.column.criterion": "Criterion",
        "decision.column.normalized": "Normalized value",
        "decision.column.contribution": "Score contribution",
        "decision.column.before_admissible": "Admissible — before",
        "decision.column.before_rank": "Rank — before",
        "decision.column.before_score": "Score — before",
        "decision.column.after_admissible": "Admissible — after",
        "decision.column.after_rank": "Rank — after",
        "decision.column.after_score": "Score — after",
        "decision.change.unchanged": (
            "The recommended variant remained unchanged: {name}."
        ),
        "decision.change.changed": ("The recommendation changed: {before} → {after}."),
        "decision.change.rejected": (
            "Variant “{name}” was rejected after the constraints changed."
        ),
        "decision.change.admitted": (
            "Variant “{name}” became admissible after the constraints changed."
        ),
        "decision.change.warning": (
            "The recommendation changed after assumptions changed: {before} → {after}."
        ),
        "decision.report.metadata": (
            "Source: {source}; test used: {test}; UTC time: {time}"
        ),
        "decision.violation.maximum_annualized_volatility": (
            "maximum volatility exceeded"
        ),
        "decision.violation.maximum_drawdown": "maximum drawdown exceeded",
        "decision.violation.maximum_concentration_index": (
            "maximum concentration exceeded"
        ),
        "decision.violation.minimum_effective_assets": ("too few effective assets"),
        "decision.violation.minimum_diversification_ratio": (
            "diversification ratio too low"
        ),
        "decision.violation.minimum_cash_reserve": ("insufficient liquidity reserve"),
        "decision.constraint.maximum_annualized_volatility": (
            "Maximum annualized volatility"
        ),
        "decision.constraint.maximum_drawdown": "Maximum drawdown",
        "decision.constraint.maximum_concentration_index": "Maximum concentration",
        "decision.constraint.minimum_effective_assets": "Minimum effective assets",
        "decision.constraint.minimum_diversification_ratio": (
            "Minimum diversification ratio"
        ),
        "decision.constraint.minimum_cash_reserve": "Minimum liquidity reserve",
        "dashboard.title": "Dashboard",
        "dashboard.caption": (
            "A concise overview of the Financial Decision Support System's key results."
        ),
        "dashboard.executive.heading": "Decision summary",
        "dashboard.executive.caption": (
            "Deterministic result for the balanced reference profile with 10% of "
            "liquidity held outside the portfolio."
        ),
        "dashboard.executive.empty": (
            "The Decision Engine returned no admissible scenario. Open the Decision "
            "Workspace to review the constraints."
        ),
        "dashboard.executive.recommendation": (
            "The {scenario} scenario ranks first under the balanced assumptions."
        ),
        "dashboard.executive.scenario": "Recommended scenario",
        "dashboard.executive.return": "Annualized return — validation",
        "dashboard.executive.volatility": "Volatility — validation",
        "dashboard.executive.sharpe": "Sharpe — validation",
        "dashboard.executive.drawdown": "Maximum drawdown",
        "dashboard.executive.forecast": "RF directional accuracy",
        "dashboard.executive.validation_notice": (
            "Historical validation results. They are neither a forecast of future "
            "returns nor individualized investment advice."
        ),
        "dashboard.actions.heading": "Explore the evidence",
        "dashboard.actions.decision": "Adjust assumptions",
        "dashboard.actions.forecast": "Review forecasts",
        "dashboard.actions.market": "View the market",
        "dashboard.actions.reports": "Open reports",
        "dashboard.technical.heading": "Technical status and scope",
        "dashboard.kpi.heading": "Key indicators",
        "dashboard.kpi.direction": "RF directional accuracy",
        "dashboard.kpi.max_sharpe": "Maximum Sharpe",
        "dashboard.kpi.lowest_volatility": "Lowest volatility",
        "dashboard.kpi.validation_observations": "Validation observations",
        "dashboard.portfolio.heading": "Portfolio snapshot",
        "dashboard.portfolio.empty": "No portfolio data is available for display.",
        "dashboard.portfolio.caption": (
            "The comparison uses validation results. Optimized strategies were "
            "constructed on the training split."
        ),
        "dashboard.column.strategy": "Strategy",
        "dashboard.column.return": "Annualized return",
        "dashboard.column.volatility": "Volatility",
        "dashboard.column.sharpe": "Sharpe",
        "dashboard.column.drawdown": "Maximum drawdown",
        "dashboard.column.effective_assets": "Effective number of assets",
        "dashboard.capabilities.heading": "Component availability",
        "dashboard.column.component": "Component",
        "dashboard.column.status": "Status",
        "dashboard.capability.rag_corpus": "RAG corpus",
        "dashboard.capability.rag_benchmark": "RAG benchmark",
        "dashboard.capability.forecast_agent": "Forecast Agent",
        "dashboard.capability.risk_agent": "Risk Agent",
        "dashboard.capability.allocation_agent": "Allocation Agent",
        "dashboard.capability.agent_runtime": "Agent runtime",
        "dashboard.status.available": "available",
        "dashboard.status.ready": "ready",
        "dashboard.status.missing": "missing",
        "dashboard.status.no_data": "no data",
        "dashboard.alerts.heading": "Analytical notes",
        "dashboard.alerts.direction": (
            "Random Forest directional accuracy is close to 50%, so forecast "
            "results should be interpreted cautiously."
        ),
        "dashboard.alerts.r2": (
            "Random Forest R² is close to zero, indicating limited ability to "
            "explain variation in future returns."
        ),
        "dashboard.alerts.drawdown": (
            "Even the minimum-volatility strategy has a material historical "
            "drawdown. Low volatility does not eliminate loss risk."
        ),
        "dashboard.alerts.test_closed": (
            "The final test split remains closed. The Dashboard presents "
            "validation results."
        ),
        "dashboard.error": "The data required by the Dashboard could not be loaded.",
        "dashboard.scope.heading": "Dashboard scope and interpretation",
        "dashboard.scope.body": (
            "- The Dashboard aggregates data already exposed by existing API endpoints.\n"
            "- The frontend does not retrain models or repeat financial calculations.\n"
            "- Presented KPIs refer to the validation split.\n"
            "- Forecast, risk and portfolio results should be interpreted together.\n"
            "- Component availability means required artifacts or runtime data are present; it does not guarantee correct future model responses.\n"
            "- The final test split remains closed until final evaluation.\n"
            "- The Dashboard is analytical and does not constitute individualized investment advice."
        ),
        "common.split.training": "training",
        "common.split.validation": "validation",
        "forecast.title": "Forecasting",
        "forecast.caption": (
            "Forecast model quality evaluation on the historical validation split."
        ),
        "forecast.info": (
            "Presented results come from stored benchmark artifacts. The final "
            "test split is not used at this stage."
        ),
        "forecast.error.load": "Forecast data could not be loaded from the backend.",
        "forecast.models.empty": "The API returned no model availability data.",
        "forecast.models.heading": "Model availability",
        "forecast.models.available": "available",
        "forecast.models.missing": "missing",
        "forecast.rf.empty": "No Random Forest results for the selected instrument.",
        "forecast.rf.metrics_empty": "No Random Forest metrics are available.",
        "forecast.column.model": "Model",
        "forecast.column.observations": "Observations",
        "forecast.column.direction": "Directional accuracy",
        "forecast.ticker.metrics_empty": (
            "No model metrics are available for the selected instrument."
        ),
        "forecast.ticker.comparison_note": (
            "Directional accuracy for the Naive zero-return model is not directly "
            "comparable with models that forecast the return sign because the "
            "naive model predicts zero."
        ),
        "forecast.global.heading": "Global comparison",
        "forecast.global.empty": "No aggregate validation results are available.",
        "forecast.global.caption": (
            "Global metrics aggregate observations for all instruments included "
            "in the benchmark."
        ),
        "forecast.metric.split": "Evaluation split",
        "forecast.metric.assets": "Instruments",
        "forecast.metric.target": "Target variable",
        "forecast.ticker.heading": "Instrument analysis",
        "forecast.ticker.empty": "The API returned no supported instruments.",
        "forecast.ticker.select": "Financial instrument",
        "forecast.ticker.error": (
            "Results for the selected instrument could not be loaded."
        ),
        "forecast.ticker.context": (
            "Instrument: {ticker} · split: {split} · target: {target}"
        ),
        "forecast.ticker.models_heading": "Model comparison",
        "forecast.interpretation.heading": "Result interpretation",
        "forecast.interpretation.body": (
            "- MAE and RMSE describe forecast return error magnitude.\n"
            "- Lower MAE and RMSE mean lower predictive error.\n"
            "- R² close to zero means the model has very limited ability to explain future return variation.\n"
            "- Directional accuracy close to 50% does not indicate a strong edge in forecasting the sign of future returns.\n"
            "- Validation results support model comparison and do not guarantee future performance.\n"
            "- Feature interpretation is presented in Explainability and Governance."
        ),
        "forecast.warning": (
            "This module presents model evaluation, not investment advice. "
            "Forecast results should be interpreted together with risk analysis "
            "and model limitations."
        ),
        "risk.title": "Risk and scenarios",
        "risk.caption": (
            "Portfolio risk analysis and scenario comparison on the historical "
            "validation split."
        ),
        "risk.info": (
            "Presented results come from stored benchmark artifacts. The final "
            "test split remains closed."
        ),
        "risk.error.load": "Risk data could not be loaded from the backend.",
        "risk.scenarios.empty": "The API returned no risk scenarios.",
        "risk.comparison.heading": "Scenario comparison",
        "risk.comparison.caption": (
            "Scenarios represent predefined portfolio construction variants. "
            "The comparison uses validation results."
        ),
        "risk.scenario.select": "Scenario",
        "risk.scenario.construction": "Scenario construction: {construction}",
        "risk.scenario.error": "Scenario details could not be loaded.",
        "risk.profile.heading": "Risk profile",
        "risk.split.heading": "Training versus validation",
        "risk.split.caption": (
            "Training results are shown only to assess scenario stability. The "
            "main interpretation concerns the validation split."
        ),
        "risk.market.heading": "Market context",
        "risk.market.observations": "Observations",
        "risk.market.assets": "Assets",
        "risk.market.periods": "Periods / year",
        "risk.market.range": "Data range",
        "risk.market.instruments": "Instruments: {tickers}",
        "risk.market.return_definition": "Return definition: {definition}",
        "risk.market.correlation": "Correlation matrix",
        "risk.metric.return": "Annualized return",
        "risk.metric.volatility": "Annualized volatility",
        "risk.metric.sharpe": "Sharpe",
        "risk.metric.drawdown": "Maximum drawdown",
        "risk.metric.effective_assets": "Effective number of assets",
        "risk.metric.diversification": "Diversification ratio",
        "risk.column.scenario": "Scenario",
        "risk.column.return": "Annualized return",
        "risk.column.volatility": "Annualized volatility",
        "risk.column.sharpe": "Sharpe",
        "risk.column.sortino": "Sortino",
        "risk.column.var": "VaR",
        "risk.column.cvar": "CVaR",
        "risk.column.drawdown": "Maximum drawdown",
        "risk.column.concentration": "Concentration",
        "risk.column.effective_assets": "Effective number of assets",
        "risk.column.diversification": "Diversification ratio",
        "risk.column.split": "Split",
        "risk.column.observations": "Observations",
        "risk.interpretation.heading": "Risk measure interpretation",
        "risk.interpretation.body": (
            "- **Volatility** describes the magnitude of historical return fluctuations.\n"
            "- **VaR** indicates a loss threshold at the selected confidence level.\n"
            "- **CVaR** describes the average loss beyond the VaR threshold.\n"
            "- **Maximum drawdown** measures the largest historical decline from a local peak to a trough.\n"
            "- **Sharpe ratio** compares excess return with total volatility.\n"
            "- **Sortino ratio** focuses on adverse volatility.\n"
            "- **Concentration index** describes portfolio weight concentration.\n"
            "- **Effective number of assets** approximates the actual diversification level.\n"
            "- **Diversification ratio** compares component risk with total portfolio risk."
        ),
        "risk.limitations.heading": "Analysis limitations",
        "risk.limitations.body": (
            "- Risk measures are estimated from historical data.\n"
            "- Results may not reflect future market regimes.\n"
            "- VaR and CVaR depend on the confidence level and observed return distribution.\n"
            "- Historical correlations may change during market stress.\n"
            "- Scenarios are predefined variants, not an interactive simulation of the future.\n"
            "- Optimized strategy weights were estimated on the training split.\n"
            "- The final test split is not used in this view.\n"
            "- The module does not constitute individualized investment advice."
        ),
        "optimization.title": "Optimization",
        "optimization.caption": (
            "Comparison of allocation strategies and portfolio construction results."
        ),
        "optimization.info": (
            "Optimized strategies were estimated on the training split, while "
            "the main results presented here concern the validation split. The "
            "final test split remains closed."
        ),
        "optimization.error.load": (
            "Portfolio data could not be loaded from the backend."
        ),
        "optimization.empty": "The API returned no portfolio strategies.",
        "optimization.metric.split": "Evaluation split",
        "optimization.metric.strategies": "Strategies",
        "optimization.metric.assets": "Instruments",
        "optimization.return_definition": "Return definition: {definition}",
        "optimization.comparison.heading": "Strategy comparison",
        "optimization.comparison.caption": (
            "The comparison includes rule-based strategies and portfolios "
            "estimated from training data."
        ),
        "optimization.scenario.select": "Strategy",
        "optimization.scenario.construction": "Construction method: {construction}",
        "optimization.scenario.error": (
            "Details for the selected strategy could not be loaded."
        ),
        "optimization.validation.heading": "Validation results",
        "optimization.weights.heading": "Portfolio structure",
        "optimization.weights.empty": "No portfolio weight data is available.",
        "optimization.contributions.heading": "Asset contributions",
        "optimization.contributions.empty": (
            "No asset contribution data is available."
        ),
        "optimization.contributions.return": "Return contribution",
        "optimization.contributions.volatility": "Volatility contribution",
        "optimization.contributions.data": "Asset contribution data",
        "optimization.cumulative.heading": "Cumulative performance",
        "optimization.cumulative.empty": (
            "No portfolio performance time series is available."
        ),
        "optimization.cumulative.final": "Final cumulative return",
        "optimization.cumulative.caption": (
            "The chart presents validation performance, not final test performance."
        ),
        "optimization.split.heading": "Training versus validation",
        "optimization.split.caption": (
            "Comparing splits helps assess whether strategy parameters remain "
            "stable outside the period used for construction."
        ),
        "optimization.interpretation.heading": "Optimization interpretation",
        "optimization.interpretation.body": (
            "- **Maximum Sharpe** selects weights that maximize the historical excess-return-to-risk ratio on the training split.\n"
            "- **Minimum Volatility** minimizes historical portfolio volatility on the training split.\n"
            "- **Equal Weight** is a simple reference without weight optimization.\n"
            "- Other scenarios represent predefined portfolio construction profiles.\n"
            "- A high training Sharpe ratio does not guarantee out-of-sample performance.\n"
            "- The training–validation comparison helps assess strategy overfitting risk."
        ),
        "optimization.limitations.heading": "Limitations",
        "optimization.limitations.body": (
            "- Optimization uses historical relationships between asset returns.\n"
            "- Expected returns, volatility and correlations may change in the future.\n"
            "- Results are sensitive to the data period used for strategy construction.\n"
            "- A mathematically optimal portfolio may not be optimal for a specific investor.\n"
            "- The view does not include individual investor constraints, taxes or a complete transaction-cost model.\n"
            "- The final test split is not used during current application development.\n"
            "- Results do not constitute individualized investment advice."
        ),
        "optimization.column.strategy": "Strategy",
        "optimization.column.asset_count": "Number of assets",
        "optimization.column.instrument": "Instrument",
        "optimization.column.weight": "Weight",
        "optimization.column.return_contribution": "Annualized return contribution",
        "optimization.column.volatility_contribution": (
            "Annualized volatility contribution"
        ),
        "optimization.column.date": "Date",
        "optimization.column.portfolio_return": "Portfolio return",
        "optimization.column.cumulative_return": "Cumulative return",
        "optimization.chart.weight": "Portfolio weight",
        "optimization.chart.contribution": "Contribution",
        "governance.title": "Explainability and governance",
        "governance.caption": (
            "Interpret model behavior and control data provenance, limitations "
            "and the evaluation process."
        ),
        "governance.error.load": "XAI data could not be loaded from the backend.",
        "governance.artifact.missing": "The XAI artifact is unavailable.",
        "governance.metric.model": "Model",
        "governance.metric.split": "Explanation split",
        "governance.metric.observations": "SHAP observations",
        "governance.metric.artifact": "XAI artifact",
        "governance.status.available": "available",
        "governance.status.missing": "missing",
        "governance.target": "Target definition: {target}",
        "governance.methods": "Interpretation methods: {methods}",
        "governance.global.heading": "Global explainability",
        "governance.global.empty": "No global XAI results are available.",
        "governance.global.warning": (
            "SHAP values describe how the model uses features. They do not prove "
            "a causal relationship between a feature and future return."
        ),
        "governance.effect.heading": "Selected feature effect",
        "governance.effect.empty": "No features with PDP and ALE analyses are available.",
        "governance.effect.select": "Feature",
        "governance.effect.error": "The feature-effect analysis could not be loaded.",
        "governance.effect.pdp_empty": "No PDP data is available.",
        "governance.effect.ale_empty": "No ALE data is available.",
        "governance.effect.source": "PDP and ALE source data",
        "governance.local.heading": "Local explanation",
        "governance.local.caption": (
            "A representative local observation stored in the XAI artifact."
        ),
        "governance.local.context": (
            "**Instrument:** {ticker}  \n**Feature date:** {feature_date}  "
            "\n**Target date:** {target_date}"
        ),
        "governance.local.actual": "Actual return",
        "governance.local.predicted": "Predicted return",
        "governance.local.base": "Base value",
        "governance.local.reconstructed": "SHAP reconstruction",
        "governance.local.empty": "No local SHAP values are available.",
        "governance.local.error": "The local XAI explanation could not be loaded.",
        "governance.section.heading": "Model governance",
        "governance.section.body": (
            "**Rules for the current project phase:**\n\n"
            "- explanations concern the Random Forest model evaluated on validation data,\n"
            "- model interpretation is separated from final evaluation,\n"
            "- the final test split remains closed,\n"
            "- benchmark artifacts are the source of data presented by the API,\n"
            "- the frontend does not recompute SHAP, PDP or ALE,\n"
            "- model interpretation results are not treated as causal evidence."
        ),
        "governance.section.valid": (
            "Explanations use the validation split. The final test split has not "
            "been opened."
        ),
        "governance.section.invalid": (
            "The explanation split is not marked as validation. Data provenance "
            "must be verified."
        ),
        "governance.limitations.heading": "Interpretation limitations",
        "governance.limitations.body": (
            "- SHAP describes model behavior, not a causal market mechanism.\n"
            "- PDP may be distorted when features are strongly correlated.\n"
            "- ALE reduces some correlation problems but remains a model interpretation method.\n"
            "- A local interpretation concerns one representative observation.\n"
            "- Model results may change with the market regime.\n"
            "- Explainability does not compensate for weak predictive performance.\n"
            "- Validation results do not guarantee future performance.\n"
            "- The module supports control and interpretation, not investment advice."
        ),
        "governance.column.feature": "Feature",
        "governance.column.identifier": "Identifier",
        "governance.column.mean_shap": "Mean |SHAP|",
        "governance.column.effect": "PDP/ALE",
        "governance.column.feature_value": "Feature value",
        "governance.yes": "yes",
        "governance.no": "no",
        "governance.feature.ticker": "{ticker} identifier",
        "governance.feature.adjusted_close_to_ma_20": "Price relative to MA(20)",
        "governance.feature.momentum_20": "Momentum 20",
        "governance.feature.adjusted_close_to_ma_5": "Price relative to MA(5)",
        "governance.feature.open_to_close_return": "Open-to-close return",
        "governance.feature.daily_return": "Daily return",
        "governance.feature.volatility_5": "Volatility 5",
        "governance.feature.volatility_20": "Volatility 20",
        "governance.feature.intraday_range": "Intraday price range",
        "governance.feature.momentum_5": "Momentum 5",
        "governance.feature.volume_change": "Volume change",
        "governance.chart.mean_shap": "Mean absolute SHAP value",
        "governance.chart.feature_value": "Feature value",
        "governance.chart.average_prediction": "Average prediction",
        "governance.chart.bin_center": "Bin center",
        "governance.chart.ale_effect": "ALE effect",
        "governance.chart.shap_value": "SHAP value",
        "ai.title": "AI Insights",
        "ai.caption": "Source-grounded synthesis and structured decision-analysis support.",
        "ai.info": (
            "The module combines controlled-corpus answers with agents that "
            "interpret results from other system components."
        ),
        "ai.tab.rag": "Ask the sources",
        "ai.tab.agents": "Agent analysis",
        "ai.available": "available",
        "ai.missing": "missing",
        "ai.no_data": "no data",
        "ai.ready": "ready",
        "ai.used": "used",
        "ai.closed": "closed",
        "ai.strategy.vector": "Vector search",
        "ai.strategy.hybrid": "Hybrid search",
        "ai.strategy.reranked": "Reranked search",
        "ai.agent_type.forecast": "Forecast analysis",
        "ai.agent_type.risk": "Risk analysis",
        "ai.agent_type.allocation": "Allocation analysis",
        "ai.agent_name.forecast_analysis_agent": "Forecast analysis agent",
        "ai.agent_name.risk_analysis_agent": "Risk analysis agent",
        "ai.agent_name.allocation_recommendation_agent": "Allocation analysis agent",
        "ai.risk.defensive": "Defensive",
        "ai.risk.balanced": "Balanced",
        "ai.risk.growth": "Growth",
        "ai.timeout": "The time limit for {operation} was exceeded.",
        "ai.timeout.help": (
            "Local language-model operations have a {seconds} s limit. If the "
            "model is initializing, wait briefly and try again."
        ),
        "ai.rag.heading": "Ask the sources",
        "ai.rag.caption": (
            "Generate an answer from the controlled corpus with Retrieval-Augmented Generation."
        ),
        "ai.rag.status_error": "The RAG module status could not be loaded.",
        "ai.rag.metric.corpus": "Corpus",
        "ai.rag.metric.evaluation": "Evaluation dataset",
        "ai.rag.metric.benchmark": "Benchmark",
        "ai.rag.metric.model": "LLM model",
        "ai.rag.documents": "Corpus documents: {count}",
        "ai.rag.queries": "Evaluation queries: {count}",
        "ai.rag.question": "Question",
        "ai.rag.placeholder": "Enter a question about information in the knowledge base...",
        "ai.rag.strategy": "Search strategy",
        "ai.rag.submit": "Generate answer",
        "ai.rag.question_empty": "Enter a question before starting retrieval.",
        "ai.rag.spinner": "Retrieving sources and generating an answer...",
        "ai.rag.operation": "answer generation",
        "ai.rag.error": "The RAG module could not provide an answer.",
        "ai.rag.answer": "Answer",
        "ai.rag.answer_empty": "No answer.",
        "ai.rag.sources": "Sources",
        "ai.rag.response_strategy": "Strategy: {strategy}",
        "ai.rag.chunk_count": "Retrieved chunks: {count}",
        "ai.rag.warning": (
            "Interpret the language-model answer together with its sources. It "
            "does not constitute individualized financial advice."
        ),
        "ai.rag.citations_empty": "The answer contains no explicit source citations.",
        "ai.rag.column.document": "Document",
        "ai.rag.column.title": "Title",
        "ai.rag.column.source": "Source",
        "ai.rag.column.page": "Page",
        "ai.rag.benchmark": "Retrieval benchmark",
        "ai.rag.benchmark_current": "Benchmark-selected configuration: {strategy}",
        "ai.rag.benchmark_empty": "No detailed retrieval benchmark table is available.",
        "ai.rag.benchmark_unavailable": "The benchmark is currently unavailable.",
        "ai.rag.benchmark_note": (
            "A default configuration does not imply a quality advantage when "
            "strategies achieve the same quality result."
        ),
        "ai.rag.column.strategy": "Strategy",
        "ai.rag.column.duration": "Duration [s]",
        "ai.rag.limitations": "RAG limitations",
        "ai.rag.limitations_body": (
            "- The answer depends on information available in the controlled corpus.\n"
            "- Retrieval may return passages only partly related to the question.\n"
            "- The language model may misinterpret a correctly retrieved source.\n"
            "- Sources should be used to verify the generated answer.\n"
            "- Source material may be synthesized in the user's language.\n"
            "- Benchmark results do not guarantee quality for every future question."
        ),
        "ai.agents.heading": "Agent analysis",
        "ai.agents.caption": (
            "Structured analysis using forecast, risk, portfolio, reinforcement-learning "
            "artifacts and RAG context."
        ),
        "ai.agents.status_error": "The agent module status could not be loaded.",
        "ai.agents.runtime_missing": "Some data required by the agent runtime is unavailable.",
        "ai.agents.metric.runtime": "Runtime",
        "ai.agents.metric.model": "LLM model",
        "ai.agents.metric.retrieval": "Retrieval",
        "ai.agents.metric.test": "Final test",
        "ai.agents.evaluation_basis": "Evaluation basis: {basis}",
        "ai.agents.validation_ok": (
            "Agent analysis uses validation artifacts. The final test split remains closed."
        ),
        "ai.agents.test_warning": "Metadata indicates use of the final test split.",
        "ai.agents.type": "Analysis type",
        "ai.agents.question": "Analytical question",
        "ai.agents.placeholder.forecast": "Ask about forecast results or model quality...",
        "ai.agents.placeholder.risk": "Ask about risk or scenario comparison...",
        "ai.agents.placeholder.allocation": "Ask about allocation or strategy comparison...",
        "ai.agents.ticker_placeholder": "Optional, e.g. SPY",
        "ai.agents.ticker_help": "Leave blank to avoid limiting analysis to one instrument.",
        "ai.agents.risk_preference": "Risk preference",
        "ai.agents.submit": "Run analysis",
        "ai.agents.question_empty": "Enter a question before starting analysis.",
        "ai.agents.spinner": "The agent is analyzing available artifacts...",
        "ai.agents.operation": "agent analysis",
        "ai.agents.error": "Agent analysis could not be completed.",
        "ai.agents.workflow": "Stored validation workflow",
        "ai.agents.workflow_empty": "No stored agent workflow results are available.",
        "ai.agents.workflow_unavailable": "Stored agent results are currently unavailable.",
        "ai.agents.workflow_metadata": (
            "Model: {model} · retrieval: {retrieval} · basis: {basis}"
        ),
        "ai.agents.question_label": "Question: {question}",
        "ai.agents.summary": "Summary",
        "ai.agents.summary_empty": "No summary.",
        "ai.agents.findings": "Key findings",
        "ai.agents.risks": "Risks",
        "ai.agents.actions": "Recommended actions",
        "ai.agents.limitations_title": "Limitations",
        "ai.agents.evidence": "Evidence",
        "ai.agents.no_items": "No data.",
        "ai.agents.evidence_empty": "No explicit evidence references are available.",
        "ai.agents.column.source_type": "Source type",
        "ai.agents.column.identifier": "Identifier",
        "ai.agents.column.description": "Description",
        "ai.agents.column.role": "Role",
        "ai.agents.column.agent": "Agent",
        "ai.agents.column.summary": "Summary",
        "ai.agents.warning": (
            "The agent output provides structured analytical support. It is not "
            "individualized investment advice."
        ),
        "ai.agents.limitations": "Agent analysis limitations",
        "ai.agents.limitations_body": (
            "- Agents interpret existing analytical artifacts instead of retraining models.\n"
            "- Answers depend on forecast, risk, optimization and RAG-context quality.\n"
            "- An agent may synthesize correct source data incorrectly.\n"
            "- Interactive results may differ across model runs.\n"
            "- A risk preference is not a complete investor profile.\n"
            "- The final test split remains closed during interface development.\n"
            "- Results do not constitute individualized investment advice."
        ),
        "reports.title": "Reports",
        "reports.caption": "Generate and export a reproducible report from system results.",
        "reports.info": (
            "The report is assembled only from API data. Generation does not retrain models."
        ),
        "reports.generate": "Generate report",
        "reports.spinner": "Loading data and generating the report...",
        "reports.error": "The report could not be generated.",
        "reports.empty": "Select “Generate report” to prepare the current validation report.",
        "reports.summary": "Report summary",
        "reports.export": "Export",
        "reports.preview": "Preview",
        "reports.preview_markdown": "Show Markdown report",
        "reports.scope": "Scope",
        "reports.final_test": "Final test",
        "reports.ai_model": "AI model",
        "reports.download_markdown": "Download Markdown report",
        "reports.download_json": "Download JSON report data",
        "reports.warning": (
            "The report presents analytical and validation results. It does not "
            "constitute individualized investment advice."
        ),
    },
}


def translate(
    key: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
    **values: object,
) -> str:
    """Return one explicit translation and fail fast for catalog defects."""
    try:
        template = _TRANSLATIONS[language][key]
    except KeyError as error:
        raise KeyError(f"Missing {language!r} translation for {key!r}.") from error

    try:
        return template.format(**values)
    except KeyError as error:
        raise ValueError(
            f"Missing interpolation value for translation {key!r}."
        ) from error


def validate_translation_catalog() -> None:
    """Require exact Polish/English key parity."""
    polish_keys = set(_TRANSLATIONS["pl"])
    english_keys = set(_TRANSLATIONS["en"])

    if polish_keys != english_keys:
        missing_polish = sorted(english_keys - polish_keys)
        missing_english = sorted(polish_keys - english_keys)
        raise ValueError(
            "Translation catalogs differ. "
            f"Missing PL: {missing_polish}; missing EN: {missing_english}."
        )
