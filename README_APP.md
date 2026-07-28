# Streamlit-Dashboard – Setup & Bedienung

Ergänzung zum Haupt-README. Beschreibt die interaktive App zum Match Predictor.

## 1. Dateien ins Repo legen

Alle Dateien kommen ins **Wurzelverzeichnis** von `football-predictor/`:

```
football-predictor/
├── app.py                  ← Streamlit-Dashboard
├── train_export.py         ← trainiert Modelle & schreibt .pkl + Auswertungs-CSVs
├── live_features.py        ← rekonstruiert Elo/Form/H2H für kommende Spiele
├── paths.py                ← findet Daten- und Modellordner
├── flags.py                ← Flaggen-Emojis für die Team-Auswahl
├── .streamlit/config.toml  ← Farbschema
├── notebooks/              (unverändert)
└── ...
```

## 2. Setup auf dem Mac

```bash
cd football-predictor

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Wichtig bei Apple Silicon:** XGBoost braucht OpenMP. Falls beim Import ein
`Library not loaded: libomp.dylib` kommt:

```bash
brew install libomp
```

## 3. Modelle erzeugen (einmalig, ca. 1–3 Minuten)

Die `.pkl`-Dateien liegen wegen `.gitignore` nicht im Repo. Deshalb zuerst:

```bash
python train_export.py
```

Schreibt nach `notebooks/models_v2/`:

| Datei | Inhalt |
|---|---|
| `xgboost_smote_best.pkl` u. a. | die vier trainierten Pipelines |
| `model_results.csv` | Accuracy / F1-Macro / Log Loss je Modell |
| `confusion_matrices.csv` | Konfusionsmatrizen im Langformat |
| `feature_importances.csv` | Feature Importances je Modell |
| `feature_list.csv` | Feature-Reihenfolge (bindend für die App) |

## 4. App starten

```bash
streamlit run app.py
```

Öffnet sich unter `http://localhost:8501`.

## 5. Was die App macht

**Tab „Match Prediction"** – Zwei Nationen auswählen, Wettbewerb und neutralen
Boden in der Seitenleiste setzen, `Vorhersage berechnen` klicken. Ausgegeben
werden Siegwahrscheinlichkeiten, ein prognostizierter Endstand, die
Ausgangslage (Elo, Momentum, Form, Direktduelle) und der komplette
Feature-Vektor zum Nachvollziehen.

**Tab „Modellvergleich"** – Testmetriken der fünf Modelle inkl. Dummy-Baseline,
Balkenvergleich und Konfusionsmatrix.

**Tab „Feature Importance"** – Top-N-Features je Modell mit Kurzinterpretation.

## 6. Wie die Live-Features entstehen

Das ist der Teil, der im Notebook fehlt: Eure 43 Features existieren nur als
Zeilen für *bereits gespielte* Partien. Für ein kommendes Spiel baut
`live_features.py` den Zustand nach:

1. **Elo** – die komplette Historie ab 1994 wird mit denselben Parametern
   nachgerechnet (Start 1500, K = 40), der Endstand ist das aktuelle Rating.
2. **Form / Torstatistik** – gleitende Mittel der letzten 3 / 5 / 10 Spiele je
   Team, mit derselben Shift-Semantik wie in der Pipeline.
3. **Elo-Momentum, Serie, Erfahrung, Ruhetage** – aus derselben Historie;
   Ruhetage ergeben sich aus dem gewählten Anstoßdatum.
4. **Head-to-Head** – kumulierte Bilanz beider Nationen.
5. **Kontext** – `tournament_importance` und `is_neutral` kommen aus der
   Seitenleiste.

Der resultierende Vektor geht in exakt der Reihenfolge aus `feature_list.csv`
in die Pipeline, inklusive `StandardScaler`.

## 7. Ergebnisprognose

Zweistufig: Aus den Rolling-Torstatistiken und der Elo-Differenz werden zwei
Poisson-Parameter geschätzt und daraus ein Torgitter (0–7 Tore je Team)
gebildet. Dieses Gitter wird anschließend so umgewichtet, dass seine
1X2-Randverteilung exakt der Klassenprognose des ML-Modells entspricht. So kann
die App nie einen Endstand zeigen, der der Siegwahrscheinlichkeit widerspricht.

Das ist eine **Heuristik obendrauf**, kein trainiertes Modell – beim Vortrag
ehrlich so benennen.

## 8. Bekannte Einschränkungen (für Bericht & Verteidigung)

- **Head-to-Head ist richtungsneutral.** `h2h_home_wins` zählt, wie oft die
  jeweilige Heimmannschaft der historischen Partie gewonnen hat – nicht, wie
  oft ein bestimmtes Team gewonnen hat. Bei „Deutschland – Brasilien" zählt
  also auch das Rückspiel in Brasilien mit umgekehrter Zuordnung. Die App
  übernimmt das bewusst, damit die Features zur Trainingsverteilung passen.
- **Elo-Momentum-Default.** Die Pipeline füllt fehlende Momentum-Werte über eine
  generische Regel mit `30` auf. Für Live-Prognosen nutzt die App stattdessen
  `0.0`; betroffen sind nur Teams ohne Historie.
- **Unentschieden bleiben schwach.** Sichtbar in der Konfusionsmatrix: das
  Modell mit der besten Accuracy (XGB + SMOTE) hat den schlechtesten F1-Macro,
  weil es Remis kaum vorhersagt. Das ist ein Ergebnis, kein Bug.
- **Ruhetage werden bei 365 gekappt**, weil längere Pausen im Trainingsdatensatz
  praktisch nicht vorkommen.
- **Aktualität.** Die App rechnet auf `matches_full.csv`. Für neue Spiele erst
  die Datenpipeline erneut laufen lassen, dann `train_export.py`.
