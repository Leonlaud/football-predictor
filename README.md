# Football Match Predictor

ML-Projekt zur Vorhersage von Fußball-Spielergebnissen mit Bezug zur WM.

## Projektstruktur

```
football-predictor/
├── data/
│   ├── raw/              # Unbearbeitete Rohdaten (z.B. football-data.co.uk, Kaggle)
│   └── processed/        # Bereinigte und feature-engineerte Daten
├── notebooks/            # Jupyter Notebooks für Exploration und Prototyping
├── src/
│   ├── data/             # Skripte zum Laden und Bereinigen der Daten
│   ├── features/         # Feature-Engineering-Code
│   ├── models/           # Trainings- und Evaluationsskripte
│   └── dashboard/         # Streamlit-Dashboard
├── reports/              # Berichte, Abbildungen, Ergebnisse
└── tests/                # Tests
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Verwendung

### Daten herunterladen / vorbereiten
```bash
python src/data/load_data.py
```

### Features berechnen
```bash
python src/features/build_features.py
```

### Modelle trainieren
```bash
python src/models/train.py
```

### Dashboard starten
```bash
streamlit run src/dashboard/app.py
```

## Datenquellen

- football-data.co.uk (historische Liga-Spielergebnisse und Quoten)
- Kaggle (Datensätze zu internationalen Spielen / WM)
- https://www.kaggle.com/datasets/patateriedata/all-international-football-results
- Elo-Ratings (eloratings.net oder clubelo.com) für internationale Teamstärken

## Modelle

- Logistic Regression (Baseline)
- XGBoost / Random Forest
- Neural Network

## Team

- Leon Laudwein
- Fabian Jendrzej
