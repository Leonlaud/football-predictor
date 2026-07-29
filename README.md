# ⚽ Football Match Outcome Predictor

Ein Machine-Learning-Projekt zur **Vorhersage von Fußball-Spielergebnissen** mit besonderem Fokus auf **internationale Länderspiele und Weltmeisterschaften**. Ziel des Projekts ist es, mithilfe historischer Spieldaten, Elo-Ratings und Feature Engineering die Wahrscheinlichkeit von Heimsieg, Unentschieden oder Auswärtssieg vorherzusagen.

---

# 📖 Projektübersicht

Im Rahmen des Moduls **KI & Intelligence Engineering** an der **DHBW Mannheim** wurde eine vollständige Machine-Learning-Pipeline entwickelt, bestehend aus:

- automatisierter Datenpipeline
- Feature Engineering
- Elo-Rating-Berechnung
- Modelltraining und Evaluation
- Hyperparameter-Optimierung
- Visualisierung der Ergebnisse
- interaktivem Dashboard zur Live-Vorhersage (Streamlit)

Im Laufe des Projekts entstanden zwei Versionen der Pipeline:

- **Version 1** als Basisimplementierung
- **Version 2** mit Performance-Optimierungen, erweiterten Features sowie Methoden zum Umgang mit Klassenungleichgewichten (SMOTE & Class Weights)

Darauf aufbauend steht ein **Streamlit-Dashboard**, mit dem sich beliebige Länderspiele interaktiv vorhersagen lassen.

---

# 📂 Projektstruktur

```text
football-predictor/
│
├── app.py                       # Streamlit-Dashboard
├── train_export.py              # Trainiert die Modelle & exportiert .pkl + Auswertungs-CSVs
├── live_features.py             # Rekonstruiert Elo/Form/H2H für kommende Spiele
├── paths.py                     # Löst Daten-/Modellpfade auf
├── flags.py                     # Flaggen-Zuordnung für die Team-Auswahl
├── .streamlit/
│   └── config.toml              # Farbschema des Dashboards
│
├── notebooks/
│   ├── datenpipeline_v2_optimized.ipynb
│   ├── modelltraining_v2.ipynb
│   ├── data/
│   │   └── processed_v2/        # Aufbereitete Datensätze (u. a. matches_full.csv)
│   └── models_v2/                # Modelle (.pkl, lokal erzeugt) & Auswertungs-CSVs
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Repository klonen:

```bash
git clone <repository-url>
cd football-predictor
```

Virtuelle Umgebung erstellen:

```bash
python -m venv venv
```

Aktivieren:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Benötigte Pakete installieren:

```bash
pip install -r requirements.txt
```

---

# ⚙️ Verwendung

## 1️⃣ Datenpipeline (Version 1)

Bereitet die historischen Spieldaten auf und erstellt den Trainingsdatensatz.

```bash
python src/datenpipeline.py
```

---

## 2️⃣ Optimierte Datenpipeline (Version 2)

Erstellt den Datensatz mit einer optimierten Pipeline inklusive effizientem Feature Engineering.

```bash
python src/datenpipeline_v2_optimized.py
```

---

## 3️⃣ Modelltraining (Version 1)

Trainiert mehrere Machine-Learning-Modelle und vergleicht deren Leistung.

```bash
python src/modelltraining.py
```

---

## 4️⃣ Optimiertes Modelltraining (Version 2)

Trainiert die Modelle unter Verwendung von:

- SMOTE
- Class Weights
- erweiterten Features
- zusätzlichen Visualisierungen

```bash
python src/modelltraining_v2.py
```

---

## 5️⃣ Modelle für das Dashboard erzeugen

Das Dashboard braucht trainierte Modelle als `.pkl`-Dateien. Diese liegen aus
Größengründen nicht im Repository und müssen einmalig lokal erzeugt werden:

```bash
python train_export.py
```

Schreibt nach `notebooks/models_v2/`:

- `xgboost_smote_best.pkl`, `xgboost_weighted_best.pkl`, `random_forest_best.pkl`, `logreg_smote_best.pkl`
- `model_results.csv` (Accuracy, F1-Macro, Log Loss je Modell)
- `confusion_matrices.csv`
- `feature_importances.csv`

---

## 6️⃣ Dashboard starten

```bash
streamlit run app.py
```

Öffnet sich unter `http://localhost:8501`. Zwei Nationen auswählen, Wettbewerb
und Spielort in der Seitenleiste einstellen und **Vorhersage berechnen**
klicken. Das Dashboard zeigt:

- **Match Prediction** – Siegwahrscheinlichkeiten, prognostizierter Endstand, Ausgangslage (Elo, Form, Head-to-Head)
- **Modellvergleich** – Testmetriken und Konfusionsmatrix aller Modelle
- **Feature Importance** – wichtigste Einflussfaktoren je Modell

Da die 43 Trainings-Features nur für bereits gespielte Partien existieren,
rekonstruiert `live_features.py` den aktuellen Zustand beider Teams
(Elo, Form, Torstatistik, Head-to-Head) aus der kompletten Spielhistorie.

---

# 🧠 Verwendete Modelle

Folgende Klassifikationsmodelle werden untersucht:

- Logistic Regression (Baseline)
- Random Forest
- XGBoost

Die Modelle werden unter anderem anhand folgender Metriken verglichen:

- Accuracy
- F1-Score (Macro)
- Log Loss

---

# ⚙️ Feature Engineering

Zur Vorhersage werden verschiedene Merkmale erzeugt:

- Elo-Rating beider Teams
- Elo-Differenz
- Heimvorteil
- Rolling-Statistiken
- Form der letzten Spiele
- Durchschnittliche Tore
- Sieg-, Remis- und Niederlagenquoten
- Weitere statistische Leistungsmerkmale

---

# 📊 Optimierungen in Version 2

Die zweite Projektversion erweitert die ursprüngliche Pipeline durch:

- ✅ Optimierte Datenpipeline ohne aufwendige Merge-Operationen
- ✅ Speicheroptimierte Verarbeitung
- ✅ Erweiterte Features
- ✅ SMOTE zur Klassenbalancierung
- ✅ Class Weights
- ✅ Umfangreiche Visualisierungen
- ✅ Verbesserte Modellrobustheit

---

# 📁 Datenquellen

Die verwendeten Datensätze stammen aus öffentlich verfügbaren Quellen:

- **GitHub:** Internationale Fußballergebnisse (`martj42/international_results`)
- **Kaggle:** Internationale Fußballspiele und Weltmeisterschaften
  - https://www.kaggle.com/datasets/patateriedata/all-international-football-results
- **Elo-Ratings:** Historische Teamstärken

---

# 🛠️ Verwendete Technologien

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook
- Streamlit
- Plotly
- Pycountry

---

# 📈 Projektablauf

```text
Historische Spieldaten
            │
            ▼
      Datenpipeline
            │
            ▼
   Feature Engineering
            │
            ▼
      Elo-Berechnung
            │
            ▼
     Trainingsdatensatz
            │
            ▼
     Modelltraining
            │
            ▼
 Hyperparameter-Tuning
            │
            ▼
      Modellvergleich
            │
            ▼
    Bestes Modell
            │
            ▼
  Streamlit-Dashboard
```

---

# 👥 Projektteam

**Leon Laudwein**  
**Fabian Jendrzej**
**Lukas Ruth**

---

# 🎓 Kontext

Dieses Projekt wurde im Rahmen des Moduls **KI & Intelligence Engineering** an der **DHBW Mannheim** entwickelt.
