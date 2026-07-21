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

Im Laufe des Projekts entstanden zwei Versionen der Pipeline:

- **Version 1** als Basisimplementierung
- **Version 2** mit Performance-Optimierungen, erweiterten Features sowie Methoden zum Umgang mit Klassenungleichgewichten (SMOTE & Class Weights)

---

# 📂 Projektstruktur

```text
football-match-outcome-predictor/
│
├── data/
│   ├── raw/                     # Rohdaten
│   ├── processed/               # Aufbereitete Datensätze
│   └── models/                  # Gespeicherte ML-Modelle
│
├── notebooks/                   # Jupyter Notebooks
│
├── src/
│   ├── datenpipeline.py
│   ├── datenpipeline_v2_optimized.py
│   ├── modelltraining.py
│   └── modelltraining_v2.py
│
├── reports/
│   ├── figures/                 # Visualisierungen
│   └── results/                 # Evaluationsergebnisse
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Repository klonen:

```bash
git clone <repository-url>
cd football-match-outcome-predictor
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
```

---

# 👥 Projektteam

**Leon Laudwein**  
**Fabian Jendrzej**
**Lukas Ruth**

---

# 🎓 Kontext

Dieses Projekt wurde im Rahmen des Moduls **KI & Intelligence Engineering** an der **DHBW Mannheim** entwickelt.
