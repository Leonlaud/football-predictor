"""
train_export.py
===============
Trainiert die Modelle aus `modelltraining_v2.ipynb` und schreibt alle
Artefakte raus, die die Streamlit-App braucht:

    models_v2/
      xgboost_smote_best.pkl
      xgboost_weighted_best.pkl
      random_forest_best.pkl
      logreg_smote_best.pkl
      feature_list.csv
      model_results.csv            (Accuracy / F1-Macro / Log Loss je Modell)
      confusion_matrices.csv       (Langformat: model, true, pred, count)
      feature_importances.csv      (Langformat: model, feature, importance)

Aufruf (einmalig, dauert ca. 1-3 Minuten):

    python train_export.py

Unterschiede zum Notebook - bewusst und minimal:
  * Jedes Modell bekommt einen *eigenen* Preprocessor. Im Notebook teilen sich
    alle Pipelines dasselbe ColumnTransformer-Objekt; sklearn klont Steps
    innerhalb einer Pipeline nicht, das Objekt wird also mehrfach ueberschrieben.
    Fuer Notebook-Ergebnisse egal, fuer gespeicherte Modelle nicht.
  * `XGB+Weights` wird als vollstaendige Pipeline gefittet, damit
    `.predict(X_roh)` spaeter funktioniert.
  * `multi_class='multinomial'` bei LogisticRegression entfaellt (seit
    scikit-learn 1.7 entfernt, ist ohnehin das Standardverhalten).
Metriken und Hyperparameter sind unveraendert.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from paths import DATA_DIR, MODEL_DIR

RANDOM_STATE = 42


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Frischer Preprocessor pro Modell - kein geteilter Zustand."""
    return ColumnTransformer([("num", StandardScaler(), feature_cols)])


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    ll = log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7))
    print(f"  {name:<16} Acc {acc:.4f} | F1-Macro {f1m:.4f} | LogLoss {ll:.4f}")
    return {"Modell": name, "Accuracy": acc, "F1-Macro": f1m, "Log Loss": ll}


def main() -> None:
    print("=" * 62)
    print("Modelltraining & Export - Fussball Match Outcome Predictor")
    print("=" * 62)
    print(f"Daten : {DATA_DIR}")
    print(f"Modelle: {MODEL_DIR}\n")

    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["date"])
    test_df = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    feature_cols = pd.read_csv(DATA_DIR / "features.csv")["feature"].tolist()

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_tv, y_tv = train_val_df[feature_cols], train_val_df["target"]
    X_tr, y_tr = train_df[feature_cols], train_df["target"]
    X_te, y_te = test_df[feature_cols], test_df["target"]

    print(f"Train+Val: {len(X_tv):,} | Test: {len(X_te):,} | Features: {len(feature_cols)}\n")

    classes = np.unique(y_tv)
    weights = compute_class_weight("balanced", classes=classes, y=y_tv)
    class_weight = {int(c): w for c, w in zip(classes, weights)}
    print(f"Class Weights: { {k: round(v, 3) for k, v in class_weight.items()} }\n")

    fitted: dict[str, object] = {}
    preds: dict[str, np.ndarray] = {}
    results: list[dict] = []

    # --- Dummy (Baseline) ------------------------------------------------
    dummy = Pipeline([
        ("preprocessor", make_preprocessor(feature_cols)),
        ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
    ]).fit(X_tr, y_tr)
    preds["Dummy"] = dummy.predict(X_te)
    results.append(evaluate("Dummy", y_te, preds["Dummy"], dummy.predict_proba(X_te)))

    # --- Logistische Regression + SMOTE ----------------------------------
    logreg = ImbPipeline([
        ("preprocessor", make_preprocessor(feature_cols)),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                   class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ]).fit(X_tv, y_tv)
    fitted["LogReg+SMOTE"] = logreg
    preds["LogReg+SMOTE"] = logreg.predict(X_te)
    results.append(evaluate("LogReg+SMOTE", y_te, preds["LogReg+SMOTE"],
                            logreg.predict_proba(X_te)))

    # --- Random Forest + Class Weights -----------------------------------
    rf = Pipeline([
        ("preprocessor", make_preprocessor(feature_cols)),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1)),
    ]).fit(X_tv, y_tv)
    fitted["Random Forest"] = rf
    preds["Random Forest"] = rf.predict(X_te)
    results.append(evaluate("Random Forest", y_te, preds["Random Forest"],
                            rf.predict_proba(X_te)))

    # --- XGBoost + SMOTE --------------------------------------------------
    xgb_smote = ImbPipeline([
        ("preprocessor", make_preprocessor(feature_cols)),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            eval_metric="mlogloss", random_state=RANDOM_STATE)),
    ]).fit(X_tv, y_tv)
    fitted["XGB+SMOTE"] = xgb_smote
    preds["XGB+SMOTE"] = xgb_smote.predict(X_te)
    results.append(evaluate("XGB+SMOTE", y_te, preds["XGB+SMOTE"],
                            xgb_smote.predict_proba(X_te)))

    # --- XGBoost + Sample Weights ----------------------------------------
    # Sample Weights muessen an den Classifier-Step durchgereicht werden.
    xgb_w = Pipeline([
        ("preprocessor", make_preprocessor(feature_cols)),
        ("clf", xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            eval_metric="mlogloss", random_state=RANDOM_STATE)),
    ])
    sample_weight = np.array([class_weight[int(y)] for y in y_tv])
    xgb_w.fit(X_tv, y_tv, clf__sample_weight=sample_weight)
    fitted["XGB+Weights"] = xgb_w
    preds["XGB+Weights"] = xgb_w.predict(X_te)
    results.append(evaluate("XGB+Weights", y_te, preds["XGB+Weights"],
                            xgb_w.predict_proba(X_te)))

    # --- Artefakte schreiben ---------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(logreg, MODEL_DIR / "logreg_smote_best.pkl")
    joblib.dump(rf, MODEL_DIR / "random_forest_best.pkl")
    joblib.dump(xgb_smote, MODEL_DIR / "xgboost_smote_best.pkl")
    joblib.dump(xgb_w, MODEL_DIR / "xgboost_weighted_best.pkl")

    pd.DataFrame({"feature": feature_cols}).to_csv(
        MODEL_DIR / "feature_list.csv", index=False)
    pd.DataFrame(results).to_csv(MODEL_DIR / "model_results.csv", index=False)

    cm_rows = []
    for name, y_pred in preds.items():
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
        for i in range(3):
            for j in range(3):
                cm_rows.append({"model": name, "true": i, "pred": j,
                                "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv(MODEL_DIR / "confusion_matrices.csv", index=False)

    imp_rows = []
    for name in ("Random Forest", "XGB+SMOTE", "XGB+Weights"):
        clf = fitted[name].named_steps["clf"]
        for feat, imp in zip(feature_cols, clf.feature_importances_):
            imp_rows.append({"model": name, "feature": feat,
                             "importance": float(imp)})
    pd.DataFrame(imp_rows).to_csv(MODEL_DIR / "feature_importances.csv", index=False)

    best = max(results, key=lambda r: r["F1-Macro"])
    print(f"\nBestes Modell nach F1-Macro: {best['Modell']}")
    print(f"Artefakte gespeichert in: {MODEL_DIR}")
    print("\nNaechster Schritt:  streamlit run app.py")


if __name__ == "__main__":
    main()
