"""
paths.py
========
Loest die Pfade zu Daten und Modellen auf. Im Repo liegen die verarbeiteten
Daten aktuell unter `notebooks/data/processed_v2` (nicht unter `data/`), und
die Modelle unter `notebooks/models_v2`. Statt das hart zu verdrahten, suchen
wir die erste existierende Variante - so laeuft die App auch, wenn ihr die
Ordner spaeter aufraeumt.

Ueberschreiben per Umgebungsvariable:
    FBP_DATA_DIR=/pfad/zu/processed_v2  FBP_MODEL_DIR=/pfad/zu/models_v2 streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_CANDIDATES = [
    ROOT / "notebooks" / "data" / "processed_v2",
    ROOT / "data" / "processed_v2",
    ROOT / "data" / "processed",
    ROOT / "notebooks" / "data" / "processed",
]

MODEL_CANDIDATES = [
    ROOT / "models_v2",
    ROOT / "notebooks" / "models_v2",
    ROOT / "data" / "models",
]


def _resolve(candidates: list[Path], env_var: str, required_file: str | None) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    for path in candidates:
        if required_file is None:
            if path.is_dir():
                return path
        elif (path / required_file).exists():
            return path
    # Nichts gefunden: ersten Kandidaten zurueckgeben, damit die
    # Fehlermeldung in der App einen konkreten Pfad nennen kann.
    return candidates[0]


DATA_DIR = _resolve(DATA_CANDIDATES, "FBP_DATA_DIR", "matches_full.csv")
MODEL_DIR = _resolve(MODEL_CANDIDATES, "FBP_MODEL_DIR", "feature_list.csv")
