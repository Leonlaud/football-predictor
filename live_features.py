"""
live_features.py
================
Rekonstruiert aus der historischen Spieltabelle (`matches_full.csv`) den
aktuellen Zustand jedes Teams und baut daraus den 43-dimensionalen
Feature-Vektor fuer ein *noch nicht gespieltes* Match.

Hintergrund: Die Datenpipeline (`datenpipeline_v2_optimized.ipynb`) berechnet
alle Features zeilenweise fuer bereits gespielte Partien. Fuer eine Live-
Vorhersage gibt es diese Zeile nicht - wir muessen den Zustand nachbauen.
Die Logik hier spiegelt die Pipeline 1:1 (gleiches Elo-K, gleiche Rolling-
Fenster, gleiche Shift-Semantik), damit die Features exakt der Verteilung
entsprechen, auf der die Modelle trainiert wurden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Konstanten (identisch zur Datenpipeline v2)
# --------------------------------------------------------------------------

ELO_START = 1500.0
ELO_K = 40.0

CLASS_LABELS = {0: "Heimsieg", 1: "Unentschieden", 2: "Auswaertssieg"}

# Reihenfolge der Features - muss der features.csv entsprechen.
FEATURE_ORDER = [
    "home_elo", "away_elo", "elo_diff", "elo_diff_sq", "elo_diff_abs",
    "elo_diff_x_neutral",
    "home_elo_momentum", "away_elo_momentum", "elo_momentum_diff",
    "home_form_3", "away_form_3", "home_form_5", "away_form_5",
    "home_form_10", "away_form_10",
    "home_avg_scored_5", "home_avg_conceded_5",
    "away_avg_scored_5", "away_avg_conceded_5",
    "home_avg_scored_10", "home_avg_conceded_10",
    "away_avg_scored_10", "away_avg_conceded_10",
    "goal_balance_home", "goal_balance_away", "goal_balance_diff",
    "h2h_home_wins", "h2h_draws", "h2h_home_losses", "h2h_win_rate",
    "h2h_total",
    "tournament_importance", "is_neutral",
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_experience", "away_experience", "experience_diff",
    "home_streak", "away_streak", "home_streak_type", "away_streak_type",
]

# Turnier-Kategorien fuer die UI -> tournament_importance der Pipeline
TOURNAMENT_CATEGORIES = {
    "FIFA World Cup": 5,
    "Europameisterschaft / Kontinentalturnier": 4,
    "Nations League": 3,
    "Qualifikationsspiel": 2,
    "Sonstiges Turnier": 2,
    "Freundschaftsspiel": 1,
}

STREAK_TYPE_MAP = {"W": 2, "D": 1, "L": 0}


# --------------------------------------------------------------------------
# Team-Zustand
# --------------------------------------------------------------------------

@dataclass
class TeamState:
    """Zustand eines Teams nach seinem letzten gespielten Match."""

    name: str
    elo: float = ELO_START
    matches_played: int = 0
    last_date: date | None = None
    points: list[int] = field(default_factory=list)       # 3 / 1 / 0
    scored: list[int] = field(default_factory=list)
    conceded: list[int] = field(default_factory=list)
    results: list[str] = field(default_factory=list)      # 'W' / 'D' / 'L'
    elo_before: list[float] = field(default_factory=list)  # Elo *vor* jedem Spiel

    # -- abgeleitete Kennzahlen ------------------------------------------

    def form(self, window: int) -> float:
        """Durchschnittliche Punkte der letzten `window` Spiele."""
        if not self.points:
            return 1.0                       # Pipeline-Default fuer 'form'
        return float(np.mean(self.points[-window:]))

    def avg_scored(self, window: int) -> float:
        if not self.scored:
            return 1.0                       # Pipeline-Default fuer 'avg'
        return float(np.mean(self.scored[-window:]))

    def avg_conceded(self, window: int) -> float:
        if not self.conceded:
            return 1.0
        return float(np.mean(self.conceded[-window:]))

    def elo_momentum(self) -> float:
        """Mittlere Elo-Veraenderung der letzten 5 Spiele.

        Die Pipeline bildet `elo.diff().rolling(5).mean().shift(1)`, arbeitet
        also nur auf den Elo-Werten *vor* den Spielen. Das bauen wir hier
        identisch nach.

        Hinweis: Die Pipeline fuellt fehlende Momentum-Werte mit 30 auf
        (Nebeneffekt der generischen fillna-Regel). Das betrifft nur die
        allerersten Spiele eines Teams. Fuer Live-Prognosen waere 30 sachlich
        falsch, deshalb nutzen wir hier 0.0 - Teams im Dropdown haben ohnehin
        immer Historie.
        """
        if len(self.elo_before) < 2:
            return 0.0
        changes = np.diff(self.elo_before)
        return float(np.mean(changes[-5:]))

    def streak(self) -> tuple[float, float]:
        """(Laenge der aktuellen Serie, Typ der Serie)."""
        if not self.results:
            return 0.0, 0.0                  # Pipeline-Default fuer 'streak'
        last = self.results[-1]
        length = 0
        for r in reversed(self.results):
            if r != last:
                break
            length += 1
        return float(length), float(STREAK_TYPE_MAP[last])

    def days_rest(self, match_date: date) -> float:
        if self.last_date is None:
            return 30.0                      # Pipeline-Default
        delta = (match_date - self.last_date).days
        # Nach oben begrenzen: im Trainingsdatensatz sind Pausen > 1 Jahr
        # praktisch nicht vorhanden, extreme Werte wuerden extrapolieren.
        return float(np.clip(delta, 0, 365))


# --------------------------------------------------------------------------
# Aufbau der Zustaende aus der Historie
# --------------------------------------------------------------------------

def tournament_importance(name: str) -> int:
    """Identisch zur Funktion in der Datenpipeline v2."""
    t = str(name).lower()
    if "world cup" in t:
        return 5
    if "euro" in t or "confederations" in t:
        return 4
    if "nations league" in t:
        return 3
    if "qualification" in t or "qualifier" in t:
        return 2
    if "friendly" in t:
        return 1
    return 2


def build_history(matches: pd.DataFrame) -> tuple[dict[str, TeamState], dict[tuple, list[int]]]:
    """Spielt die komplette Historie ab und liefert Team-Zustaende + H2H-Bilanzen.

    Rueckgabe:
        team_states : {teamname: TeamState}
        h2h         : {(team_a, team_b) sortiert: [heimsiege, remis, heimniederlagen]}

    Die H2H-Zaehlung uebernimmt bewusst die Semantik der Pipeline: gezaehlt
    wird, wie oft die *jeweilige Heimmannschaft* der historischen Partie
    gewonnen hat - unabhaengig davon, welches Team das war. Die Bilanz ist
    damit richtungsneutral. (Bekannte Vereinfachung des Feature-Sets.)
    """
    df = matches.sort_values("date").reset_index(drop=True)

    states: dict[str, TeamState] = {}
    h2h: dict[tuple, list[int]] = {}

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hs, as_ = int(row.home_score), int(row.away_score)
        match_date = pd.Timestamp(row.date).date()

        h = states.setdefault(home, TeamState(home))
        a = states.setdefault(away, TeamState(away))

        # --- Elo vor dem Spiel merken, dann aktualisieren ---------------
        h.elo_before.append(h.elo)
        a.elo_before.append(a.elo)

        expected_home = 1 / (1 + 10 ** ((a.elo - h.elo) / 400))

        if hs > as_:
            score_home, h_pts, a_pts, h_res, a_res = 1.0, 3, 0, "W", "L"
        elif hs == as_:
            score_home, h_pts, a_pts, h_res, a_res = 0.5, 1, 1, "D", "D"
        else:
            score_home, h_pts, a_pts, h_res, a_res = 0.0, 0, 3, "L", "W"

        h.elo += ELO_K * (score_home - expected_home)
        a.elo += ELO_K * ((1 - score_home) - (1 - expected_home))

        # --- Rolling-Historie fortschreiben -----------------------------
        for team, pts, sc, cd, res in (
            (h, h_pts, hs, as_, h_res),
            (a, a_pts, as_, hs, a_res),
        ):
            team.points.append(pts)
            team.scored.append(sc)
            team.conceded.append(cd)
            team.results.append(res)
            team.matches_played += 1
            team.last_date = match_date
            # Speicher begrenzen - mehr als 10 Spiele braucht kein Feature
            if len(team.points) > 12:
                team.points = team.points[-12:]
                team.scored = team.scored[-12:]
                team.conceded = team.conceded[-12:]
                team.results = team.results[-12:]
                team.elo_before = team.elo_before[-12:]

        # --- Head-to-Head ----------------------------------------------
        key = tuple(sorted((home, away)))
        bilanz = h2h.setdefault(key, [0, 0, 0])
        if hs > as_:
            bilanz[0] += 1
        elif hs == as_:
            bilanz[1] += 1
        else:
            bilanz[2] += 1

    return states, h2h


# --------------------------------------------------------------------------
# Feature-Vektor fuer ein einzelnes, kommendes Match
# --------------------------------------------------------------------------

def build_match_features(
    home_team: str,
    away_team: str,
    states: dict[str, TeamState],
    h2h: dict[tuple, list[int]],
    match_date: date,
    tournament_imp: int,
    is_neutral: bool,
) -> dict[str, float]:
    """Baut den Feature-Vektor fuer eine kommende Partie."""
    h = states.get(home_team, TeamState(home_team))
    a = states.get(away_team, TeamState(away_team))

    f: dict[str, float] = {}

    # Elo -----------------------------------------------------------------
    f["home_elo"] = h.elo
    f["away_elo"] = a.elo
    f["elo_diff"] = h.elo - a.elo
    f["elo_diff_sq"] = f["elo_diff"] ** 2
    f["elo_diff_abs"] = abs(f["elo_diff"])
    f["is_neutral"] = int(is_neutral)
    f["elo_diff_x_neutral"] = f["elo_diff"] * f["is_neutral"]

    f["home_elo_momentum"] = h.elo_momentum()
    f["away_elo_momentum"] = a.elo_momentum()
    f["elo_momentum_diff"] = f["home_elo_momentum"] - f["away_elo_momentum"]

    # Form ----------------------------------------------------------------
    for w in (3, 5, 10):
        f[f"home_form_{w}"] = h.form(w)
        f[f"away_form_{w}"] = a.form(w)

    # Tore ----------------------------------------------------------------
    for w in (5, 10):
        f[f"home_avg_scored_{w}"] = h.avg_scored(w)
        f[f"home_avg_conceded_{w}"] = h.avg_conceded(w)
        f[f"away_avg_scored_{w}"] = a.avg_scored(w)
        f[f"away_avg_conceded_{w}"] = a.avg_conceded(w)

    f["goal_balance_home"] = f["home_avg_scored_5"] - f["home_avg_conceded_5"]
    f["goal_balance_away"] = f["away_avg_scored_5"] - f["away_avg_conceded_5"]
    f["goal_balance_diff"] = f["goal_balance_home"] - f["goal_balance_away"]

    # Head-to-Head --------------------------------------------------------
    wins, draws, losses = h2h.get(tuple(sorted((home_team, away_team))), [0, 0, 0])
    total = wins + draws + losses
    f["h2h_home_wins"] = float(wins)
    f["h2h_draws"] = float(draws)
    f["h2h_home_losses"] = float(losses)
    f["h2h_total"] = float(total)
    f["h2h_win_rate"] = wins / total if total > 0 else 0.33

    # Kontext -------------------------------------------------------------
    f["tournament_importance"] = float(tournament_imp)
    f["home_days_rest"] = h.days_rest(match_date)
    f["away_days_rest"] = a.days_rest(match_date)
    f["rest_diff"] = f["home_days_rest"] - f["away_days_rest"]
    f["home_experience"] = float(h.matches_played)
    f["away_experience"] = float(a.matches_played)
    f["experience_diff"] = f["home_experience"] - f["away_experience"]

    f["home_streak"], f["home_streak_type"] = h.streak()
    f["away_streak"], f["away_streak_type"] = a.streak()

    return f


def features_to_frame(features: dict[str, float], feature_order: list[str]) -> pd.DataFrame:
    """Dict -> DataFrame in exakt der Spaltenreihenfolge des Trainings."""
    return pd.DataFrame([[features[c] for c in feature_order]], columns=feature_order)


# --------------------------------------------------------------------------
# Ergebnisprognose (Poisson-Gitter, kalibriert auf die Modell-Prognose)
# --------------------------------------------------------------------------

def _poisson_pmf(lam: float, k_max: int) -> np.ndarray:
    """Poisson-Wahrscheinlichkeiten fuer 0..k_max Tore (ohne scipy)."""
    ks = np.arange(k_max + 1)
    log_p = -lam + ks * math.log(max(lam, 1e-9)) - np.array(
        [math.lgamma(k + 1) for k in ks]
    )
    return np.exp(log_p)


def predict_scoreline(
    features: dict[str, float],
    class_probs: np.ndarray,
    max_goals: int = 7,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Schaetzt die wahrscheinlichsten Endstaende.

    Zweistufig:
      1. Aus Angriffs-/Abwehrstaerke (Rolling-Tore) und Elo-Differenz werden
         die erwarteten Tore beider Teams als Poisson-Parameter geschaetzt.
      2. Das resultierende Torgitter wird so umgewichtet, dass seine
         1X2-Randverteilung exakt der Klassenprognose des ML-Modells
         entspricht. Die App zeigt damit nie einen Endstand, der der
         Siegwahrscheinlichkeit widerspricht.

    Rueckgabe: (Torgitter [heim x auswaerts], Top-Ergebnisse als [("2:1", p)])
    """
    lam_h = 0.5 * (features["home_avg_scored_5"] + features["away_avg_conceded_5"])
    lam_a = 0.5 * (features["away_avg_scored_5"] + features["home_avg_conceded_5"])

    # Elo-Korrektur: staerkeres Team trifft haeufiger
    adj = float(np.clip(features["elo_diff"] / 400.0, -1.5, 1.5))
    lam_h *= math.exp(0.25 * adj)
    lam_a *= math.exp(-0.25 * adj)

    # Heimvorteil nur, wenn nicht auf neutralem Boden gespielt wird
    if not features["is_neutral"]:
        lam_h *= 1.10
        lam_a *= 0.95

    lam_h = float(np.clip(lam_h, 0.2, 5.0))
    lam_a = float(np.clip(lam_a, 0.2, 5.0))

    grid = np.outer(_poisson_pmf(lam_h, max_goals), _poisson_pmf(lam_a, max_goals))
    grid /= grid.sum()

    # --- Umgewichtung auf die Modell-Klassenwahrscheinlichkeiten ---------
    idx_h, idx_a = np.indices(grid.shape)
    masks = [idx_h > idx_a, idx_h == idx_a, idx_h < idx_a]  # Sieg / Remis / Niederlage

    for mask, target_p in zip(masks, class_probs):
        current = grid[mask].sum()
        grid[mask] *= (target_p / current) if current > 1e-12 else 0.0
    grid /= grid.sum()

    flat = [(f"{i}:{j}", float(grid[i, j])) for i in range(grid.shape[0])
            for j in range(grid.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)

    return grid, flat[:5]
