"""
app.py
======
Streamlit-Dashboard zum Fussball Match Outcome Predictor (DHBW Mannheim,
KI & Intelligence Engineering, SoSe 2026).

Start:
    streamlit run app.py

Voraussetzung: `python train_export.py` wurde einmal ausgefuehrt, damit die
trainierten Modelle als .pkl vorliegen.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from flags import flag, with_flag
from live_features import (
    CLASS_LABELS,
    TOURNAMENT_CATEGORIES,
    build_history,
    build_match_features,
    features_to_frame,
    predict_scoreline,
)
from paths import DATA_DIR, MODEL_DIR

# --------------------------------------------------------------------------
# Grundeinstellungen & Design
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dunkles Farbschema. Alle Schriften liegen auf hellen Toenen, damit sie
# unabhaengig vom Betriebssystem-Theme lesbar bleiben.
BG = "#0B1F14"          # Seitenhintergrund
SURFACE = "#12301F"     # Karten / Panels
BORDER = "#1F4D33"      # Rahmen
TEXT = "#F2FBF5"        # Primaerschrift (nahezu weiss)
TEXT_MUTED = "#A8C6B4"  # Sekundaerschrift
GREEN = "#22A15A"       # Balken / Akzentflaechen
GREEN_LIGHT = "#4ADE80"  # Akzentschrift
DARK = SURFACE           # Alias, damit aeltere Referenzen weiter greifen

MODEL_FILES = {
    "XGBoost + SMOTE": "xgboost_smote_best.pkl",
    "XGBoost + Class Weights": "xgboost_weighted_best.pkl",
    "Random Forest": "random_forest_best.pkl",
    "Logistische Regression": "logreg_smote_best.pkl",
}

# Anzeigename in der App -> Modellname in model_results.csv
RESULT_KEYS = {
    "XGBoost + SMOTE": "XGB+SMOTE",
    "XGBoost + Class Weights": "XGB+Weights",
    "Random Forest": "Random Forest",
    "Logistische Regression": "LogReg+SMOTE",
}

st.markdown(
    f"""
    <style>
      /* Dunkles Theme fest verdrahten. Alle Schriften liegen bewusst auf
         hellen Toenen (nahezu weiss), damit sie unabhaengig davon lesbar
         bleiben, ob Browser oder Betriebssystem einen Dark Mode erzwingen. */
      [data-testid="stAppViewContainer"], [data-testid="stHeader"],
      [data-testid="stBottomBlockContainer"], .main {{
        background-color: {BG} !important;
        color-scheme: dark;
      }}
      [data-testid="stSidebar"] {{
        background-color: {SURFACE} !important;
        border-right: 1px solid {BORDER};
      }}

      /* Grundschrift durchgaengig hell */
      [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
      .stMarkdown, .stMarkdown p, .stMarkdown li,
      h1, h2, h3, h4, h5, h6, label, .stSelectbox label {{
        color: {TEXT} !important;
      }}
      [data-testid="stCaptionContainer"], .stCaption, small {{
        color: {TEXT_MUTED} !important;
      }}
      code {{ color: {GREEN_LIGHT} !important; background: rgba(74,222,128,0.10) !important; }}

      .block-container {{ padding-top: 1.4rem; max-width: 1400px; }}

      .app-header {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.6rem;
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 1.4rem;
      }}
      .app-header h1 {{
        color: {TEXT}; font-size: 1.55rem; font-weight: 700;
        margin: 0; letter-spacing: -0.01em;
      }}
      .app-header .sub {{ color: {GREEN_LIGHT}; font-size: 0.82rem; margin-top: 0.15rem; }}
      .badge {{
        background: rgba(74,222,128,0.14); color: {GREEN_LIGHT};
        border: 1px solid rgba(74,222,128,0.35); border-radius: 999px;
        padding: 0.35rem 0.9rem; font-size: 0.8rem; font-weight: 600;
        white-space: nowrap;
      }}

      .card {{
        background: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 1rem 1.2rem; height: 100%;
      }}
      .card h4 {{ margin: 0 0 0.2rem 0; color: {TEXT}; font-size: 0.95rem; }}
      .card p  {{ margin: 0; color: {TEXT_MUTED}; font-size: 0.82rem; }}

      .probrow {{ display: flex; align-items: center; gap: 0.8rem; margin: 0.55rem 0; }}
      .probrow .lbl {{ width: 230px; font-size: 0.92rem; color: {TEXT}; font-weight: 500; }}
      .probrow .track {{ flex: 1; background: rgba(255,255,255,0.10); border-radius: 999px; height: 13px; }}
      .probrow .fill  {{ background: {GREEN_LIGHT}; border-radius: 999px; height: 13px; }}
      .probrow .val   {{ width: 66px; text-align: right; font-weight: 700; color: {TEXT}; }}

      .scorebox {{
        border: 1px solid {BORDER}; border-radius: 10px; text-align: center;
        padding: 0.6rem 0.2rem; background: {SURFACE};
      }}
      .scorebox .s {{ font-size: 1.35rem; font-weight: 700; color: {TEXT}; }}
      .scorebox .p {{ font-size: 0.8rem; color: {GREEN_LIGHT}; font-weight: 600; }}

      .callout {{
        background: {SURFACE}; color: {TEXT};
        border: 1px solid {BORDER}; border-left: 4px solid {GREEN_LIGHT};
        border-radius: 12px; padding: 0.9rem 1.2rem; margin-top: 0.6rem;
      }}
      .callout b {{ color: {GREEN_LIGHT}; }}

      /* Metriken */
      div[data-testid="stMetricValue"] {{ color: {TEXT} !important; }}
      div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{
        color: {TEXT_MUTED} !important;
      }}

      /* Tabs */
      button[data-baseweb="tab"] {{ color: {TEXT_MUTED} !important; }}
      button[data-baseweb="tab"][aria-selected="true"] {{ color: {GREEN_LIGHT} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Laden (gecacht)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Lade Spielhistorie …")
def load_matches() -> pd.DataFrame:
    path = DATA_DIR / "matches_full.csv"
    if not path.exists():
        return pd.DataFrame()
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "target"]
    return pd.read_csv(path, usecols=cols, parse_dates=["date"])


@st.cache_resource(show_spinner="Berechne Team-Zustaende (Elo, Form, H2H) …")
def load_history(_matches: pd.DataFrame):
    return build_history(_matches)


@st.cache_resource(show_spinner="Lade Modelle …")
def load_models() -> dict:
    models = {}
    for label, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[label] = joblib.load(path)
    return models


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = MODEL_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_feature_order() -> list[str]:
    df = load_csv("feature_list.csv")
    if df.empty:
        df = pd.read_csv(DATA_DIR / "features.csv")
    return df["feature"].tolist()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

matches = load_matches()
models = load_models()

if matches.empty:
    st.error(
        f"**Keine Spieldaten gefunden.** Erwartet wird `matches_full.csv` in "
        f"`{DATA_DIR}`. Führt die Datenpipeline aus "
        f"(`notebooks/datenpipeline_v2_optimized.ipynb`) oder setzt "
        f"`FBP_DATA_DIR` auf den richtigen Ordner."
    )
    st.stop()

if not models:
    st.error(
        f"**Keine trainierten Modelle gefunden** in `{MODEL_DIR}`. "
        f"Führt einmalig `python train_export.py` aus – das Skript trainiert "
        f"die Modelle aus euren CSVs und legt die `.pkl`-Dateien dort ab."
    )
    st.stop()

last_match = matches["date"].max().date()

st.markdown(
    f"""
    <div class="app-header">
      <div>
        <h1>⚽ Match Predictor</h1>
        <div class="sub">Länderspiel-Prognose · DHBW Mannheim · KI &amp; Intelligence Engineering</div>
      </div>
      <div class="badge">Datenstand: {last_match.strftime('%d.%m.%Y')} · {len(matches):,} Spiele</div>
    </div>
    """.replace(",", "."),
    unsafe_allow_html=True,
)

states, h2h = load_history(matches)
feature_order = load_feature_order()
teams = sorted(states.keys())

# --------------------------------------------------------------------------
# Sidebar: Modell- und Spieleinstellungen
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Modell")
    model_label = st.selectbox(
        "Welches Modell rechnet?",
        list(models.keys()),
        key="model_choice",
        help="XGBoost + SMOTE erzielt im Test die beste Accuracy, "
             "LogReg + SMOTE den besten F1-Macro über alle drei Klassen.",
    )
    model = models[model_label]

    st.divider()
    st.subheader("Spielkontext")
    tournament_label = st.selectbox(
        "Wettbewerb", list(TOURNAMENT_CATEGORIES.keys()), index=0
    )
    tournament_imp = TOURNAMENT_CATEGORIES[tournament_label]

    neutral = st.toggle(
        "Neutraler Boden",
        value=(tournament_imp == 5),
        help="Bei WM-Spielen findet die Partie meist auf neutralem Platz statt – "
             "der Heimvorteil entfällt dann.",
    )
    match_date = st.date_input(
        "Anstoßdatum",
        value=max(last_match + timedelta(days=3), date.today()),
        help="Bestimmt, wie viele Tage Pause beide Teams hatten.",
    )

    st.divider()
    st.caption(
        f"Daten: `{DATA_DIR.name}` · Modelle: `{MODEL_DIR.name}`\n\n"
        "Klassen: 0 = Heimsieg · 1 = Unentschieden · 2 = Auswärtssieg"
    )


# --------------------------------------------------------------------------
# Hilfsfunktionen fuer die Darstellung
# --------------------------------------------------------------------------

def prob_bar(label: str, value: float) -> str:
    return (
        f'<div class="probrow">'
        f'<div class="lbl">{label}</div>'
        f'<div class="track"><div class="fill" style="width:{value * 100:.1f}%"></div></div>'
        f'<div class="val">{value * 100:.1f} %</div>'
        f"</div>"
    )


def confidence_label(probs: np.ndarray) -> tuple[str, str]:
    """Grobe Einordnung, wie eindeutig die Prognose ist."""
    top = float(np.max(probs))
    if top >= 0.60:
        return "hoch", GREEN_LIGHT
    if top >= 0.45:
        return "mittel", "#F0C05A"
    return "niedrig", "#F08A72"


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_predict, tab_compare, tab_importance = st.tabs(
    ["Match Prediction", "Modellvergleich", "Feature Importance"]
)

# ==========================================================================
# TAB 1 – Match Prediction
# ==========================================================================
with tab_predict:
    # Bewusst keine Vorbelegung: die App startet mit leerer Auswahl.
    c1, c2, c3 = st.columns([5, 1, 5])
    with c1:
        home_team = st.selectbox(
            "Heimmannschaft", teams, index=None,
            placeholder="Mannschaft wählen …",
            format_func=with_flag, key="home_team",
        )
    with c2:
        st.markdown(
            f"<div style='text-align:center;padding-top:2.1rem;"
            f"font-weight:700;color:{TEXT_MUTED}'>vs</div>",
            unsafe_allow_html=True,
        )
    with c3:
        away_team = st.selectbox(
            "Auswärtsmannschaft", teams, index=None,
            placeholder="Mannschaft wählen …",
            format_func=with_flag, key="away_team",
        )

    teams_chosen = bool(home_team) and bool(away_team)
    same_team = teams_chosen and home_team == away_team

    run = st.button(
        "Vorhersage berechnen  ▶", type="primary", width="stretch",
        disabled=not teams_chosen or same_team,
    )

    if same_team:
        st.warning("Bitte zwei verschiedene Mannschaften auswählen.")

    # Signatur der aktuellen Eingaben - erkennt, ob ein gespeichertes
    # Ergebnis noch zu den Einstellungen in der Seitenleiste passt.
    current_sig = (
        home_team, away_team, tournament_imp, bool(neutral),
        match_date.isoformat(), model_label,
    )

    if run and teams_chosen and not same_team:
        features = build_match_features(
            home_team, away_team, states, h2h,
            match_date=match_date,
            tournament_imp=tournament_imp,
            is_neutral=neutral,
        )
        X = features_to_frame(features, feature_order)
        probs = model.predict_proba(X)[0]
        # Ergebnis merken, damit es einen Tab-Wechsel ueberlebt. Streamlit
        # fuehrt bei jeder Interaktion das ganze Skript neu aus; ohne
        # session_state waere `run` dann wieder False und die Ausgabe weg.
        st.session_state["prediction"] = {
            "sig": current_sig, "home": home_team, "away": away_team,
            "features": features, "probs": probs,
        }

    saved = st.session_state.get("prediction")

    if saved is not None:
        home_team = saved["home"]
        away_team = saved["away"]
        features = saved["features"]
        probs = saved["probs"]

        if saved["sig"] != current_sig:
            st.info(
                "Die Einstellungen haben sich geändert – das Ergebnis unten "
                "stammt noch aus der letzten Berechnung. Für aktuelle Werte "
                "erneut **Vorhersage berechnen** klicken.",
                icon="🔄",
            )

        conf_text, conf_color = confidence_label(probs)
        st.markdown(
            f"<div style='text-align:right;color:{conf_color};font-weight:600;"
            f"margin:0.4rem 0 0.2rem 0'>Modellvertrauen: {conf_text}</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns([3, 2])

        # ---- Siegwahrscheinlichkeiten --------------------------------
        with left:
            st.markdown("##### Siegwahrscheinlichkeiten")
            html = "".join([
                prob_bar(f"{flag(home_team)} {home_team} gewinnt", probs[0]),
                prob_bar("Unentschieden", probs[1]),
                prob_bar(f"{flag(away_team)} {away_team} gewinnt", probs[2]),
            ])
            st.markdown(html, unsafe_allow_html=True)

            # ---- Ergebnisprognose ------------------------------------
            _, top_scores = predict_scoreline(features, probs)
            st.markdown("##### Ergebnisprognose")
            best_score, best_p = top_scores[0]
            st.markdown(
                f'<div class="callout">Wahrscheinlichster Endstand: '
                f"<b>{home_team} {best_score.replace(':', ' : ')} {away_team}</b> "
                f"&nbsp;·&nbsp; {best_p * 100:.1f} %</div>",
                unsafe_allow_html=True,
            )

            cols = st.columns(len(top_scores))
            for col, (score, p) in zip(cols, top_scores):
                col.markdown(
                    f'<div class="scorebox"><div class="s">{score}</div>'
                    f'<div class="p">{p * 100:.1f} %</div></div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "Der Endstand wird über ein Poisson-Modell aus den Rolling-Tor-"
                "statistiken geschätzt und anschließend so umgewichtet, dass er "
                "exakt zur Klassenprognose des ML-Modells passt."
            )

        # ---- Kontext zum Duell ---------------------------------------
        with right:
            st.markdown("##### Ausgangslage")
            m1, m2 = st.columns(2)
            m1.metric(f"Elo {home_team}", f"{features['home_elo']:.0f}",
                      f"{features['home_elo_momentum']:+.1f} Momentum")
            m2.metric(f"Elo {away_team}", f"{features['away_elo']:.0f}",
                      f"{features['away_elo_momentum']:+.1f} Momentum")

            m3, m4 = st.columns(2)
            m3.metric("Form (Ø Punkte, 5 Spiele)", f"{features['home_form_5']:.2f}")
            m4.metric("Form (Ø Punkte, 5 Spiele)", f"{features['away_form_5']:.2f}")

            total = int(features["h2h_total"])
            if total:
                st.markdown(
                    f"**Direkte Duelle:** {total} Spiele – "
                    f"{int(features['h2h_home_wins'])} Heimsiege, "
                    f"{int(features['h2h_draws'])} Remis, "
                    f"{int(features['h2h_home_losses'])} Auswärtssiege"
                )
                st.caption(
                    "Bilanz aus Sicht der jeweiligen Heimmannschaft der "
                    "historischen Partie – das Feature ist richtungsneutral."
                )
            else:
                st.markdown("**Direkte Duelle:** keine seit 1994")

            recent = matches[
                ((matches.home_team == home_team) & (matches.away_team == away_team))
                | ((matches.home_team == away_team) & (matches.away_team == home_team))
            ].sort_values("date", ascending=False).head(5)

            if not recent.empty:
                st.markdown("**Letzte Begegnungen**")
                view = pd.DataFrame({
                    "Datum": recent["date"].dt.strftime("%d.%m.%Y"),
                    "Partie": recent["home_team"] + " – " + recent["away_team"],
                    "Ergebnis": recent["home_score"].astype(str) + ":"
                                + recent["away_score"].astype(str),
                    "Wettbewerb": recent["tournament"],
                })
                st.dataframe(view, hide_index=True, width="stretch")

        # ---- Rohe Features -------------------------------------------
        with st.expander("Verwendeter Feature-Vektor (43 Features)"):
            st.dataframe(
                pd.DataFrame({"Feature": feature_order,
                              "Wert": [round(features[c], 4) for c in feature_order]}),
                hide_index=True, width="stretch", height=420,
            )
    else:
        st.info(
            "Mannschaften wählen, Wettbewerb in der Seitenleiste einstellen und "
            "**Vorhersage berechnen** klicken. Elo, Form, Torstatistik und "
            "Head-to-Head werden automatisch aus der Historie rekonstruiert."
        )


# ==========================================================================
# TAB 2 – Modellvergleich
# ==========================================================================
with tab_compare:
    results = load_csv("model_results.csv")

    if results.empty:
        st.warning("`model_results.csv` fehlt – bitte `python train_export.py` ausführen.")
    else:
        st.markdown("##### Testmetriken (Zeitraum ab 2022, ungesehene Daten)")

        col_table, col_chart = st.columns([3, 2])

        with col_table:
            styled = results.copy()
            for c in ("Accuracy", "F1-Macro", "Log Loss"):
                styled[c] = styled[c].round(4)
            st.dataframe(styled, hide_index=True, width="stretch")

        with col_chart:
            plot_df = results[results["Modell"] != "Dummy"]
            fig = go.Figure()
            fig.add_bar(x=plot_df["Modell"], y=plot_df["Accuracy"],
                        name="Accuracy", marker_color=GREEN)
            fig.add_bar(x=plot_df["Modell"], y=plot_df["F1-Macro"],
                        name="F1-Macro", marker_color=GREEN_LIGHT)
            fig.update_layout(
                barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[0, 0.8], title=None,
                           gridcolor="rgba(255,255,255,0.10)"),
                legend=dict(orientation="h", y=1.15, x=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig, width="stretch")

        best_acc = results.loc[results["Accuracy"].idxmax()]
        best_f1 = results.loc[results["F1-Macro"].idxmax()]
        st.markdown(
            f'<div class="callout">Beste Accuracy: <b>{best_acc["Modell"]}</b> '
            f'({best_acc["Accuracy"]:.1%}) &nbsp;·&nbsp; bester F1-Macro: '
            f'<b>{best_f1["Modell"]}</b> ({best_f1["F1-Macro"]:.3f})</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Accuracy und F1-Macro zeigen unterschiedliche Sieger: XGBoost + SMOTE "
            "trifft insgesamt häufiger richtig, vernachlässigt aber die "
            "Unentschieden. Die logistische Regression verteilt ihre Vorhersagen "
            "gleichmäßiger über alle drei Klassen – deshalb der höhere F1-Macro."
        )

        # ---- Konfusionsmatrix -----------------------------------------
        cms = load_csv("confusion_matrices.csv")
        if not cms.empty:
            st.divider()
            st.markdown("##### Konfusionsmatrix")
            cm_models = sorted(cms["model"].unique())
            preselect = RESULT_KEYS.get(model_label)
            choice = st.selectbox(
                "Modell",
                cm_models,
                index=cm_models.index(preselect) if preselect in cm_models else 0,
                key="cm_model",
                help="Vorbelegt mit dem Modell aus der Seitenleiste.",
            )
            sub = cms[cms["model"] == choice]
            matrix = sub.pivot(index="true", columns="pred", values="count").values
            labels = ["Heimsieg", "Unentschieden", "Auswärtssieg"]
            row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
            text = [[f"{matrix[i][j]}<br>{row_pct[i][j]:.1f} %" for j in range(3)]
                    for i in range(3)]

            fig = go.Figure(go.Heatmap(
                z=matrix, x=labels, y=labels, text=text, texttemplate="%{text}",
                colorscale=[[0, SURFACE], [1, GREEN_LIGHT]], showscale=False,
            ))
            fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Vorhergesagt", yaxis_title="Tatsächlich",
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Zeilen = tatsächliches Ergebnis, Spalten = Vorhersage. "
                "Die mittlere Zeile zeigt das Kernproblem: Unentschieden werden "
                "systematisch als Sieg einer Seite vorhergesagt."
            )


# ==========================================================================
# TAB 3 – Feature Importance
# ==========================================================================
with tab_importance:
    imps = load_csv("feature_importances.csv")

    if imps.empty:
        st.warning("`feature_importances.csv` fehlt – bitte `python train_export.py` ausführen.")
    else:
        col_sel, col_n = st.columns([3, 1])
        with col_sel:
            imp_model = st.selectbox("Modell", sorted(imps["model"].unique()), key="imp_model")
        with col_n:
            top_n = st.slider("Top N", 5, 43, 15)

        sub = (imps[imps["model"] == imp_model]
               .sort_values("importance", ascending=False)
               .head(top_n)
               .sort_values("importance"))

        chart_col, text_col = st.columns([3, 2])

        with chart_col:
            fig = go.Figure(go.Bar(
                x=sub["importance"], y=sub["feature"], orientation="h",
                marker_color=GREEN_LIGHT,
                text=[f"{v:.3f}" for v in sub["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=max(360, 26 * len(sub)),
                margin=dict(l=10, r=40, t=20, b=10),
                xaxis_title="Wichtigkeit (relativ)", yaxis_title=None,
                xaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig, width="stretch")

        with text_col:
            top3 = (imps[imps["model"] == imp_model]
                    .sort_values("importance", ascending=False)
                    .head(3)["feature"].tolist())
            st.markdown("##### Kurzinterpretation")
            st.markdown(
                f"- Die drei stärksten Einflussfaktoren sind "
                f"`{top3[0]}`, `{top3[1]}` und `{top3[2]}`.\n"
                "- Elo-basierte Features dominieren – sie fassen die gesamte "
                "Spielhistorie in einer Zahl zusammen.\n"
                "- Form- und Torfeatures liefern zusätzliche Kurzfrist-Information, "
                "sind untereinander aber stark korreliert.\n"
                "- Kontextfeatures wie `is_neutral` oder `tournament_importance` "
                "haben spürbar geringeres Gewicht – der Heimvorteil steckt bei "
                "Länderspielen teilweise schon im Elo."
            )
            st.info(
                "Achtung bei der Interpretation: Baum-basierte Importances "
                "bevorzugen Features mit vielen unterschiedlichen Werten. "
                "Für die Ausarbeitung wäre eine Permutation Importance oder "
                "SHAP die belastbarere Aussage.",
                icon="ℹ️",
            )
