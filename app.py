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

DARK = "#0F2A1D"
GREEN = "#1B7F4C"
GREEN_LIGHT = "#4ADE80"
GREEN_PALE = "#EAF7EF"
GREY = "#6B7C72"

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
      /* Hintergrund fest auf Weiss setzen - unabhaengig davon, ob der
         Browser oder das Betriebssystem einen Dark Mode erzwingt. Ohne das
         ueberschreibt z.B. Safari/Chrome im Dark Mode den Seitenhintergrund
         auf dunkel, waehrend unsere Textfarben fuer hellen Hintergrund
         gestaltet sind - das Ergebnis waere dunkler Text auf dunklem Grund. */
      [data-testid="stAppViewContainer"], [data-testid="stHeader"],
      [data-testid="stSidebar"], .main {{
        background-color: #FFFFFF !important;
        color-scheme: light;
      }}
      .block-container {{ padding-top: 1.4rem; max-width: 1400px; }}

      .app-header {{
        background: {DARK};
        border-radius: 14px;
        padding: 1.1rem 1.6rem;
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 1.4rem;
      }}
      .app-header h1 {{
        color: #FFFFFF; font-size: 1.55rem; font-weight: 700;
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
        background: {GREEN_PALE}; border: 1px solid #CFE8D9;
        border-radius: 12px; padding: 1rem 1.2rem; height: 100%;
      }}
      .card h4 {{ margin: 0 0 0.2rem 0; color: {DARK}; font-size: 0.95rem; }}
      .card p  {{ margin: 0; color: {GREY}; font-size: 0.82rem; }}

      .probrow {{ display: flex; align-items: center; gap: 0.8rem; margin: 0.55rem 0; }}
      .probrow .lbl {{ width: 210px; font-size: 0.92rem; color: {DARK}; font-weight: 500; }}
      .probrow .track {{ flex: 1; background: #E3EFE8; border-radius: 999px; height: 13px; }}
      .probrow .fill  {{ background: {GREEN}; border-radius: 999px; height: 13px; }}
      .probrow .val   {{ width: 62px; text-align: right; font-weight: 700; color: {DARK}; }}

      .scorebox {{
        border: 1px solid #CFE8D9; border-radius: 10px; text-align: center;
        padding: 0.6rem 0.2rem; background: #FFFFFF;
      }}
      .scorebox .s {{ font-size: 1.35rem; font-weight: 700; color: {DARK}; }}
      .scorebox .p {{ font-size: 0.8rem; color: {GREEN}; font-weight: 600; }}

      .callout {{
        background: {DARK}; color: #FFFFFF; border-radius: 12px;
        padding: 0.9rem 1.2rem; margin-top: 0.6rem;
      }}
      .callout b {{ color: {GREEN_LIGHT}; }}

      div[data-testid="stMetricValue"] {{ color: {DARK}; }}
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
        return "hoch", GREEN
    if top >= 0.45:
        return "mittel", "#B7791F"
    return "niedrig", "#B44A3A"


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
    default_home = teams.index("Germany") if "Germany" in teams else 0
    default_away = teams.index("Ivory Coast") if "Ivory Coast" in teams else 1

    c1, c2, c3 = st.columns([5, 1, 5])
    with c1:
        home_team = st.selectbox("Heimmannschaft", teams, index=default_home,
                                 format_func=with_flag)
    with c2:
        st.markdown(
            "<div style='text-align:center;padding-top:2.1rem;"
            "font-weight:700;color:#6B7C72'>vs</div>",
            unsafe_allow_html=True,
        )
    with c3:
        away_team = st.selectbox("Auswärtsmannschaft", teams, index=default_away,
                                 format_func=with_flag)

    run = st.button("Vorhersage berechnen  ▶", type="primary", width="stretch")

    if home_team == away_team:
        st.warning("Bitte zwei verschiedene Mannschaften auswählen.")
    elif run:
        features = build_match_features(
            home_team, away_team, states, h2h,
            match_date=match_date,
            tournament_imp=tournament_imp,
            is_neutral=neutral,
        )
        X = features_to_frame(features, feature_order)
        probs = model.predict_proba(X)[0]

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
                yaxis=dict(range=[0, 0.8], title=None),
                legend=dict(orientation="h", y=1.15, x=0),
                plot_bgcolor="rgba(0,0,0,0)",
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
                colorscale=[[0, "#FFFFFF"], [1, GREEN]], showscale=False,
            ))
            fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Vorhergesagt", yaxis_title="Tatsächlich",
                yaxis=dict(autorange="reversed"),
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
                marker_color=GREEN,
                text=[f"{v:.3f}" for v in sub["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=max(360, 26 * len(sub)),
                margin=dict(l=10, r=40, t=20, b=10),
                xaxis_title="Wichtigkeit (relativ)", yaxis_title=None,
                plot_bgcolor="rgba(0,0,0,0)",
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
