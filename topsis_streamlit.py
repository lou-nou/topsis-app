import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image

# =========================================================
# Helpers
# =========================================================
def clean_numeric(x):
    """
    Convertit '170 946 €', '2,1 %', '45,5' -> float.
    Laisse passer les floats intacts. Retourne np.nan si vide.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("\xa0", "")
    for token in ["€", "%"]:
        s = s.replace(token, "")
    s = s.replace(" ", "").replace(",", ".")
    return float(s) if s != "" else np.nan


def topsis(criteria_data: pd.DataFrame, weights: pd.Series, impacts: list[str], eps: float = 1e-12):
    """
    TOPSIS standard (version Excel-friendly) :
      1) Normalisation vectorielle par colonne
      2) Pondération par poids
      3) Idéaux +/- (inversion si impact 'min')
      4) Distances euclidiennes
      5) Score = d- / (d+ + d-)
    """
    # 1) Normalisation (colonne par colonne)
    denom = np.sqrt((criteria_data ** 2).sum(axis=0))
    denom = denom.replace(0, eps)     # sécurité si colonne nulle
    norm = criteria_data / denom

    # 2) Pondération (alignement par noms de colonnes)
    w = weights.reindex(criteria_data.columns).fillna(0.0)
    w = w / max(w.sum(), eps)         # renormalise si besoin
    V = norm * w

    # 3) Idéal + / -
    ideal_best = V.max(axis=0).copy()
    ideal_worst = V.min(axis=0).copy()

    # 4) Inversion pour les critères 'min' (à minimiser)
    for j, impact in enumerate(impacts):
        if impact == "min":
            ideal_best.iloc[j], ideal_worst.iloc[j] = ideal_worst.iloc[j], ideal_best.iloc[j]

    # 5) Distances et score
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    scores = d_minus / (d_plus + d_minus + eps)
    return scores


# =========================================================
# UI : logo + titre
# =========================================================
try:
    logo = Image.open("logo.png")
    st.image(logo, width=120)
except FileNotFoundError:
    pass

st.title("🔍 Analyse multicritère TOPSIS avec sensibilité")

# =========================================================
# Upload
# =========================================================
uploaded_file = st.file_uploader(
    "📤 Charger un fichier Excel (.xlsx) contenant les feuilles 'données' et 'ponderation'",
    type=["xlsx"]
)
if uploaded_file is None:
    st.info("Veuillez importer un fichier pour commencer.")
    st.stop()

# =========================================================
# Lecture Excel
# =========================================================
try:
    df_data = pd.read_excel(uploaded_file, sheet_name="données", engine="openpyxl")
    df_weights = pd.read_excel(uploaded_file, sheet_name="ponderation", engine="openpyxl")
except Exception as e:
    st.error(f"Erreur de lecture du fichier : {e}")
    st.stop()

# =========================================================
# Préparation des données (feuille 'données')
#  - Colonne 1 : Action (nom de l'alternative)
#  - Colonnes 2..n : critères (numériques)
# =========================================================
if df_data.shape[1] < 2:
    st.error("La feuille 'données' doit contenir au moins 2 colonnes (Action + ≥1 critère).")
    st.stop()

actions = df_data.iloc[:, 0]
criteria_data = df_data.iloc[:, 1:].copy()
criteria_names = criteria_data.columns.tolist()

# Nettoyage éventuel des colonnes critères (€, %, virgules…) -> numérique
for c in criteria_data.columns:
    if not pd.api.types.is_numeric_dtype(criteria_data[c]):
        criteria_data[c] = criteria_data[c].map(clean_numeric)
criteria_data = criteria_data.astype(float)

# Vérif colonnes entièrement nulles
if ((criteria_data.fillna(0) == 0).all()).any():
    st.warning("Attention : au moins un critère est entièrement nul — vérifiez vos données.")

# =========================================================
# Lecture simple des POIDS & DIRECTIONS (1=critère, 2=poids, 3=direction)
#  - On ignore les noms de colonnes pour éviter toute ambiguïté.
#  - Directions acceptées (insensibles à la casse/espaces): 'maximiser' ou 'minimiser'
# =========================================================
if df_weights.shape[1] < 3:
    st.error("La feuille 'ponderation' doit contenir au moins 3 colonnes (Critère, Poids, Direction).")
    st.stop()

crit_col = df_weights.columns[0]   # noms des critères
w_col    = df_weights.columns[1]   # poids
d_col    = df_weights.columns[2]   # direction

# Poids : alignés sur les colonnes de 'données'
weights_series = (
    df_weights.set_index(crit_col)[w_col]
    .map(clean_numeric)
    .reindex(criteria_names)            # même ordre que 'données'
    .fillna(0.0)
)
# Renormalisation (somme = 1)
sum_w = weights_series.sum()
if sum_w <= 0:
    st.error("La somme des poids est nulle. Vérifiez la feuille 'ponderation' (colonne 2).")
    st.stop()
weights_series = weights_series / sum_w

# Directions : 'maximiser' / 'minimiser' -> 'max' / 'min'
dir_raw = (
    df_weights.set_index(crit_col)[d_col]
    .astype(str).str.replace("\xa0"," ", regex=False).str.strip().str.lower()
    .reindex(criteria_names)
)

dir_map = {"maximiser": "max", "minimiser": "min"}
impacts_series = dir_raw.map(dir_map)

# Contrôle qualité : direction non reconnue
if impacts_series.isna().any():
    st.error("Certaines directions ne sont pas reconnues. Utilisez EXACTEMENT 'maximiser' ou 'minimiser' dans la 3e colonne.")
    st.dataframe(pd.DataFrame({
        "Critère": criteria_names,
        "Direction lue": dir_raw
    })[impacts_series.isna()])
    st.stop()

impacts = impacts_series.tolist()

# =========================================================
# Calcul TOPSIS principal
# =========================================================
scores = topsis(criteria_data, weights_series, impacts)
df_result = pd.DataFrame({"Action": actions, "Score TOPSIS": scores})
df_result["Classement"] = df_result["Score TOPSIS"].rank(ascending=False, method="min").astype(int)
df_result = df_result.sort_values(by="Score TOPSIS", ascending=False).reset_index(drop=True)

# =========================================================
# Affichage résultats
# =========================================================
st.subheader("📊 Résultats de l’analyse TOPSIS")
st.dataframe(df_result, use_container_width=True)

fig_bar = px.bar(
    df_result, x="Action", y="Score TOPSIS",
    color="Classement", text="Classement",
    title="Scores TOPSIS par action"
)
st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# Analyse de sensibilité (variation ±10% d’un poids, renormalisation)
# =========================================================
st.subheader("📉 Analyse de sensibilité")
excluded_criterion = st.selectbox(
    "Critère à exclure des variations (optionnel)",
    options=["(aucun)"] + criteria_names,
    index=0
)

sensitivity_results = []
for i, crit in enumerate(criteria_names):
    if excluded_criterion != "(aucun)" and crit == excluded_criterion:
        continue
    for variation in (-0.10, +0.10):  # ±10 %
        new_w = weights_series.copy()
        new_w.iloc[i] = new_w.iloc[i] * (1 + variation)
        new_w = new_w / max(new_w.sum(), 1e-12)
        new_scores = topsis(criteria_data, new_w, impacts)
        for action, s in zip(actions, new_scores):
            sensitivity_results.append({
                "Critère modifié": crit,
                "Variation": f"{int(variation*100)}%",
                "Action": action,
                "Score TOPSIS": s
            })

df_sensitivity = pd.DataFrame(sensitivity_results)

if not df_sensitivity.empty:
    selected_criteria = st.multiselect(
        "Filtrer les critères affichés dans l’analyse de sensibilité",
        options=df_sensitivity["Critère modifié"].unique(),
        default=list(df_sensitivity["Critère modifié"].unique())
    )
    filtered_df = df_sensitivity[df_sensitivity["Critère modifié"].isin(selected_criteria)]
    fig_sens = px.line(
        filtered_df, x="Action", y="Score TOPSIS",
        color="Critère modifié", line_dash="Variation", markers=True,
        title="Variation des scores TOPSIS selon les poids des critères"
    )
    fig_sens.update_layout(yaxis=dict(range=[0, 1]), height=700)
    st.plotly_chart(fig_sens, use_container_width=True)

# =========================================================
# Export Excel
# =========================================================
st.subheader("📥 Télécharger les résultats")
output_buffer = BytesIO()
with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
    df_result.to_excel(writer, index=False, sheet_name="Résultats TOPSIS")
    if not df_sensitivity.empty:
        df_sensitivity.to_excel(writer, index=False, sheet_name="Sensibilité")

st.download_button(
    label="Télécharger les résultats au format Excel",
    data=output_buffer.getvalue(),
    file_name="resultats_topsis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
