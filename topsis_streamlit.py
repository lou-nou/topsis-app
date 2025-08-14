import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image

# --- Affichage du logo ---
try:
    logo = Image.open("logo.png")
    st.image(logo, width=120)
except FileNotFoundError:
    pass

st.title("🔍 Analyse multicritère TOPSIS avec sensibilité")

# --- Upload du fichier Excel ---
uploaded_file = st.file_uploader("📤 Charger un fichier Excel (.xlsx) contenant les feuilles 'données' et 'ponderation'", type=["xlsx"])

if uploaded_file is None:
    st.info("Veuillez importer un fichier pour commencer.")
    st.stop()

# --- Lecture des données ---
try:
    df_data = pd.read_excel(uploaded_file, sheet_name='données', engine='openpyxl')
    df_weights = pd.read_excel(uploaded_file, sheet_name='ponderation', engine='openpyxl')
except Exception as e:
    st.error(f"Erreur de lecture du fichier : {e}")
    st.stop()

# --- Préparation des données ---
actions = df_data.iloc[:, 0]
criteria_data = df_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
criteria_names = criteria_data.columns

# --- Normalisation des poids ---
weights_series = df_weights.set_index(df_weights.columns[0]).iloc[:, 0]
weights_series = weights_series.reindex(criteria_names).fillna(0)
weights_series = weights_series / weights_series.sum()

# --- Détermination des impacts ---
impacts = ['min' if ('coût' in crit.lower() or 'roi' in crit.lower() or
                     'facilité' in crit.lower() or 'acceptabilité' in crit.lower()) else 'max'
           for crit in criteria_names]

# --- Fonction TOPSIS ---
def topsis(criteria_data, weights, impacts):
    norm_matrix = criteria_data / np.sqrt((criteria_data ** 2).sum())
    weighted_matrix = norm_matrix * weights

    ideal_best = weighted_matrix.max()
    ideal_worst = weighted_matrix.min()

    for i, impact in enumerate(impacts):
        if impact == 'min':
            ideal_best.iloc[i], ideal_worst.iloc[i] = ideal_worst.iloc[i], ideal_best.iloc[i]

    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    scores = dist_worst / (dist_best + dist_worst)
    return scores

# --- Calcul TOPSIS ---
scores = topsis(criteria_data, weights_series, impacts)
df_result = pd.DataFrame({
    'Action': actions,
    'Score TOPSIS': scores
})
df_result['Classement'] = df_result['Score TOPSIS'].rank(ascending=False).astype(int)
df_result = df_result.sort_values(by='Score TOPSIS', ascending=False)

# --- Analyse de sensibilité ---
excluded_criterion = "Économie d’énergie (%)"  # critère exclu par défaut (modifiable)
sensitivity_results = []

for i, crit in enumerate(criteria_names):
    if crit != excluded_criterion:
        for variation in [-0.1, 0.1]:
            new_weights = weights_series.copy()
            new_weights.iloc[i] *= (1 + variation)
            new_weights /= new_weights.sum()
            new_scores = topsis(criteria_data, new_weights, impacts)
            for action, score in zip(actions, new_scores):
                sensitivity_results.append({
                    'Critère modifié': crit,
                    'Variation': f"{int(variation * 100)}%",
                    'Action': action,
                    'Score TOPSIS': score
                })

df_sensitivity = pd.DataFrame(sensitivity_results)

# --- Résultats principaux ---
st.subheader("📊 Résultats de l’analyse TOPSIS")
st.dataframe(df_result, use_container_width=True)

fig_bar = px.bar(df_result, x='Action', y='Score TOPSIS', color='Classement', text='Classement',
                 title="Scores TOPSIS par action")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Résultats de sensibilité ---
st.subheader("📉 Analyse de sensibilité")
selected_criteria = st.multiselect(
    "Filtrer les critères affichés dans l’analyse de sensibilité",
    options=df_sensitivity['Critère modifié'].unique(),
    default=list(df_sensitivity['Critère modifié'].unique())
)

filtered_df = df_sensitivity[df_sensitivity['Critère modifié'].isin(selected_criteria)]

fig_sens = px.line(
    filtered_df, x='Action', y='Score TOPSIS',
    color='Critère modifié', line_dash='Variation', markers=True,
    title="Variation des scores TOPSIS selon les poids des critères"
)
fig_sens.update_layout(yaxis=dict(range=[0, 1]), height=700)
st.plotly_chart(fig_sens, use_container_width=True)

# --- Téléchargement des résultats ---
st.subheader("📥 Télécharger les résultats")

output_buffer = BytesIO()
with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
    df_result.to_excel(writer, index=False, sheet_name='Résultats TOPSIS')
    df_sensitivity.to_excel(writer, index=False, sheet_name='Sensibilité')

st.download_button(
    label="Télécharger les résultats au format Excel",
    data=output_buffer.getvalue(),
    file_name="resultats_topsis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

