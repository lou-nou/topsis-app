import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="TOPSIS Flexible", layout="wide")
st.title("⚙️ Outil d'évaluation TOPSIS - Critères & Actions personnalisables")

# Étape 1 : Saisie des critères
st.subheader("1. Liste des critères")
crit_input = st.text_area(
    "Entrez un critère par ligne (ex: Coût, ROI, Acceptabilité...)",
    "Coût (€)\nROI attendu\nFacilité de mise en œuvre\nAcceptabilité-utilisateurs\nÉconomie d’énergie (%)"
)
criteria_names = [c.strip() for c in crit_input.split('\n') if c.strip()]

# Étape 2 : Saisie des actions
st.subheader("2. Liste des actions à comparer")
actions_input = st.text_area(
    "Entrez un nom d'action par ligne",
    "Action A\nAction B\nAction C"
)
action_names = [a.strip() for a in actions_input.split('\n') if a.strip()]

# Étape 3 : Table de saisie des données
st.subheader("3. Remplissez la matrice des données (valeurs pour chaque critère)")
default_matrix = pd.DataFrame(
    0.0,
    index=action_names,
    columns=criteria_names
)
df_data = st.data_editor(
    default_matrix,
    key="criteria_input_matrix",
    use_container_width=True,
    num_rows="fixed"
)

# Étape 4 : Table pour les poids
st.subheader("4. Définir les poids (entre 0 et 100 pour chaque critère)")
default_weights = pd.DataFrame({
    "Critère": criteria_names,
    "Poids": [20.0] * len(criteria_names)
})
df_weights = st.data_editor(
    default_weights,
    key="weights_input_table",
    num_rows="fixed"
)

# Étape 5 : Bouton d'exécution
if st.button("⚙️ Exécuter l'analyse TOPSIS"):

    # Normaliser les poids
    weights_series = df_weights.set_index("Critère")["Poids"]
    weights_series = weights_series.reindex(criteria_names).fillna(0)
    weights_series = weights_series / weights_series.sum()

    # Déterminer automatiquement les impacts
    impacts = ['min' if any(x in crit.lower() for x in ['coût', 'roi', 'facilité', 'acceptabilité']) else 'max'
               for crit in criteria_names]

    # Fonction TOPSIS
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

    # Appliquer TOPSIS
    scores = topsis(df_data, weights_series, impacts)
    df_result = pd.DataFrame({
        "Action": df_data.index,
        "Score TOPSIS": scores
    })
    df_result["Classement"] = df_result["Score TOPSIS"].rank(ascending=False).astype(int)
    df_result = df_result.sort_values("Score TOPSIS", ascending=False)

    # Affichage des résultats
    st.success("Analyse TOPSIS effectuée ✅")
    st.dataframe(df_result, use_container_width=True)

    st.plotly_chart(
        px.bar(df_result, x="Action", y="Score TOPSIS", color="Classement", text="Classement",
               title="Classement TOPSIS des actions"),
        use_container_width=True
    )

    # Export Excel
    with st.expander("📤 Exporter les résultats TOPSIS"):
        if st.button("Télécharger les résultats Excel"):
            df_result.to_excel("resultats_topsis.xlsx", index=False)
            st.write("✅ Fichier `resultats_topsis.xlsx` généré.")

