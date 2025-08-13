import streamlit as st
import pandas as pd
from datetime import datetime
import os

# --- Configuration de la page ---
st.set_page_config(
    page_title="Évaluation de critères",
    page_icon="📊",
    layout="wide",  # plein écran
    initial_sidebar_state="collapsed"
)

# --- Styles CSS personnalisés ---
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Roboto, sans-serif;
        background-color: #f8f9fa;
        color: #212529;
        margin: 0;
        padding: 0;
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    .stSlider label {
        font-size: 1.1rem !important;
        font-weight: 500;
        color: #34495e;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 1.05rem;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    @media (max-width: 768px) {
        h1 {
            font-size: 1.6rem;
        }
        h2, h3 {
            font-size: 1.3rem;
        }
        .stSlider label {
            font-size: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Données ---
criteres = [
    "Impact énergie-climat",
    "Coût d’investissement (CAPEX)",
    "Aide-subvention",
    "Retour sur investissement (ROI)",
    "Temps de retour sur investissement",
    "Facilité de mise en œuvre",
    "Effet levier ou structurant",
    "Durée de vie de l’action",
    "Acceptabilité pour les utilisateurs",
    "Visibilité et exemplarité"
]

axes = {
    "Pertinence stratégique": "Niveau de contribution directe du critère aux objectifs et priorités de l'établissement",
    "Capacité discriminante": "Capacité du critère à distinguer clairement les différentes options",
    "Fiabilité de l’évaluation": "Niveau de fiabilité et facilité d'obtention des données",
    "Acceptabilité politique ou sociale": "Niveau d'acceptabilité et de soutien par les parties prenantes",
    "Temporalité / Durabilité": "Stabilité de l'importance du critère dans le temps"
}

# --- Initialisation ---
if "page" not in st.session_state:
    st.session_state.page = 0
if "reponses" not in st.session_state:
    st.session_state.reponses = []

def enregistrer_reponse(notes, critere):
    existing = next((r for r in st.session_state.reponses if r.get("Critère") == critere), None)
    if existing:
        existing.update(notes)
    else:
        notes["Critère"] = critere
        st.session_state.reponses.append(notes)

def sauvegarder_reponses():
    df = pd.DataFrame(st.session_state.reponses)
    df["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "evaluation_criteres.csv"
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", index=False, header=False)
    else:
        df.to_csv(file_path, index=False)

def afficher_resume():
    st.header("📝 Récapitulatif de vos évaluations")
    df = pd.DataFrame(st.session_state.reponses)
    if df.empty:
        st.write("Aucune réponse enregistrée.")
        return
    cols = ['Critère'] + [col for col in df.columns if col != 'Critère']
    df = df[cols]
    st.dataframe(df, use_container_width=True)

# --- Affichage ---
critere = criteres[st.session_state.page]
st.title("📊 Évaluation de critères d'aide à la décision")
st.subheader(f"Critère {st.session_state.page + 1} sur {len(criteres)} : {critere}")

notes = {}
for axe, definition in axes.items():
    st.markdown(f"**{axe}** — *{definition}*")
    note = st.slider(
        f"Note pour {axe}",
        1, 10, 5,
        key=f"slider-{st.session_state.page}-{axe}"
    )
    notes[axe] = note

st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.page > 0:
        st.button("⬅️ Précédent", on_click=lambda: setattr(st.session_state, 'page', st.session_state.page - 1))
with col2:
    if st.session_state.page < len(criteres) - 1:
        st.button("Suivant ➡️", on_click=lambda: (
            enregistrer_reponse(notes, critere),
            setattr(st.session_state, 'page', st.session_state.page + 1)
        ))
    else:
        if st.button("📄 Voir le récapitulatif"):
            enregistrer_reponse(notes, critere)
            st.session_state.show_resume = True

if 'show_resume' in st.session_state and st.session_state.show_resume:
    st.markdown("---")
    afficher_resume()
    if st.button("✅ Enregistrer toutes les réponses"):
        sauvegarder_reponses()
        st.success("✅ Évaluations enregistrées avec succès.")
        st.balloons()
        st.session_state.show_resume = False
