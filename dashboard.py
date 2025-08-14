import streamlit as st
from news import get_news_for_ticker
from competitors import get_competitors
from growth import get_growth_stocks

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")
# 👉 Test d'affichage de la clé API
st.write("Clé API :", API_KEY)


st.set_page_config(page_title="📊 Stock Watcher", layout="wide")
st.title("📈 Stock Watcher Dashboard")


tickers_input = st.text_input("Entrez des tickers (séparés par des virgules)", "AAPL, TSLA")

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📰 Actualités")
        for t in tickers:
            st.markdown(f"**{t}**")
            news = get_news_for_ticker(t)
            for item in news:
                st.markdown(f"- [{item['title']}]({item['link']})")

    with col2:
        st.subheader("📊 Concurrents")
        for t in tickers:
            st.markdown(f"**{t}**")
            competitors = get_competitors(t)
            st.markdown(", ".join(competitors) if competitors else "Aucun concurrent trouvé.")

st.divider()
st.subheader("🚀 Actions à fort potentiel de croissance (3–12 mois)")
growth = get_growth_stocks()
st.table(growth)