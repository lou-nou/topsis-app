import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Charger les données Excel
df = pd.read_excel("CAMECO.xlsx")
df.columns = df.columns.str.strip()

# Créer l'application Dash
app = dash.Dash(__name__)

# Interface utilisateur
app.layout = html.Div([
    html.H1("Tableau de bord multi-métriques", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Filtrer par catégorie :"),
        dcc.Dropdown(
            id='categorie-filter',
            options=[{'label': c, 'value': c} for c in df['Catégorie'].unique()],
            value=df['Catégorie'].unique()[0]
        ),
    ], style={'width': '40%', 'margin': 'auto', 'padding': '20px'}),

    # Grille 2x2 des graphiques
    html.Div([
        html.Div([
            dcc.Graph(id='graph-fermeture', style={'width': '48%', 'height': '400px'}),
            dcc.Graph(id='graph-volume', style={'width': '48%', 'height': '400px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        html.Div([
            dcc.Graph(id='graph-ouverture', style={'width': '48%', 'height': '400px'}),
            dcc.Graph(id='graph-boxplot', style={'width': '48%', 'height': '400px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),

        html.Div([
            dcc.Graph(id='graph-heatmap', style={'width': '100%', 'height': '500px'})
        ], style={'marginTop': '30px'})
    ], style={'maxWidth': '95%', 'margin': 'auto'})
])

# Callback pour mise à jour
@app.callback(
    Output('graph-fermeture', 'figure'),
    Output('graph-volume', 'figure'),
    Output('graph-ouverture', 'figure'),
    Output('graph-boxplot', 'figure'),
    Output('graph-heatmap', 'figure'),
    Input('categorie-filter', 'value')
)
def update_graphs(categorie):
    dff = df[df['Catégorie'] == categorie]

    fig1 = px.line(dff, x='Date', y='Fermeture', title='Cours de Fermeture')
    fig2 = px.bar(dff, x='Date', y='Volume', title='Volume échangé')
    fig3 = px.line(dff, x='Date', y='Ouverture', title='Cours d\'Ouverture')

    fig_box = px.box(
        dff,
        x='Catégorie',
        y='Fermeture',
        title='Distribution par Catégorie'
    )

    df_heat = df.copy()
    df_heat['Mois'] = pd.to_datetime(df_heat['Date'], errors='coerce').dt.to_period("M").astype(str)

    pivot = df_heat.pivot_table(
        index='Catégorie',
        columns='Mois',
        values='Fermeture',
        aggfunc='mean'
    )

    fig_heat = px.imshow(
        pivot,
        labels=dict(x="Mois", y="Catégorie", color="Fermeture Moy."),
        title="Carte thermique des moyennes de Fermeture"
    )

    return fig1, fig2, fig3, fig_box, fig_heat

# Lancement en local
if __name__ == '__main__':
    app.run(debug=True)
