import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px

# Lecture des données depuis le fichier Excel
excel_file = 'donnees_actions.xlsx'
df_data = pd.read_excel(excel_file, sheet_name='données', engine='openpyxl')
df_weights = pd.read_excel(excel_file, sheet_name='ponderation', engine='openpyxl')

# Préparation des données
actions = df_data.iloc[:, 0]
criteria_data = df_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
criteria_names = criteria_data.columns

# Création du vecteur de poids normalisé
weights_series = df_weights.set_index(df_weights.columns[0]).iloc[:, 0]
weights_series = weights_series.reindex(criteria_names).fillna(0)
weights_series = weights_series / weights_series.sum()

# Détermination des impacts
impacts = ['min' if ('coût' in crit.lower() or 'roi' in crit.lower() or
                     'facilité de mise en œuvre' in crit.lower() or
                     'acceptabilité-utilisateurs' in crit.lower()) else 'max'
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

# Calcul initial
scores = topsis(criteria_data, weights_series, impacts)
df_result = pd.DataFrame({
    'Action': actions,
    'Score TOPSIS': scores
})
df_result['Classement'] = df_result['Score TOPSIS'].rank(ascending=False).astype(int)
df_result = df_result.sort_values(by='Score TOPSIS', ascending=False)

# Analyse de sensibilité
sensitivity_results = []
excluded_criterion = "Économie d’énergie (%)"

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

# Application Dash
app = dash.Dash(__name__)
app.title = "Analyse TOPSIS avec Sensibilité"

app.layout = html.Div([
html.H3("Données d'origine"),
dash_table.DataTable(
    id='data-table',
    columns=[{"name": col, "id": col} for col in df_data.columns],
    data=df_data.to_dict('records'),
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left'},
),

html.H3("Poids des critères"),
dash_table.DataTable(
    id='weights-table',
    columns=[{"name": "Critère", "id": "Critère"}, {"name": "Poids", "id": "Poids"}],
    data=[{"Critère": crit, "Poids": poids} for crit, poids in weights_series.items()],
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left'},
),

html.H3("Direction des critères (Impacts)"),
dash_table.DataTable(
    id='impacts-table',
    columns=[{"name": "Critère", "id": "Critère"}, {"name": "Impact", "id": "Impact"}],
    data=[{"Critère": crit, "Impact": imp} for crit, imp in zip(criteria_names, impacts)],
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left'},
),

    html.H2("Analyse TOPSIS des actions"),
    dash_table.DataTable(
        id='topsis-table',
        columns=[{"name": i, "id": i} for i in df_result.columns],
        data=df_result.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        sort_action='native'
    ),
    dcc.Graph(
        id='bar-graph',
        figure=px.bar(df_result, x='Action', y='Score TOPSIS', color='Classement',
                      title="Scores TOPSIS par action", text='Classement')
    ),
    html.H2("Analyse de sensibilité des scores TOPSIS"),
    dcc.Graph(
        id='sensitivity-graph',
        figure=px.line(df_sensitivity, x='Action', y='Score TOPSIS', color='Critère modifié',
                       line_dash='Variation', markers=True,
                       title="Variation des scores TOPSIS selon les poids des critères")
        .update_layout(
            yaxis=dict(range=[0, 1]),
            height=700
        )
    ),
    html.H3("Filtrer les critères affichés dans la sensibilité"),
    dcc.Dropdown(
        id='critere-dropdown',
        options=[{'label': crit, 'value': crit} for crit in df_sensitivity['Critère modifié'].unique()],
        value=list(df_sensitivity['Critère modifié'].unique()),
        multi=True
    ),
    html.Button("Exporter les résultats", id="export-button", n_clicks=0),
    html.Div(id="export-message")
])

@app.callback(
    Output('sensitivity-graph', 'figure'),
    Input('critere-dropdown', 'value')
)
def update_sensitivity_graph(selected_criteria):
    filtered_df = df_sensitivity[df_sensitivity['Critère modifié'].isin(selected_criteria)]
    fig = px.line(filtered_df, x='Action', y='Score TOPSIS', color='Critère modifié',
                  line_dash='Variation', markers=True,
                  title="Variation des scores TOPSIS selon les poids des critères")
    fig.update_layout(yaxis=dict(range=[0, 1]), height=700)
    return fig

@app.callback(
    Output("export-message", "children"),
    Input("export-button", "n_clicks")
)
def export_results(n_clicks):
    if n_clicks > 0:
        df_result.to_excel("resultats_topsis.xlsx", index=False)
        df_sensitivity.to_excel("sensibilite_topsis.xlsx", index=False)
        return "Les fichiers 'resultats_topsis.xlsx' et 'sensibilite_topsis.xlsx' ont été exportés."
    return ""

if __name__ == '__main__':
    app.run(debug=True)

