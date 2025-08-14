
import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Analyse TOPSIS - Classement d'actions"),
    
    html.Div([
        html.Label("Noms des actions (séparés par des virgules)"),
        dcc.Input(id='actions', type='text', value='Action A, Action B, Action C', style={'width': '100%'}),
        
        html.Label("Critères (séparés par des virgules)"),
        dcc.Input(id='criteres', type='text', value='Rendement, Volatilité, P/E', style={'width': '100%'}),
        
        html.Label("Poids des critères (séparés par des virgules)"),
        dcc.Input(id='poids', type='text', value='0.4, 0.3, 0.3', style={'width': '100%'}),
        
        html.Label("Impacts des critères (max ou min, séparés par des virgules)"),
        dcc.Input(id='impacts', type='text', value='max, min, min', style={'width': '100%'}),
        
        html.Label("Valeurs (une ligne par action, valeurs séparées par des virgules)"),
        dcc.Textarea(id='valeurs', value='0.08, 0.15, 12\n0.06, 0.10, 18\n0.09, 0.20, 10', style={'width': '100%', 'height': 100}),
        
        html.Br(),
        html.Button("Lancer l'analyse TOPSIS", id='run-button', n_clicks=0),
    ], style={'width': '50%', 'margin': 'auto'}),
    
    html.Hr(),
    
    html.Div(id='resultats'),
    dcc.Graph(id='graphique')
])

def topsis(df, weights, impacts):
    norm_df = df / np.sqrt((df**2).sum())
    weighted_df = norm_df * weights
    ideal_best = weighted_df.max() if impacts[0] == 'max' else weighted_df.min()
    ideal_worst = weighted_df.min() if impacts[0] == 'max' else weighted_df.max()
    for i in range(1, len(impacts)):
        if impacts[i] == 'max':
            ideal_best[i] = weighted_df.iloc[:, i].max()
            ideal_worst[i] = weighted_df.iloc[:, i].min()
        else:
            ideal_best[i] = weighted_df.iloc[:, i].min()
            ideal_worst[i] = weighted_df.iloc[:, i].max()
    distance_best = np.sqrt(((weighted_df - ideal_best)**2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_df - ideal_worst)**2).sum(axis=1))
    score = distance_worst / (distance_best + distance_worst)
    return score

@app.callback(
    Output('resultats', 'children'),
    Output('graphique', 'figure'),
    Input('run-button', 'n_clicks'),
    State('actions', 'value'),
    State('criteres', 'value'),
    State('poids', 'value'),
    State('impacts', 'value'),
    State('valeurs', 'value')
)
def run_topsis(n_clicks, actions, criteres, poids, impacts, valeurs):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    actions = [a.strip() for a in actions.split(',')]
    criteres = [c.strip() for c in criteres.split(',')]
    poids = [float(p.strip()) for p in poids.split(',')]
    impacts = [i.strip() for i in impacts.split(',')]
    lignes = [list(map(float, l.strip().split(','))) for l in valeurs.strip().split('\n')]
    
    df = pd.DataFrame(lignes, columns=criteres, index=actions)
    scores = topsis(df, np.array(poids), impacts)
    df_result = pd.DataFrame({'Action': actions, 'Score TOPSIS': scores})
    df_result['Classement'] = df_result['Score TOPSIS'].rank(ascending=False).astype(int)
    df_result = df_result.sort_values(by='Score TOPSIS', ascending=False)

    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in df_result.columns],
        data=df_result.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'}
    )

    fig = px.bar(df_result, x='Action', y='Score TOPSIS', color='Classement', title='Scores TOPSIS des actions')

    return table, fig

if __name__ == '__main__':
    app.run(debug=True)
