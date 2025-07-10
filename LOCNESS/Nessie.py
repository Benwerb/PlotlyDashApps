import dash_bootstrap_components as dbc
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re
from data_loader import GliderDataLoader, GulfStreamLoader
import datetime as dt

def get_first_10_pH_average(df_latest):
    df_MLD_average = df_latest.drop_duplicates(subset=['Station'], keep='first').copy()

    # Initialize new columns
    df_MLD_average['pHinsitu[Total]'] = np.nan
    df_MLD_average['Chl_a[mg/m^3]'] = np.nan

    for station in df_latest['Station'].unique():
        station_data = df_latest[df_latest['Station'] == station]
        first_10 = station_data.head(5)
        
        if 'pHinsitu[Total]' in first_10.columns:
            avg_pH = first_10['pHinsitu[Total]'].mean()
            avg_chl = first_10['Chl_a[mg/m^3]'].mean()
            
            # Add directly to DataFrame
            mask = df_MLD_average['Station'] == station
            df_MLD_average.loc[mask, 'pHinsitu[Total]'] = avg_pH
            df_MLD_average.loc[mask, 'Chl_a[mg/m^3]'] = avg_chl

    # Keep only the columns you want
    df_MLD_average = df_MLD_average[['Station', 'Lat [°N]', 'Lon [°E]', 'pHinsitu[Total]', 'Chl_a[mg/m^3]']]

    return df_MLD_average

gs = GulfStreamLoader()
GulfStreamBounds = gs.load_data()

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY ]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn


loader = GliderDataLoader()
# Load the most recent file (automatically done if no filename provided)
df_latest = loader.load_data()
station_min, station_max = df_latest["Station"].min(), df_latest["Station"].max()
date_min, date_max = df_latest["Date"].min(), df_latest["Date"].max() 

app.layout = dbc.Container([
    # Top row - Map
    dbc.Row([
        
        html.H2('NESSIE', className='text-info text-start',
                        style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '20px'}),
        dbc.Col([
            html.Div([
                # Map plot
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='map-plot',
                            style={'height': '70vh', 'width': '100%'}
                        ),
                        dcc.Interval(id="interval-refresh", interval=60*1000, n_intervals=0)  # every 60 sec
                    ])
                ]),
                # Range slider
                html.Div([
                    dcc.RangeSlider(
                        id='RangeSlider',
                        updatemode='mouseup',  # don't let it update till mouse released
                        min=station_min,
                        max=station_max,
                        # value=[unix_time_millis(start_date), unix_time_millis(end_date)],
                        value=[station_min, station_max],
                        # marks=get_marks_every_6_hours(start_date, end_date),
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'padding': '20px 10px', 'margin-top': '10px'})
            ], style={'padding': '10px', 'backgroundColor': '#e3f2fd'})
        ], width=12)
    ]),
    
    # Bottom row - Settings and Parameters side by side
    dbc.Row([
        # Left column - Settings
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Assets:"),
                        dcc.Checklist(
                                id='SN209',
                                options=[{'label': 'SN209', 'value': 'overlay'}],
                                value=['overlay'],
                                labelStyle={'display': 'block'}
                            ),
                        dcc.Checklist(
                                id='SN210',
                                options=[{'label': 'SN210', 'value': 'overlay'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            ),
                        dcc.Checklist(
                                id='SN070',
                                options=[{'label': 'SN070', 'value': 'overlay'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            ),
                        dcc.Checklist(
                                id='SN0075',
                                options=[{'label': 'SN0075', 'value': 'overlay'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            ),
                        dcc.Checklist(
                                id='LRAUV',
                                options=[{'label': 'LRAUV', 'value': 'overlay'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            ),
                        dcc.Checklist(
                                id='Ship',
                                options=[{'label': 'Ship', 'value': 'overlay'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=4),
        
        # Right column - Parameters
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Parameters"),
                        dcc.RadioItems(
                            id='Parameters',
                            options=[
                                {'label': 'Station', 'value': 'Station'},
                                {'label': 'pHin MLD mean', 'value': 'pHinsitu[Total]'},
                                {'label': 'Rodamine MLD mean', 'value': ''},
                                {'label': 'MLD [m]', 'value': ''},
                            ],
                            value='Station'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=4),
        
        # Right column - Parameters
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Map Options"),
                        dcc.Checklist(
                                id='gsbounds',
                                options=[{'label': 'Overlay Gulf Stream', 'value': 'overlay'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=4)
    ]),

   # Third Row - Plotting
    dbc.Row([
        # First column (left side)
        dbc.Col([
            html.Div([
                # pHin
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='pHin-plot',
                            style={'height': '70vh', 'width': '100%'}
                        ),
                    ])
                ]),
            ], style={'padding': '10px', 'backgroundColor': '#e3f2fd'})
        ], width=6),
        
        # Second column (right side)
        dbc.Col([
            html.Div([
                # Second plot or content
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='doxy-plot',  # Give it a unique ID
                            style={'height': '70vh', 'width': '100%'}
                        ),
                    ])
                ]),
            ], style={'padding': '10px', 'backgroundColor': '#e3f2fd'})
        ], width=6),
        
    ]),
], fluid=True, className='dashboard-container')

@app.callback(
    Output('map-plot','figure'),
    [Input("interval-refresh", "n_intervals"),
     Input('Parameters', 'value'),
     Input('gsbounds','value')]
)
def update_map(n, selected_parameter, gs_overlay):
    df_latest = loader.load_data()
    df_map = get_first_10_pH_average(df_latest)

    if isinstance(loader.file_list, list):
        # Create base figure
        map_fig = px.scatter_map(
            df_map, lat="Lat [°N]", lon="Lon [°E]",
            map_style="satellite",
            zoom=8,
            color=selected_parameter,
            hover_name=selected_parameter
            # labels={"Station": "Profile"}
            
        )
    map_fig.update_layout(legend_title='Spray209')
    
    # If Gulf Stream overlay is checked, add trace
    if 'overlay' in gs_overlay:
        map_fig.add_trace(go.Scattermap(
            lat=GulfStreamBounds['Lat'],
            lon=GulfStreamBounds['Lon'],
            mode='lines',
            name='Gulf Stream',
            marker=dict(size=6, color='deepskyblue'),
            line=dict(width=2, color='deepskyblue')
        ))
    return map_fig

@app.callback(
    Output('pHin-plot','figure'),
    [Input("interval-refresh", "n_intervals")]
)
def update_pHin_plot(n):
    df_latest = loader.load_data()

    scatter_fig_pHin = px.scatter(
        df_latest, x="pHinsitu[Total]", y="Depth[m]",
        labels={"pHinsitu[Total]", "Depth[m]", "Profile"},
        title=f"pHinsitu[Total] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_pHin.update_yaxes(autorange="reversed")
    
    return scatter_fig_pHin

@app.callback(
    Output('doxy-plot','figure'),
    [Input("interval-refresh", "n_intervals")]
)
def update_doxy_plot(n):
    df_latest = loader.load_data()

    scatter_fig_doxy = px.scatter(
        df_latest, x="Oxygen[µmol/kg]", y="Depth[m]",
        labels={"Oxygen[µmol/kg]", "Depth[m]", "Profile"},
        title=f"Oxygen[µmol/kg] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_doxy.update_yaxes(autorange="reversed")
    
    return scatter_fig_doxy

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port, debug=True)