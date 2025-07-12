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
    df_MLD_average = df_latest.drop_duplicates(subset=['Station', 'Cruise'], keep='first').copy()

    # Initialize new columns
    df_MLD_average['pHinsitu[Total]'] = np.nan
    df_MLD_average['Chl_a[mg/m^3]'] = np.nan

    # Group by both Station and Cruise
    for (station, cruise), group in df_latest.groupby(['Station', 'Cruise']):
        first_10 = group.head(5)
        
        if 'pHinsitu[Total]' in first_10.columns:
            avg_pH = first_10['pHinsitu[Total]'].mean()
            avg_chl = first_10['Chl_a[mg/m^3]'].mean()
            
            # Add directly to DataFrame using both Station and Cruise
            mask = (df_MLD_average['Station'] == station) & (df_MLD_average['Cruise'] == cruise)
            df_MLD_average.loc[mask, 'pHinsitu[Total]'] = avg_pH
            df_MLD_average.loc[mask, 'Chl_a[mg/m^3]'] = avg_chl

    # Keep only the columns you want
    df_MLD_average = df_MLD_average[['Station', 'Cruise', 'Lat [°N]', 'Lon [°E]', 'pHinsitu[Total]', 'Chl_a[mg/m^3]']]

    return df_MLD_average
# Define variable-specific percentile limits
def get_clim(df, color_column):
    if color_column == 'ChlorophyllA':
        lower, upper = np.percentile(df[color_column].dropna(), [5, 99])
    else:
        lower, upper = np.percentile(df[color_column].dropna(), [1, 99])

    step = max(round((upper - lower) / 100, 3), 0.001)

    return lower, upper, step

def filter_glider_assets(df, glider_overlay):
    """ Fliter gliderviz files based on selected gliders from assets menu. """
    if glider_overlay:
            # Extract numeric parts from the selected cruise strings (e.g., 'SN203' -> '203')
            selected_cruises = [s.replace('SN', '') for s in glider_overlay if 'SN' in s]
            # Apply filter for any of the selected cruises
            df_filtered = df[df["Cruise"].astype(str).str.contains('|'.join(selected_cruises))]
    else:
        df_filtered = df.iloc[0:0] 

    return df_filtered

def apply_common_scatter_styling(fig, colorbar_title="Station", colorbar_orientation='v'):
    """
    Apply common layout styling to a Plotly scatter figure.

    Parameters:
    ----------
    fig : plotly.graph_objects.Figure or plotly.express.Figure
        The figure to style.
    colorbar_title : str
        Title for the colorbar.
    colorbar_orientation : str
        'v' for vertical (default), 'h' for horizontal.
    """
    fig.update_yaxes(autorange="reversed")
    
    colorbar_config = dict(
        title=dict(text=colorbar_title),
        orientation=colorbar_orientation,
        len=0.8,
        thickness=25,
    )
    
    if colorbar_orientation == 'v':
        colorbar_config.update(x=1.02, y=0.5)
    else:  # horizontal
        colorbar_config.update(x=0.5, y=-0.2, xanchor='center', yanchor='top')

    fig.update_layout(coloraxis_colorbar=colorbar_config)
    return fig

def make_depth_scatter_plot(
    df, x, y="Depth[m]", title=None,
    color="Station", symbol="Cruise",
    labels=None,
    colorbar_title=None,
    colorbar_orientation='v'
):
    """
    Create a reusable scatter plot of a variable vs. depth with consistent styling.

    Parameters:
    ----------
    df : DataFrame
        Input data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis (default: "Depth[m]").
    title : str or None
        Plot title. If None, a default is generated.
    color : str
        Column to use for color.
    symbol : str
        Column to use for symbol.
    labels : dict or None
        Custom axis labels.
    colorbar_title : str or None
        Title for the colorbar. Defaults to the `color` column name.
    colorbar_orientation : str
        'v' or 'h' for vertical or horizontal colorbar.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        Styled scatter plot.
    """
    if title is None:
        title = f"{x} vs. {y}"
    if labels is None:
        labels = {x: x, y: y, color: color, symbol: symbol}
    if colorbar_title is None:
        colorbar_title = color

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        labels=labels,
        title=title
    )

    # Reverse y-axis (for depth plots)
    fig.update_yaxes(autorange="reversed")

    # Add colorbar settings
    colorbar_config = dict(
        title=dict(text=colorbar_title),
        orientation=colorbar_orientation,
        len=0.8,
        thickness=25
    )

    if colorbar_orientation == 'v':
        colorbar_config.update(x=1.02, y=0.5)
    else:
        colorbar_config.update(x=0.5, y=-0.2, xanchor='center', yanchor='top')

    fig.update_layout(coloraxis_colorbar=colorbar_config)

    return fig

def range_slider_marks(df, target_mark_count=10):
    """
    Generate RangeSlider marks at evenly spaced full-hour intervals,
    aligned to the nearest hour, based on a target number of marks.

    Parameters:
    ----------
    df : pandas.DataFrame
        Must contain 'Datetime' and 'UnixTimestamp' columns.
    target_mark_count : int
        Approximate number of marks to generate.

    Returns:
    -------
    dict
        Dictionary of {UnixTimestamp: formatted datetime string}
    """
    # Sort and get min/max
    df = df.sort_values("Datetime")
    t_min = df['Datetime'].min()
    t_max = df['Datetime'].max()

    if pd.isna(t_min) or pd.isna(t_max) or t_max <= t_min:
        return {}

    # Round t_min up to next full hour
    t_start = (t_min + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    # Total range in seconds
    total_seconds = (t_max - t_start).total_seconds()
    if total_seconds <= 0:
        return {}

    # Compute spacing interval (rounded to nearest hour step)
    interval_seconds = total_seconds // target_mark_count
    interval_hours = max(1, int(round(interval_seconds / 3600)))

    # Generate evenly spaced timestamps
    timestamps = pd.date_range(start=t_start, end=t_max, freq=f'{interval_hours}h')

    # Convert to Unix timestamp and format labels
    marks = {int(ts.timestamp()): ts.strftime('%m/%d %H:%M') for ts in timestamps}
    return marks

gs = GulfStreamLoader()
GulfStreamBounds = gs.load_data()
glider_ids = ['SN203', 'SN209', 'SN070', 'SN0075']
# glider_ids = ['203', '209', '070', '075']

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY ]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn


loader = GliderDataLoader(filenames=['25420901RT.txt', '25520301RT.txt'])
# Load the most recent file (automatically done if no filename provided)
df_latest = loader.load_data()
station_min, station_max = df_latest["Station"].min(), df_latest["Station"].max()
date_min, date_max = df_latest["Date"].min(), df_latest["Date"].max() 
unix_min, unix_max = df_latest["UnixTimestamp"].min(), df_latest["UnixTimestamp"].max() 
unix_max_minus_12hrs = unix_max - 60*60*12
marks = range_slider_marks(df_latest, 20)

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
                        dcc.Interval(id="interval-refresh", interval=60*1000*5, n_intervals=0)  # every 5 minutes
                    ])
                ]),
                # Range slider
                html.Div([
                    dcc.RangeSlider(
                        id='RangeSlider',
                        updatemode='mouseup',  # don't let it update till mouse released
                        min=unix_min,
                        max=unix_max,
                        step=3600, # 1 hour
                        value=[unix_max_minus_12hrs, unix_max],
                        marks=marks,
                        # tooltip={"placement": "bottom", "always_visible": False}
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
                            id='glider_overlay_checklist',
                            options=[{'label': s, 'value': s} for s in glider_ids],
                            value=['SN203'],
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
                                {'label': 'Rodamine MLD mean', 'value': 'Ro'},
                                {'label': 'MLD [m]', 'value': 'Mld'},
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

# @app.callback(
#     [Output('RangeSlider', 'min'),
#      Output('RangeSlider', 'max'),
#      Output('RangeSlider', 'value'),
#      Output('RangeSlider', 'marks')],
#       [Input('glider_overlay_checklist', 'value'),
#       Input('interval-refresh', 'n_intervals')]
# )
# def update_range_slider(glider_overlay, n):
#     df_latest = loader.load_data()
#     unix_min, unix_max = df_latest["UnixTimestamp"].min(), df_latest["UnixTimestamp"].max() 
#     unix_max_minus_12hrs = unix_max - 60*60*12
#     marks = range_slider_marks(df_latest, 20)
#     return unix_min, unix_max, [unix_max_minus_12hrs, unix_max], marks

@app.callback(
    Output('map-plot','figure'),
    [Input('interval-refresh', 'n_intervals'),
     Input('Parameters', 'value'),
     Input('gsbounds','value'),
     Input('glider_overlay_checklist', 'value')]
)
def update_map(n, selected_parameter, gs_overlay, glider_overlay):
    df_latest = loader.load_data()
    df_map = get_first_10_pH_average(df_latest)
    df_map_filter = filter_glider_assets(df_map, glider_overlay)
    
    cmin, cmax, cstep = get_clim(df_map, selected_parameter)

    # Create base figure
    map_fig = px.scatter_map(
        df_map_filter, lat="Lat [°N]", lon="Lon [°E]",
        map_style="satellite",
        zoom=8,
        color=selected_parameter,
        hover_name=selected_parameter,
        range_color=[cmin, cmax],
        # labels={"Station": "Profile"}
        
    )
    map_fig.update_layout(legend_title='Spray203')
    
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
    [Output('pHin-plot','figure'),
     Output('doxy-plot','figure')],
    [Input('interval-refresh', 'n_intervals'),
     Input('glider_overlay_checklist', 'value')]
)
def update_scatter_plots(n, glider_overlay):
    df_latest = loader.load_data()
    df_latest_filter = filter_glider_assets(df_latest, glider_overlay)

    scatter_fig_pHin = make_depth_scatter_plot(
        df_latest_filter,
        x="pHinsitu[Total]",
        title="pHinsitu[Total] vs. Depth"
    )
    scatter_fig_doxy = make_depth_scatter_plot(
        df_latest_filter,
        x="Oxygen[µmol/kg]",
        title="Oxygen[µmol/kg] vs. Depth"
    )
    
    return scatter_fig_pHin, scatter_fig_doxy

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port, debug=True)