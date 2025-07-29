import dash_bootstrap_components as dbc
from dash import dash, Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re
from data_loader import GliderDataLoader, GulfStreamLoader, MapDataLoader, GliderGridDataLoader, MPADataLoader
import datetime as dt
from typing import cast, List, Dict, Any

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

def get_MLD_avg(df):
    """
    For each (Cruise, Station) group, compute the mean pHinsitu[Total] and Chl_a[mg/m^3] for rows where
    Sigma_theta[kg/m^3] is between sigma_surface (minimum in group) and sigma_surface+0.03.
    Also compute the mixed layer depth (MLD) as the difference between max and min depth in the mask.
    Returns a DataFrame with columns:
    ['Station', 'Cruise', 'Lat [°N]', 'Lon [°E]', 'pHinsitu[Total]', 'Chl_a[mg/m^3]', 'MLD']
    """
    results = []
    for (cruise, station), group in df.groupby(['Cruise', 'Station']):
        if 'Sigma_theta[kg/m^3]' not in group.columns or 'pHinsitu[Total]' not in group.columns:
            continue
        sigma_surface = group['Sigma_theta[kg/m^3]'].min()
        mask = (group['Sigma_theta[kg/m^3]'] >= sigma_surface) & (group['Sigma_theta[kg/m^3]'] <= sigma_surface + 0.03)
        mean_pH = group.loc[mask, 'pHinsitu[Total]'].mean()
        mean_chl = group.loc[mask, 'Chl_a[mg/m^3]'].mean() if 'Chl_a[mg/m^3]' in group.columns else np.nan
        lat = group['Lat [°N]'].iloc[0] if 'Lat [°N]' in group.columns else np.nan
        lon = group['Lon [°E]'].iloc[0] if 'Lon [°E]' in group.columns else np.nan
        if 'Depth[m]' in group.columns and mask.any():
            mld = group.loc[mask, 'Depth[m]'].max() - group.loc[mask, 'Depth[m]'].min()
        else:
            mld = np.nan
        results.append({
            'Station': station,
            'Cruise': cruise,
            'Lat [°N]': lat,
            'Lon [°E]': lon,
            'pHinsitu[Total]': mean_pH,
            'Chl_a[mg/m^3]': mean_chl,
            'MLD': mld
        })
    return pd.DataFrame(results)
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
        Must contain 'Datetime' and 'unixTimestamp' columns.
    target_mark_count : int
        Approximate number of marks to generate.

    Returns:
    -------
    dict
        Dictionary of {unixTimestamp: formatted datetime string}
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
glider_ids = ['SN203', 'SN209', 'SN210', 'SN211','SN069']

# Initialize the app with a Bootstrap theme
external_stylesheets = cast(List[str | Dict[str, Any]], [dbc.themes.FLATLY])
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn


loader = GliderDataLoader(filenames=['25420901RT.txt', '25520301RT.txt'])
# Load the most recent file (automatically done if no filename provided)
df_latest = loader.load_data()
map_loader = MapDataLoader()
df_map = map_loader.load_data()
station_min, station_max = df_latest["Station"].min(), df_latest["Station"].max()
date_min, date_max = df_latest["Date"].min(), df_latest["Date"].max() 
unix_min, unix_max = df_latest["unixTimestamp"].min(), df_latest["unixTimestamp"].max() 
unix_max_minus_12hrs = unix_max - 60*60*12
marks = range_slider_marks(df_latest, 20)

app.layout = dbc.Container([
    # Top row - Header
    dbc.Row([
        dbc.Col([
            html.H2('NESSIE', className='text-info text-start',
                    style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '10px'}),
            html.P(f'Last Updated: {df_latest["Datetime"].max().strftime("%Y-%m-%d %H:%M:%S")} UTC', 
                   className='text-muted text-start',
                   style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '20px'})
        ], width=12)
    ]),
    # Map row
    dbc.Row([
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
                        allowCross=False,
                        # tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], style={'padding': '20px 10px', 'margin-top': '10px'})
            ], style={'padding': '10px', 'backgroundColor': '#e3f2fd'})
        ], width=12)
    ]),
    # Bottom row - Settings and Parameters side by side
    dbc.Row([
        # Gliders column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Gliders:"),
                        dcc.Checklist(
                            id='glider_overlay_checklist',
                            options=[{'label': s, 'value': s} for s in glider_ids],
                            value=[],
                            labelStyle={'display': 'block'}
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=2),
        # Assets column
        # dbc.Col([
        #     html.Div([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("Assets:"),
        #                 dcc.Checklist(
        #                     id='assets_checklist',
        #                     options=[
        #                         {'label': 'RV Connecticut', 'value': 'RV Connecticut'},
        #                         {'label': 'LRAUV', 'value': 'LRAUV'},
        #                         {'label': 'Drifter A', 'value': 'Drifter A'},
        #                         {'label': 'Drifter B', 'value': 'Drifter B'},
        #                         {'label': 'Drifter C', 'value': 'Drifter C'}
        #                     ],
        #                     value=[],
        #                     labelStyle={'display': 'block'}
        #                 )
        #             ])
        #         ])
        #     ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        # ], width=3),
        # Map parameters column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Map Color Variable"),
                        dcc.RadioItems(
                            id='Parameters',
                            options=[
                                {'label': 'Datetime', 'value': 'unixTimestamp'},
                                {'label': 'temperature [^oC]', 'value': 'temperature'},
                                {'label': 'salinity [psu]', 'value': 'salinity'},
                                {'label': 'pHin', 'value': 'pHin'},
                                {'label': 'rhodamine', 'value': 'rhodamine'},
                                {'label': 'MLD [m]', 'value': 'MLD'},
                            ],
                            value='unixTimestamp'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=2),
        # Map parameters column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Layers"),
                        dcc.RadioItems(
                            id='Layers',
                            options=[
                                {'label': 'Mixed Layer Depth Avg', 'value': 'MLD'},
                                {'label': 'Surface', 'value': 'Surface'}
                            ],
                            value='MLD'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=2),
        # Cast Direction column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cast Direction"),
                        dcc.RadioItems(
                            id='CastDirection',
                            options=[
                                {'label': 'Up+Down', 'value': 'UpDown'},
                                {'label': 'Upcast', 'value': 'Up'},
                                {'label': 'Downcast', 'value': 'Down'},
                                {'label': 'Mean', 'value': 'Mean'}
                            ],
                            value='UpDown'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=2),
        # Map Options column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Map Options"),
                        dcc.Checklist(
                                id='map_options',
                                options=[
                                    {'label': 'Overlay Gulf Stream', 'value': 'overlay'},
                                    {'label': 'Overlay Glider Grid', 'value': 'glider_grid'},
                                    {'label': 'Overlay Stellwagen Bank MPA', 'value': 'mpa'}
                                    ],
                                value=[],
                                labelStyle={'display': 'block'}
                            )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], width=2),
    ]),
    # Third Row - Plotting
    # (Removed main plots row)
    # Tabs for lazy loading
    dcc.Tabs(id='lazy-tabs', value='tab-phin-rho', children=[
        dcc.Tab(label='pH & Rho', value='tab-phin-rho', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='pHin-plot', style={'height': '70vh', 'width': '100%'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='rho-plot', style={'height': '70vh', 'width': '100%'})
                        ])
                    ])
                ], width=6),
            ])
        ]),
        dcc.Tab(label='Temperature and Salinity', value='tab-1', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='temp-plot', style={'height': '70vh', 'width': '100%'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='salinity-plot', style={'height': '70vh', 'width': '100%'})
                        ])
                    ])
                ], width=6),
            ])
        ]),
        dcc.Tab(label='Oxygen & Chlorophyll', value='tab-oxy-chl', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='doxy-plot', style={'height': '70vh', 'width': '100%'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='chl-plot', style={'height': '70vh', 'width': '100%'})
                        ])
                    ])
                ], width=6),
            ])
        ]),
        dcc.Tab(label='pH Diagnostics', value='tab-2', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vrs-plot', style={'height': '40vh', 'width': '100%'})
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vrs-std-plot', style={'height': '40vh', 'width': '100%'})
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ik-plot', style={'height': '40vh', 'width': '100%'})
                        ])
                    ])
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vk-plot', style={'height': '40vh', 'width': '100%'})
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vk-std-plot', style={'height': '40vh', 'width': '100%'})
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ib-plot', style={'height': '40vh', 'width': '100%'})
                        ])
                    ])
                ], width=4),
            ]),
        ]),
        # --- New Property Plot Tab ---
        dcc.Tab(label='Property Plot', value='tab-property', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select X-axis:'),
                    dcc.Dropdown(id='property-x-dropdown'),
                ], width=3),
                dbc.Col([
                    html.Label('Select Y-axis:'),
                    dcc.Dropdown(id='property-y-dropdown'),
                ], width=3),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='property-plot', style={'height': '60vh', 'width': '100%'})
                ], width=12)
            ])
        ]),
    ]),
], fluid=True, className='dashboard-container')

# --- Combined callback for map and scatter plots ---
@app.callback(
    [Output('map-plot', 'figure'),
     Output('pHin-plot', 'figure'),
     Output('doxy-plot', 'figure'),
     Output('temp-plot', 'figure'),
     Output('salinity-plot', 'figure'),
     Output('chl-plot', 'figure'),
     Output('rho-plot', 'figure'),
     Output('vrs-plot', 'figure'),
     Output('vrs-std-plot', 'figure'),
     Output('vk-plot', 'figure'),
     Output('vk-std-plot', 'figure'),
     Output('ik-plot', 'figure'),
     Output('ib-plot', 'figure'),
     Output('property-plot', 'figure'),  # New output for property plot
     Output('property-x-dropdown', 'options'), # New output for x dropdown options
     Output('property-y-dropdown', 'options')  # New output for y dropdown options
    ],
    [Input('interval-refresh', 'n_intervals'),
     Input('Parameters', 'value'),
     Input('map_options', 'value'),
     Input('glider_overlay_checklist', 'value'),
     Input('lazy-tabs', 'value'),
     Input('RangeSlider', 'value'),
     Input('Layers', 'value'),
     Input('CastDirection', 'value'),
     Input('property-x-dropdown', 'value'), # New input for x axis
     Input('property-y-dropdown', 'value')  # New input for y axis
    ]
)
def update_all_figs(n, selected_parameter, map_options, glider_overlay, selected_tab, range_value, selected_layer, selected_cast_direction, property_x, property_y):
    # Load glider and filter by selected gliders
    df_latest = loader.load_data()
    df_latest = filter_glider_assets(df_latest, glider_overlay)
    # Load map data
    df_map = map_loader.load_data()
    # load glider grid
    glider_grid_loader = GliderGridDataLoader()
    df_glider_grid = glider_grid_loader.load_data()
    # load mpa data
    mpa_loader = MPADataLoader()
    df_mpa = mpa_loader.load_data()
    # Filter by date range
    if range_value[0] == range_value[1]:
        df_latest_filter = df_latest
        df_map_filtered = df_map
    else:
        df_latest_filter = df_latest[
            (df_latest['unixTimestamp'] >= range_value[0]) &
            (df_latest['unixTimestamp'] <= range_value[1])
        ]
        df_map_filtered = df_map[
            (df_map['unixTimestamp'] >= range_value[0]) &
            (df_map['unixTimestamp'] <= range_value[1])
        ]
    # Filter by layers
    if selected_layer == 'MLD':
        df_map_filtered = df_map_filtered[df_map_filtered['Layer'] == 'MLD']
    elif selected_layer == 'Surface':
        df_map_filtered = df_map_filtered[df_map_filtered['Layer'] == 'Surface']
    # Filter by cast direction
    if selected_cast_direction == 'Mean':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Mean') | (df_map_filtered['CastDirection'] == 'Constant')]
    elif selected_cast_direction == 'UpDown':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Up') | (df_map_filtered['CastDirection'] == 'Down') | (df_map_filtered['CastDirection'] == 'Constant')]
    elif selected_cast_direction == 'Up':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Up') | (df_map_filtered['CastDirection'] == 'Constant')]
        df_latest_filter["depth_diff"] = df_latest_filter["Depth[m]"].diff()
        df_latest_filter = df_latest_filter[df_latest_filter["depth_diff"] < 0]
        df_latest_filter.drop(columns="depth_diff", inplace=True)
    elif selected_cast_direction == 'Down':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Down') | (df_map_filtered['CastDirection'] == 'Constant')]
        df_latest_filter["depth_diff"] = df_latest_filter["Depth[m]"].diff()
        df_latest_filter = df_latest_filter[df_latest_filter["depth_diff"] > 0]
        df_latest_filter.drop(columns="depth_diff", inplace=True)
    # Dataframes for map plot
    df_ship = df_map_filtered[df_map_filtered['Cruise'] == "RV Connecticut"]
    df_SN203 = df_map_filtered[df_map_filtered['Cruise'] == "25520301"]
    df_SN209 = df_map_filtered[df_map_filtered['Cruise'] == "25420901"]

    # Handle empty DataFrame case
    is_map_df = isinstance(df_map_filtered, pd.DataFrame)
    is_latest_df = isinstance(df_latest_filter, pd.DataFrame)
    if not is_map_df or df_map_filtered.empty:
        blank_fig = go.Figure()
        return (
            blank_fig, blank_fig, blank_fig,
            blank_fig, blank_fig, blank_fig,
            blank_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), [], []
        )
    # Set map center
    if len(df_ship) > 0:
        last_ship_lat = np.array(df_ship['lat'])[-1]
        last_ship_lon = np.array(df_ship['lon'])[-1]
    else:
        last_ship_lat = np.array(df_map_filtered['lat'])[-1]
        last_ship_lon = np.array(df_map_filtered['lon'])[-1]

    # Last Glider Location SN203
    if len(df_SN203) > 0:
        last_glider_lat_SN203 = np.array(df_SN203['lat'])[-1]
        last_glider_lon_SN203 = np.array(df_SN203['lon'])[-1]
    else:
        last_glider_lat_SN203 = []
        last_glider_lon_SN203 = []
    # Last Glider Location SN209
    if len(df_SN209) > 0:
        last_glider_lat_SN209 = np.array(df_SN209['lat'])[-1]
        last_glider_lon_SN209 = np.array(df_SN209['lon'])[-1]
    else:
        last_glider_lat_SN209 = []
        last_glider_lon_SN209 = []
    map_fig = go.Figure()
    # Set hard color limits for pHin and rhodamine
    if selected_parameter == 'pHin' and selected_layer == 'Surface':
        cmin, cmax = 8, 8.2
        cscale = 'bluered'
    elif selected_parameter == 'pHin' and selected_layer == 'MLD':
        cmin, cmax = 8, 8.2
        cscale = 'bluered'
    elif selected_parameter == 'rhodamine':
        cmin, cmax = 0, 2
        cscale = 'jet'
    else:
        cmin, cmax = None, None
        cscale = 'Cividis'
    # Only do this if coloring by unixTimestamp
    if selected_parameter == 'unixTimestamp' and len(df_ship) > 0:
        # Get unique unixTimestamps and corresponding datetimes
        unix_vals = df_ship['unixTimestamp'].values
        datetimes = df_ship['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').values

        # Choose 10 evenly spaced indices
        n_ticks = 10
        if len(unix_vals) > n_ticks:
            idxs = np.linspace(0, len(unix_vals) - 1, n_ticks, dtype=int)
            tickvals = unix_vals[idxs]
            ticktext = datetimes[idxs]
        else:
            tickvals = unix_vals
            ticktext = datetimes
    else:
        tickvals = None
        ticktext = None
   
    # Set hovertext based on selected parameter
    ship_hovertext = df_ship['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_ship[selected_parameter]
    
    map_fig.add_trace(go.Scattermap(
            lat=df_ship['lat'],
            lon=df_ship['lon'],
            mode='markers',
            name='RV Connecticut',
            hovertext=ship_hovertext,
            marker=dict(
                size=6,
                color=df_ship[selected_parameter],
                colorscale=cscale,
                showscale=True,
                colorbar=dict(len=0.6,tickvals=tickvals,ticktext=ticktext),
                cmin=cmin,
                cmax=cmax,
            ),
        ))
    map_fig.add_trace(go.Scattermap(
        lat=[last_ship_lat],
        lon=[last_ship_lon],
        mode='markers',
        name='RV Connecticut Last Location',
        marker=dict(
            size=20,
            symbol='ferry',
            showscale=False,
        ),
        showlegend=False
    ))
    # Set hovertext for SN209 based on selected parameter
    sn203_hovertext = df_SN203['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN203[selected_parameter]
    
    map_fig.add_trace(go.Scattermap(
        lat=df_SN203['lat'],
        lon=df_SN203['lon'],
        mode='markers',
        name='SN203',
        hovertext=sn203_hovertext,
        marker=dict(
            size=6, 
            color=df_SN203[selected_parameter],
            colorscale=cscale,  
            showscale=False,
            colorbar=dict(len=0.6),
            cmin=cmin,
            cmax=cmax,
        ),
    ))
    map_fig.add_trace(go.Scattermap(
        lat=[last_glider_lat_SN203],
        lon=[last_glider_lon_SN203],
        mode='markers',
        name='SN203 Last Location',
        marker=dict(
            size=20,
            symbol='airport',
            showscale=False,
        ),
        showlegend=False
    ))
    # Set hovertext for SN209 based on selected parameter
    sn209_hovertext = df_SN209['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN209[selected_parameter]
    
    map_fig.add_trace(go.Scattermap(
    lat=df_SN209['lat'],
    lon=df_SN209['lon'],
    mode='markers',
    name='SN209',
    hovertext=sn209_hovertext,
    marker=dict(
        size=6, 
        color=df_SN209[selected_parameter],
        colorscale=cscale,
        showscale=False,
        cmin=cmin,
        cmax=cmax,
    ),
    ))
    map_fig.add_trace(go.Scattermap(
        lat=[last_glider_lat_SN209],
        lon=[last_glider_lon_SN209],
        mode='markers',
        name='SN209 Last Location',
        marker=dict(
            size=20,
            symbol='airport',
            showscale=False,
        ),
        showlegend=False
    ))
    if 'overlay' in map_options:
        map_fig.add_trace(go.Scattermap(
        lat=GulfStreamBounds['Lat'],
        lon=GulfStreamBounds['Lon'],
        mode='lines',
        name='Gulf Stream',
        marker=dict(size=6, color='deepskyblue'),
        line=dict(width=2, color='deepskyblue'),
        ))

    if 'glider_grid' in map_options:
        map_fig.add_trace(go.Scattermap(
            lat=df_glider_grid['Lat'],
            lon=df_glider_grid['Lon'],
            mode='markers',
            name='Glider Grid',
            marker=dict(size=6, color='black', opacity=0.5),
            text = df_glider_grid['Grid_ID'],
            textposition = "bottom right",
        ))

    if 'mpa' in map_options:
        map_fig.add_trace(go.Scattermap(
            lat=df_mpa['Lat'],
            lon=df_mpa['Lon'],
            mode='lines',
            name='Stellwagen Bank MPA',
            line=dict(width=4, color='red'),
            text = 'Stellwagen Bank MPA',
            textposition = "bottom right",
        ))

    # map_fig.update_layout(map = {'zoom': 8, 'style': 'satellite', 'center': {'lat': last_ship_lat, 'lon': last_ship_lon}})
    map_fig.update_layout(
        map_style="white-bg",
        map_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "Esri, Garmin, GEBCO, NOAA NGDC, and other contributors",
                "source": [
                    "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
                ]
            },
        ],
    )   
    map_fig.update_layout(
        map=dict(
            center={'lat': last_ship_lat, 'lon': last_ship_lon},
            zoom=8,
        )
    )
    map_fig.update_layout(margin={"r":0,"t":20,"l":0,"b":0})

    # For scatter plots: full filtered DataFrame
    if len(df_latest_filter) == 0:
        scatter_fig_pHin = go.Figure()
        scatter_fig_doxy = go.Figure()
        scatter_fig_temp = go.Figure()
        scatter_fig_salinity = go.Figure()
        scatter_fig_chl = go.Figure()
        scatter_fig_rho = go.Figure()
        scatter_fig_vrs = go.Figure()
        scatter_fig_vrs_std = go.Figure()
        scatter_fig_vk = go.Figure()
        scatter_fig_vk_std = go.Figure()
        scatter_fig_ik = go.Figure()
        scatter_fig_ib = go.Figure()
    else:
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
        scatter_fig_temp = make_depth_scatter_plot(
            df_latest_filter,
            x="Temperature[°C]",
            title="Temperature[°C] vs. Depth"
        )
        scatter_fig_salinity = make_depth_scatter_plot(
            df_latest_filter,
            x="Salinity[pss]",
            title="Salinity[pss] vs. Depth"
        )
        scatter_fig_chl = make_depth_scatter_plot(
            df_latest_filter,
            x="Chl_a[mg/m^3]",
            title="Chl_a[mg/m^3] vs. Depth"
        )
        scatter_fig_rho = make_depth_scatter_plot(
            df_latest_filter,
            x="Sigma_theta[kg/m^3]",
            title="Sigma_theta[kg/m^3] vs. Depth"
        )
        scatter_fig_vrs = make_depth_scatter_plot(
            df_latest_filter,
            x="VRS[Volts]",
            title="VRS[Volts] vs. Depth"
        )
        scatter_fig_vrs_std = make_depth_scatter_plot(
            df_latest_filter,
            x="VRS_STD[Volts]",
            title="VRS_STD[Volts] vs. Depth"
        )
        scatter_fig_vk = make_depth_scatter_plot(
            df_latest_filter,
            x="VK[Volts]",
            title="VK[Volts] vs. Depth"
        )
        scatter_fig_vk_std = make_depth_scatter_plot(
            df_latest_filter,
            x="VK_STD[Volts]",
            title="VK_STD[Volts] vs. Depth"
        )
        scatter_fig_ik = make_depth_scatter_plot(
            df_latest_filter,
            x="IK[nA]",
            title="IK[nA] vs. Depth"
        )
        scatter_fig_ib = make_depth_scatter_plot(
            df_latest_filter,
            x="Ib[nA]",
            title="Ib[nA] vs. Depth"
        )
    # Property plot dropdown options
    if len(df_latest_filter) > 0:
        dropdown_options = [
            {'label': col, 'value': col}
            for col in df_latest_filter.columns if 'QF' not in col
        ]
        # Get unique unixTimestamps and corresponding datetimes
        unix_vals = df_latest_filter['unixTimestamp'].values
        datetimes = df_latest_filter['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').values

        # Choose 10 evenly spaced indices
        n_ticks = 10
        if len(unix_vals) > n_ticks:
            idxs = np.linspace(0, len(unix_vals) - 1, n_ticks, dtype=int)
            tickvals = unix_vals[idxs]
            ticktext = datetimes[idxs]
        else:
            tickvals = unix_vals
            ticktext = datetimes
    else:
        dropdown_options = []
    # Property plot figure
    if selected_tab == 'tab-property' and property_x and property_y and len(df_latest_filter) > 0:
        fig_property = px.scatter(
            df_latest_filter, x=property_x, y=property_y,
            
            # colorbar=dict(len=1,tickvals=tickvals,ticktext=ticktext),
            title=f'{property_x} vs. {property_y}',
            template='plotly_white',
            color='unixTimestamp' if 'unixTimestamp' in df_latest_filter.columns else None,
        )
        fig_property.update_yaxes(autorange="reversed")
        fig_property.update_traces(marker=dict(size=6))
        if 'unixTimestamp' in df_latest_filter.columns:
            fig_property.update_layout(
                coloraxis_colorbar=dict(
                    len=1,
                    tickvals=tickvals,
                    ticktext=ticktext
                )
            )
    else:
        fig_property = go.Figure()
    # Always update map, pHin, doxy
    # Tab 1: update temp, salinity; Tab 2: update vrs, vrs_std, vk, vk_std, ik, ib; Tab-phin-rho: update pHin and Rho; Tab-oxy-chl: update O2 and Chl; Tab-property: update property plot
    if selected_tab == 'tab-phin-rho':
        return (
            map_fig, scatter_fig_pHin, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            scatter_fig_rho,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-1':
        return (
            map_fig, dash.no_update, dash.no_update,
            scatter_fig_temp, scatter_fig_salinity, dash.no_update,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-oxy-chl':
        return (
            map_fig, dash.no_update, scatter_fig_doxy,
            dash.no_update, dash.no_update, scatter_fig_chl,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-2':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            scatter_fig_vrs, scatter_fig_vrs_std, scatter_fig_vk, scatter_fig_vk_std, scatter_fig_ik, scatter_fig_ib,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-property':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            fig_property, dropdown_options, dropdown_options
        )
    else:
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )

@app.callback(
    [
        Output('RangeSlider', 'min'),
        Output('RangeSlider', 'max'),
        Output('RangeSlider', 'value'),
        Output('RangeSlider', 'marks'),
    ],
    [
        Input('glider_overlay_checklist', 'value'),
        Input('interval-refresh', 'n_intervals'),
    ]
)

def update_range_slider(glider_overlay, n):
    df_map = map_loader.load_data()
    if df_map.empty:
        # Return safe defaults if no data
        return 0, 1, [0, 1], {0: "No data"}
    unix_min = df_map["unixTimestamp"].min()
    unix_max = df_map["unixTimestamp"].max()
    unix_max_minus_12hrs = unix_max - 60*60*12
    marks = range_slider_marks(df_map, 20)
    return unix_min, unix_max, [unix_min, unix_max], marks

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=str(port), debug=True)
    app.run(host="0.0.0.0", port=str(port), debug=True)
    app.run(host="0.0.0.0", port=str(port), debug=True)