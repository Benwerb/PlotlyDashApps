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
# from nessie_interpolation_function import create_spatial_interpolation
import datetime as dt
from typing import cast, List, Dict, Any
import pytz
from plotly.validator_cache import ValidatorCache
import numpy as np
# from scipy.interpolate import griddata
from database_tools import get_dives_data, get_map_data, get_ph_drift_data, get_available_missions, get_mission_metadata, get_dive_range
import psycopg2
from urllib.parse import urlparse


# Define variable-specific percentile limits
def get_clim(df, color_column):
    if color_column == 'ChlorophyllA':
        lower, upper = np.percentile(df[color_column].dropna(), [5, 99])
    else:
        lower, upper = np.percentile(df[color_column].dropna(), [1, 99])

    step = max(round((upper - lower) / 100, 3), 0.001)

    return lower, upper, step

def apply_common_scatter_styling(fig, colorbar_title="divenumber", colorbar_orientation='v'):
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
    df, x, y="depth", title=None,
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
        Column name for y-axis (default: "depth").
    title : str or None
        Plot title. If None, a default is generated.
    labels : dict or None
        Custom axis labels.
    colorbar_title : str or None
        Title for the colorbar.
    colorbar_orientation : str
        'v' or 'h' for vertical or horizontal colorbar.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        Styled scatter plot. Returns empty figure with message if columns missing.
    """
    # Check if required columns exist
    if x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing data column: {x if x not in df.columns else y}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    # Filter out rows where x or y is NaN
    df_valid = df.dropna(subset=[x, y])
    
    if len(df_valid) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data for {x} vs {y}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    if title is None:
        title = f"{x} vs. {y}"

    try:
        fig = px.scatter(
            df_valid,
            x=x,
            y=y,
            labels=labels,
            title=title
        )

        # Reverse y-axis if it's a depth column
        if 'depth' in y.lower():
            fig.update_yaxes(autorange="reversed")

        # Add colorbar settings
        colorbar_config = dict(
            orientation=colorbar_orientation,
            len=0.8,
            thickness=25
        )

        if colorbar_orientation == 'v':
            colorbar_config.update(x=1.02, y=0.5)
        else:
            colorbar_config.update(x=0.5, y=-0.2, xanchor='center', yanchor='top')

        fig.update_layout(coloraxis_colorbar=colorbar_config)
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )

    return fig

def make_depth_line_plot(
    df, x, y="depth", title=None,
    color="unixtime",
    labels=None,
    colorbar_title=None,
    colorbar_orientation='v'
):
    """
    Create a reusable line plot of a variable vs. depth with consistent styling.

    Parameters:
    ----------
    df : DataFrame
        Input data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis (default: "depth").
    title : str or None
        Plot title. If None, a default is generated.
    color : str
        Column to use for color.
    labels : dict or None
        Custom axis labels.
    colorbar_title : str or None
        Title for the colorbar. Defaults to the `color` column name.
    colorbar_orientation : str
        'v' or 'h' for vertical or horizontal colorbar.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        Styled line plot. Returns empty figure with message if columns missing.
    """
    # Check if required columns exist
    if x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing data column: {x if x not in df.columns else y}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    # Filter out rows where x or y is NaN
    df_valid = df.dropna(subset=[x, y])
    
    if len(df_valid) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data for {x} vs {y}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    if title is None:
        title = f"{x} vs. {y}"
    
    # Set up labels - rename "unixtime" to "datetime" for better UX
    if labels is None:
        color_label = "datetime" if color == "unixtime" else color
        labels = {x: x, y: y, color: color_label}
    
    if colorbar_title is None:
        colorbar_title = "datetime" if color == "unixtime" else color
    
    # Check if color column exists, use default if not
    if color not in df_valid.columns:
        color = None
    
    try:
        # Get unique values and assign colors
        if color:
            # If using unixtime for color, create a formatted datetime column for legend
            if color == "unixtime" and 'datetime' in df_valid.columns:
                # Create a formatted datetime string column for the legend
                df_valid = df_valid.copy()
                df_valid['datetime_formatted'] = df_valid['datetime'].dt.strftime('%m/%d %H:%M')
                color_for_legend = 'datetime_formatted'
            else:
                color_for_legend = color
            
            # Reverse the order so most recent data is plotted LAST (on top)
            # This also puts most recent first in the legend
            df_valid = df_valid.iloc[::-1].copy()
            
            unique_vals = df_valid[color_for_legend].unique()
            num_colors = len(unique_vals)
            
            if num_colors == 1:
                viridis_colors = px.colors.sample_colorscale("Cividis", 0.5)
            else:
                viridis_colors = px.colors.sample_colorscale("Cividis", [i / (num_colors - 1) for i in range(num_colors)])
            
            fig = px.line(
                df_valid,
                x=x,
                y=y,
                color=color_for_legend,
                labels=labels,
                title=title,
                markers=True,
                color_discrete_sequence=viridis_colors,
                line_group=color_for_legend
            )
        else:
            # No color column, just plot the line
            fig = px.line(
                df_valid,
                x=x,
                y=y,
                labels=labels,
                title=title,
                markers=True
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
    except Exception as e:
        print(f"Error creating depth line plot: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    # fig.update_layout(
    # legend=dict(
    #     x=0.01,
    #     y=0.99,
    #     xanchor='left',
    #     yanchor='top',
    #     bgcolor='rgba(255,255,255,0.6)',  # semi-transparent white
    #     bordercolor='black',
    #     borderwidth=1,
    #     traceorder='normal',
    #     font=dict(size=10),
    # )
    # )
    # fig.update_layout(
    #     legend=dict(
    #         orientation="h",       # horizontal layout
    #         yanchor="bottom",      # anchor the legend box from its bottom
    #         y=1.02,                # slightly above the plot area (1.0 is top edge)
    #         xanchor="left",
    #         x=0,
    #         bgcolor='rgba(255,255,255,0.7)',  # optional background
    #         bordercolor='black',
    #         borderwidth=1,
    #         font=dict(size=10)
    #     )
    # )

    return fig

def range_slider_marks(df, target_mark_count=10):
    """
    Generate RangeSlider marks at evenly spaced full-hour intervals,
    aligned to the nearest hour, based on a target number of marks.

    Parameters:
    ----------
    df : pandas.DataFrame
        Must contain 'datetime' and 'unixTimestamp' columns.
    target_mark_count : int
        Approximate number of marks to generate.

    Returns:
    -------
    dict
        Dictionary of {unixTimestamp: formatted datetime string}
    """
    # Sort and get min/max
    df = df.sort_values("datetime")
    t_min = df['datetime'].min()
    t_max = df['datetime'].max()

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
    # marks = {int(ts.timestamp()): ts.strftime('%m/%d %H:%M') for ts in timestamps}
    marks = {
        int(ts.timestamp()): {
            'label': ts.strftime('%m/%d') + '\n' + ts.strftime('%H:%M'),
            'style': {'fontSize': '12px', 'whiteSpace': 'pre'}
        }
        for ts in timestamps
    }

    return marks

def make_contour_plot(df, z_column, title=None):
    """
    Create a contour plot of datetime vs depth with the specified z-column as contours.

    Parameters:
    ----------
    df : DataFrame
        Input data with 'datetime', 'depth', and z_column columns.
    z_column : str
        Column name for the contour values.
    title : str or None
        Plot title. If None, a default is generated.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        Styled contour plot. Returns empty figure with message if columns missing.
    """
    # Check if required columns exist
    if 'datetime' not in df.columns or 'depth' not in df.columns or z_column not in df.columns:
        fig = go.Figure()
        missing_cols = []
        if 'datetime' not in df.columns:
            missing_cols.append('datetime')
        if 'depth' not in df.columns:
            missing_cols.append('depth')
        if z_column not in df.columns:
            missing_cols.append(z_column)
        
        fig.add_annotation(
            text=f"Missing data columns: {', '.join(missing_cols)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    # Filter out rows where required columns are NaN
    df_valid = df.dropna(subset=['datetime', 'depth', z_column])
    
    if len(df_valid) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data for {z_column} contour plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    if title is None:
        title = f"{z_column} vs. Time and Depth"
    
    try:
        # Create contour plot using plotly
        fig = go.Figure()
        
        # Add contour plot
        fig.add_trace(go.Contour(
            x=df_valid['datetime'],
            y=df_valid['depth'],
            z=df_valid[z_column],
            colorscale='Viridis',
            line_smoothing=0.85,
            contours=dict(
                showlines=True,
                showlabels=True,
                labelfont=dict(size=12, color="white")
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),  # Reverse depth axis so surface is at top
            template='plotly_white',
            coloraxis_colorbar=dict(
                title=z_column,
                orientation='v',
                len=0.8,
                thickness=25,
                x=1.02,
                y=0.5
            )
        )
        
        # Format x-axis to show dates nicely
        fig.update_xaxes(
            tickformat='%m/%d %H:%M',
            tickangle=45
        )
        
    except Exception as e:
        print(f"Error creating contour plot: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating contour plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
    
    return fig

def get_latest_divenumber() -> int | None:
    """
    Fetch the latest (maximum) divenumber from public.real_time_binned using DATABASE_URL.

    Returns the integer divenumber or None if unavailable.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None

    conn = None
    try:
        # Render provides full Postgres URLs; require SSL if supported
        conn = psycopg2.connect(db_url, sslmode="require")
        with conn.cursor() as cur:
            cur.execute("select max(divenumber) from public.real_time_binned;")
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
            return None
    except Exception:
        return None
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


# Initialize the app with a Bootstrap theme
external_stylesheets = cast(List[str | Dict[str, Any]], [dbc.themes.FLATLY])
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn

# Fetch available missions from database with metadata (sorted by most recent data)
missions_metadata = get_mission_metadata()
available_missions = missions_metadata['mission_id'].tolist()
default_mission = available_missions[0] if available_missions else "25820301"

# Create dropdown options with date ranges
mission_dropdown_options = [
    {
        'label': f"{row['mission_id']} ({row['start_date'].strftime('%m/%d/%y')} - {row['end_date'].strftime('%m/%d/%y')}, {int(row['total_dives'])} dives)",
        'value': row['mission_id']
    }
    for _, row in missions_metadata.iterrows()
]

# Get the FULL range of dive numbers for the default mission (without loading all data)
try:
    station_min, station_max = get_dive_range(default_mission)
except Exception as e:
    print(f"Error getting dive range: {e}")
    station_min, station_max = 1, 10

# Default to last 10 dives, but show full range in slider
default_slider_min = max(station_min, station_max - 10)
default_slider_max = station_max

# Load the most recent mission data (only last 10 dives for initial display)
# Use minimal columns for initial load - will be optimized per tab in callback
initial_columns = ['divenumber', 'depth', 'unixtime', 'phin', 'phin_canb', 'tc', 'psal', 'doxy', 'sigma', 'vrse', 'vrse_std', 'vk', 'vk_std', 'ik', 'ib']
map_columns = ['lat', 'lon', 'unixtime', 'divenumber', 'depth']
ph_drift_columns = ['divenumber', 'phin', 'phin_canb', 'unixtime', 'depth']

try:
    df_latest = get_dives_data(default_mission, columns=initial_columns)
    df_map = get_map_data(default_mission, depth=0, columns=map_columns)
    df_ph_drift = get_ph_drift_data(default_mission, depth=450, columns=ph_drift_columns)
except Exception as e:
    print(f"Error loading initial data: {e}")
    df_latest = pd.DataFrame()
    df_map = pd.DataFrame()
    df_ph_drift = pd.DataFrame()

# Safe extraction of datetime values
if len(df_latest) > 0 and 'unixtime' in df_latest.columns:
    date_min, date_max = df_latest["unixtime"].min(), df_latest["unixtime"].max() 
    unix_min, unix_max = df_latest["unixtime"].min(), df_latest["unixtime"].max() 
    unix_max_minus_12hrs = unix_max - 60*60*12
    datetime_max = df_latest["unixtime"].max()
else:
    date_min = date_max = unix_min = unix_max = datetime_max = None
    unix_max_minus_12hrs = None

app.layout = dbc.Container([
    # Top row - Header
    dbc.Row([
        dbc.Col([
            html.H2('GlideApp2', className='text-info text-start',
                    style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '10px'}),
        ], width=12)
    ]),
    # Mission selection row
    dbc.Row([
        dbc.Col([
            html.Label('Select Mission:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='mission-dropdown',
                options=mission_dropdown_options,
                value=default_mission,
                clearable=False,
                style={'width': '100%'}
            )
        ], xs=12, sm=12, md=8, lg=6, xl=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='mission-info', style={'fontSize': '14px'})
                ], style={'padding': '10px'})
            ], style={'height': '100%'})
        ], xs=12, sm=12, md=4, lg=6, xl=6)
    ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}),
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
                # Range slider - shows full dive range, defaults to last 10
                html.Div([
                    html.Label('Dive Number Range:', 
                              style={'fontWeight': 'bold', 'fontSize': '14px', 'marginBottom': '5px'}),
                    dcc.RangeSlider(
                        id='RangeSlider',
                        updatemode='mouseup',  # don't let it update till mouse released
                        min=station_min,  # Full range minimum
                        max=station_max,  # Full range maximum
                        step=1,
                        value=[default_slider_min, default_slider_max],  # Default to last 10 dives
                        marks=None,
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": True, "template": "Dive {value}"}
                    )
                ], style={
                    'padding': '10px 5px',
                    'marginTop': '10px',
                    'width': '100%',
                    'boxSizing': 'border-box'
                })
            ], style={'padding': '10px', 'backgroundColor': '#e3f2fd'})
        ], width=12)
    ]),
    # # Bottom row - Settings and Parameters side by side
    # dbc.Row([
    #     # Gliders column
    #     dbc.Col([
    #         html.Div([
    #             dbc.Card([
    #                 dbc.CardBody([
    #                     html.H4("Gliders:"),
    #                     dcc.Checklist(
    #                         id='glider_overlay_checklist',
    #                         options=[{'label': s, 'value': s} for s in glider_ids],
    #                         value=[],
    #                         labelStyle={'display': 'block'}
    #                     )
    #                 ])
    #             ])
    #         ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
    #     ], xs=12, sm=12, md=6, lg=4, xl=2),
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
        # ], xs=12, md=2),
        # Map parameters column
        # dbc.Col([
        #     html.Div([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("Map Color Variable"),
        #                 dcc.RadioItems(
        #                     id='Parameters',
        #                     options=[
        #                         {'label': 'Datetime', 'value': 'unixtime'},
        #                         {'label': 'temperature [^oC]', 'value': 'tc'},
        #                         {'label': 'salinity [psu]', 'value': 'psal'},
        #                         {'label': 'pHin', 'value': 'phin'},
        #                         {'label': 'rhodamine', 'value': 'chla'},
        #                         # {'label': 'MLD [m]', 'value': 'MLD'},
        #                     ],
        #                     value='unixTimestamp'
        #                 )
        #             ])
        #         ])
        #     ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        # ], xs=12, sm=12, md=6, lg=4, xl=2),
        # # Map parameters column
        # dbc.Col([
        #     html.Div([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("Layers"),
        #                 dcc.RadioItems(
        #                     id='Layers',
        #                     options=[
        #                         {'label': 'Mixed Layer Depth Avg', 'value': 'MLD'},
        #                         {'label': 'Surface', 'value': 'Surface'}
        #                     ],
        #                     value='MLD'
        #                 )
        #             ])
        #         ])
        #     ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        # ], xs=12, sm=12, md=6, lg=4, xl=2),
        # Cast Direction column
        # dbc.Col([
        #     html.Div([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("Cast Direction"),
        #                 dcc.RadioItems(
        #                     id='CastDirection',
        #                     options=[
        #                         {'label': 'Up+Down', 'value': 'UpDown'},
        #                         {'label': 'Upcast', 'value': 'Up'},
        #                         {'label': 'Downcast', 'value': 'Down'},
        #                         {'label': 'Mean', 'value': 'Mean'}
        #                     ],
        #                     value='Up'
        #                 )
        #             ])
        #         ])
        #     ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        # ], xs=12, sm=12, md=6, lg=4, xl=2),
        # Map Options column
    #     dbc.Col([
    #         html.Div([
    #             dbc.Card([
    #                 dbc.CardBody([
    #                     html.H4("Map Options"),
    #                     dcc.Checklist(
    #                             id='map_options',
    #                             options=[
    #                                 {'label': 'Drifters', 'value': 'drifters'},
    #                                 {'label': 'Overlay Gulf Stream', 'value': 'overlay'},
    #                                 {'label': 'Overlay Glider Grid', 'value': 'glider_grid'},
    #                                 {'label': 'Overlay Stellwagen Bank MPA', 'value': 'mpa'},
    #                                 {'label': 'Overlay gomofs tracks', 'value': 'gomofs'},
    #                                 {'label': 'Overlay doppio tracks', 'value': 'doppio'}
    #                                 ],
    #                             value=[],
    #                             labelStyle={'display': 'block'}
    #                         )
    #                 ])
    #             ])
    #         ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
    #     ], xs=12, sm=12, md=6, lg=4, xl=2),
    # ]),
    # Third Row - Plotting
    # (Removed main plots row)
    # Tabs for lazy loading
    dcc.Tabs(id='lazy-tabs', value='tab-phin-phin-canyonb', children=[
        dcc.Tab(label='pHin & ΔpHin', value='tab-phin-phin-canyonb', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='pHin-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ph-delta-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ])
        ]),
        # --- pH Drift Tab ---
        dcc.Tab(label='pH Drift', value='tab-ph-drift', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select Depth (m):', style={'fontWeight': 'bold', 'fontSize': '14px', 'marginBottom': '5px'}),
                    dcc.Slider(
                        id='ph-drift-depth-dropdown',
                        min=0,
                        max=1000,
                        step=10,
                        value=450,  # Default to 450 m
                        marks={i: f'{i}m' for i in range(0, 1001, 100)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], xs=12, sm=12, md=12, lg=12, xl=12, style={'marginBottom': '20px', 'padding': '10px'})
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='ph-drift-plot', style={'height': '60vh', 'minHeight': '300px', 'width': '100%'})
                ], xs=12, sm=12, md=12,  lg=12, xl=12)
            ])
        ]),
        dcc.Tab(label='Temperature and Salinity', value='tab-temp-psal', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='temp-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='salinity-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ])
        ]),
        dcc.Tab(label='Oxygen & Sigma Theta', value='tab-oxy-sigma', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='doxy-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='chl-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ])
        ]),
        dcc.Tab(label='pH Diagnostics', value='tab-2', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vrs-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vrs-std-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ik-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vk-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vk-std-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ib-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
            ]),
        ]),
        # --- Contour Plot Tab ---
        dcc.Tab(label='Contour Plot', value='tab-contour', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select Z-axis Variable:'),
                    dcc.Dropdown(id='contour-z-dropdown', value='phin'),
                ], xs=12, sm=12, md=12, lg=12, xl=12, style={'marginBottom': '20px', 'padding': '10px'}),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='contour-plot', style={'height': '60vh', 'minHeight': '300px', 'width': '100%'})
                ], xs=12, sm=12, md=12, lg=12, xl=12)
            ])
        ]),
        # --- Property Plot Tab ---
        dcc.Tab(label='Property Plot', value='tab-property', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select X-axis:'),
                    dcc.Dropdown(id='property-x-dropdown',value='phin'),
                ], xs=12, sm=12, md=3, lg=3, xl=3),
                dbc.Col([
                    html.Label('Select Y-axis:'),
                    dcc.Dropdown(id='property-y-dropdown'),
                ], xs=12, sm=12, md=3, lg=3, xl=3),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='property-plot', style={'height': '60vh', 'minHeight': '300px', 'width': '100%'})
                ], xs=12, sm=12, md=12, lg=12, xl=12)
            ])
        ]),
    ]),
], fluid=True, className='dashboard-container')

# --- Callback to refresh mission dropdown options on interval ---
@app.callback(
    Output('mission-dropdown', 'options'),
    [Input('interval-refresh', 'n_intervals')]
)
def refresh_mission_dropdown(n):
    """
    Periodically refresh the mission dropdown options to show new missions.
    This ensures new data in the database is reflected in the UI.
    """
    try:
        # Fetch fresh metadata from database
        fresh_metadata = get_mission_metadata()
        
        # Create updated dropdown options with date ranges
        updated_options = [
            {
                'label': f"{row['mission_id']} ({row['start_date'].strftime('%m/%d/%y')} - {row['end_date'].strftime('%m/%d/%y')}, {int(row['total_dives'])} dives)",
                'value': row['mission_id']
            }
            for _, row in fresh_metadata.iterrows()
        ]
        return updated_options
    except Exception as e:
        print(f"Error refreshing mission dropdown: {e}")
        # Return existing options as fallback
        return mission_dropdown_options

# --- Callback to update RangeSlider and mission info when mission changes ---
@app.callback(
    [Output('RangeSlider', 'min'),
     Output('RangeSlider', 'max'),
     Output('RangeSlider', 'value'),
     Output('mission-info', 'children')],
    [Input('mission-dropdown', 'value')]
)
def update_range_slider_and_info(selected_mission):
    """
    Update the range slider and display mission metadata based on the selected mission.
    
    The slider shows the FULL range of dive numbers available for the mission,
    but defaults to displaying only the last 10 dives.
    """
    if not selected_mission:
        return 1, 10, [1, 10], "No mission selected"
    
    # Get the FULL range of dive numbers for this mission (without loading all data)
    min_dive, max_dive = get_dive_range(selected_mission)
    
    if min_dive == 1 and max_dive == 10:  # Check if it's the fallback default
        return 1, 10, [1, 10], "No data available for this mission"
    
    # Default value: last 10 dives (but user can drag slider to see all)
    default_min = max(min_dive, max_dive - 10)
    
    # Get FRESH mission metadata for display (don't use stale global variable)
    try:
        fresh_metadata = get_mission_metadata()
        mission_meta = fresh_metadata[fresh_metadata['mission_id'] == selected_mission]
        if len(mission_meta) > 0:
            meta = mission_meta.iloc[0]
            mission_info = html.Div([
                html.Strong("Mission Details:"),
                html.Br(),
                html.Span(f"📅 Start: {meta['start_date'].strftime('%Y-%m-%d %H:%M')}"),
                html.Br(),
                html.Span(f"📅 End: {meta['end_date'].strftime('%Y-%m-%d %H:%M')}"),
                html.Br(),
                html.Span(f"🌊 Total Dives: {int(meta['total_dives'])}"),
            ])
        else:
            mission_info = f"Mission: {selected_mission}"
    except Exception as e:
        print(f"Error fetching mission metadata: {e}")
        mission_info = f"Mission: {selected_mission}"
    
    # Return: slider_min, slider_max, slider_value [start, end], mission_info
    return min_dive, max_dive, [default_min, max_dive], mission_info

# --- Combined callback for map and scatter plots ---
@app.callback(
    [Output('map-plot', 'figure'),
     Output('pHin-plot', 'figure'),
     Output('doxy-plot', 'figure'),
     Output('temp-plot', 'figure'),
     Output('salinity-plot', 'figure'),
     Output('chl-plot', 'figure'),
     Output('ph-delta-plot', 'figure'),
     Output('ph-drift-plot', 'figure'),
     Output('vrs-plot', 'figure'),
     Output('vrs-std-plot', 'figure'),
     Output('vk-plot', 'figure'),
     Output('vk-std-plot', 'figure'),
     Output('ik-plot', 'figure'),
     Output('ib-plot', 'figure'),
     Output('contour-plot', 'figure'),  # New output for contour plot
     Output('contour-z-dropdown', 'options'), # New output for contour z dropdown options
     Output('property-plot', 'figure'),  # Property plot output
     Output('property-x-dropdown', 'options'), # Property plot x dropdown options
     Output('property-y-dropdown', 'options')  # Property plot y dropdown options
    ],
    [Input('interval-refresh', 'n_intervals'),
    Input('mission-dropdown', 'value'),
    Input('RangeSlider', 'value'),
    #  Input('Parameters', 'value'),
    #  Input('map_options', 'value'),
    #  Input('glider_overlay_checklist', 'value'),
     Input('lazy-tabs', 'value'),
    #  Input('RangeSlider', 'value'),
    Input('contour-z-dropdown', 'value'),
    Input('property-x-dropdown', 'value'),
    Input('property-y-dropdown', 'value'),
    Input('ph-drift-depth-dropdown', 'value'),
    ]
)
def update_all_figs(n, selected_mission, range_slider_value, selected_tab, contour_z, property_x, property_y, ph_drift_depth):
    # Load glider and filter by selected gliders
    min_dive = int(range_slider_value[0])
    max_dive = int(range_slider_value[1])
    # print(f"Mission: {selected_mission}, Range: {min_dive} to {max_dive}")
    
    # Define columns needed for each plot type
    # Map data needs: lat, lon, unixtime, datetime, divenumber
    map_columns = ['lat', 'lon', 'unixtime', 'divenumber', 'depth']
    
    # pH drift data needs: divenumber, phin, phin_canb, unixtime, datetime
    ph_drift_columns = ['divenumber', 'phin', 'phin_canb', 'unixtime', 'depth']
    
    # Determine columns needed based on selected tab
    all_plot_columns = set()
    if selected_tab == 'tab-phin-phin-canyonb':
        all_plot_columns.update(['phin', 'phin_canb', 'depth', 'unixtime', 'divenumber'])
    elif selected_tab == 'tab-ph-drift':
        # pH drift tab only needs pH drift data
        pass
    elif selected_tab == 'tab-temp-psal':
        all_plot_columns.update(['tc', 'psal', 'depth', 'unixtime', 'divenumber'])
    elif selected_tab == 'tab-oxy-sigma':
        all_plot_columns.update(['doxy', 'sigma', 'depth', 'unixtime', 'divenumber'])
    elif selected_tab == 'tab-2':
        all_plot_columns.update(['vrse', 'vrse_std', 'vk', 'vk_std', 'ik', 'ib', 'depth', 'unixtime', 'divenumber'])
    elif selected_tab == 'tab-contour':
        # For contour plot, we need all columns to populate dropdown
        # But we'll optimize this by getting a smaller set for the dropdown
        all_plot_columns.update(['depth', 'unixtime', 'divenumber', 'phin', 'tc', 'psal', 'doxy', 'sigma', 'phin_canb'])
    elif selected_tab == 'tab-property':
        # For property plot, we need all columns to populate dropdown
        # But we'll optimize this by getting a smaller set for the dropdown
        all_plot_columns.update(['depth', 'unixtime', 'divenumber', 'phin', 'tc', 'psal', 'doxy', 'sigma', 'phin_canb'])
    
    # Convert to list and ensure we have the essential columns
    essential_columns = ['divenumber', 'depth', 'unixtime']
    plot_columns = list(all_plot_columns.union(set(essential_columns)))
    
    # Load data with error handling and column selection
    try:
        df_latest = get_dives_data(selected_mission, min_dive=min_dive, max_dive=max_dive, columns=plot_columns)
        df_map = get_map_data(selected_mission, min_dive=min_dive, max_dive=max_dive, columns=map_columns)
        df_ph_drift = get_ph_drift_data(selected_mission, depth=ph_drift_depth, columns=ph_drift_columns)
    except Exception as e:
        print(f"Error loading data: {e}")
        df_latest = pd.DataFrame()
        df_map = pd.DataFrame()
        df_ph_drift = pd.DataFrame()

    # Get unique unixtimes and corresponding datetimes for map colorbar
    if len(df_map) > 0 and 'unixtime' in df_map.columns:
        unix_vals_map = df_map['unixtime'].values
        datetimes_map = df_map['datetime'].dt.strftime('%m/%d %H:%M').values if 'datetime' in df_map.columns else df_map['unixtime'].values
        
        # Ensure unix_vals_map and datetimes_map have the same length
        min_len_map = min(len(unix_vals_map), len(datetimes_map))
        unix_vals_map = unix_vals_map[:min_len_map]
        datetimes_map = datetimes_map[:min_len_map]
    else:
        unix_vals_map = np.array([])
        datetimes_map = np.array([])
        min_len_map = 0

    # Choose evenly spaced ticks for map colorbar
    n_ticks_map = 5
    if min_len_map > n_ticks_map:
        idxs_map = np.linspace(0, min_len_map - 1, n_ticks_map, dtype=int)
        tickvals_map = unix_vals_map[idxs_map]
        ticktext_map = datetimes_map[idxs_map]
    else:
        tickvals_map = unix_vals_map
        ticktext_map = datetimes_map

    # Create map figure
    map_fig = go.Figure()
    
    # Only add map traces if we have valid data
    if len(df_map) > 0 and 'lat' in df_map.columns and 'lon' in df_map.columns:
        # Filter out any rows with NaN lat/lon
        valid_map_data = df_map.dropna(subset=['lat', 'lon'])
        
        if len(valid_map_data) > 1:
            # Reverse order so oldest data is plotted first (bottom) and newest on top
            # This makes recent data more visible
            track_data = valid_map_data.iloc[1:].iloc[::-1]  # Exclude most recent, then reverse
            
            # Add glider track (all points except the most recent, oldest to newest)
            map_fig.add_trace(go.Scattermap(
                lat=track_data['lat'],
                lon=track_data['lon'],
                mode='markers',
                name='Glider Track',
                hovertext=track_data['datetime'].dt.strftime('%m/%d %H:%M') if 'datetime' in track_data.columns else None,
                marker=dict(
                    size=10,
                    color=track_data['unixtime'] if 'unixtime' in track_data.columns else 'blue',
                    colorscale='Cividis',
                    cmin=valid_map_data['unixtime'].min() if 'unixtime' in valid_map_data.columns else None,
                    cmax=valid_map_data['unixtime'].max() if 'unixtime' in valid_map_data.columns else None,
                    showscale=True if len(tickvals_map) > 0 else False,
                    colorbar=dict(
                        len=0.6,
                        thickness=15,
                        tickvals=tickvals_map,
                        ticktext=ticktext_map,
                        x=0.5,
                        y=0.05,
                        xanchor='center',
                        yanchor='bottom',
                        orientation='h'
                    ) if len(tickvals_map) > 0 else None,
                ),
                showlegend=False,
            ))
        
        if len(valid_map_data) > 0:
            # Add most recent location marker
            map_fig.add_trace(go.Scattermap(
                lat=[valid_map_data['lat'].iloc[0]],
                lon=[valid_map_data['lon'].iloc[0]],
                mode='markers',
                name='Current Location',
                hovertext=[valid_map_data['datetime'].dt.strftime('%m/%d %H:%M').iloc[0]] if 'datetime' in valid_map_data.columns else None,
                marker=dict(
                    size=15,
                    symbol='airport',
                    color='red',
                    showscale=False,
                ),
                showlegend=False,
            ))
            
            # Center map on most recent position
            center_lat = valid_map_data['lat'].iloc[0]
            center_lon = valid_map_data['lon'].iloc[0]
        else:
            # Default center if no valid data
            center_lat = 42.0  # Default to Gulf of Maine region
            center_lon = -70.0
    else:
        # No data - use default center
        center_lat = 42.0
        center_lon = -70.0
        map_fig.add_annotation(
            text="No location data available for selected dives",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
    
    # Update map layout
    map_fig.update_layout(
        map_style="white-bg",
        map_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "source": [
                    "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
                ]
            },
        ],
        map=dict(
            center={'lat': center_lat, 'lon': center_lon},
            zoom=8,
        ),
        margin={"r":0,"t":20,"l":0,"b":0},
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1
        )
    )

    # For scatter plots: full filtered DataFrame
    if len(df_latest) == 0 or 'depth' not in df_latest.columns:
        # Create empty figures with informative message
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available for selected dives",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        scatter_fig_pHin = empty_fig
        scatter_fig_doxy = go.Figure()
        scatter_fig_temp = go.Figure()
        scatter_fig_salinity = go.Figure()
        scatter_fig_sigma = go.Figure()
        scatter_fig_ph_delta = go.Figure()
        scatter_fig_ph_drift = go.Figure()
        scatter_fig_vrs = go.Figure()
        scatter_fig_vrs_std = go.Figure()
        scatter_fig_vk = go.Figure()
        scatter_fig_vk_std = go.Figure()
        scatter_fig_ik = go.Figure()
        scatter_fig_ib = go.Figure()
    else:
        scatter_fig_pHin = make_depth_line_plot(
            df_latest,
            x="phin",
            title="pHinsitu[Total] vs. Depth"
        )
        scatter_fig_doxy = make_depth_line_plot(
            df_latest,
            x="doxy",
            title="Oxygen[µmol/kg] vs. Depth"
        )
        scatter_fig_temp = make_depth_line_plot(
            df_latest,
            x="tc",
            title="Temperature[°C] vs. Depth"
        )
        scatter_fig_salinity = make_depth_line_plot(
            df_latest,
            x="psal",
            title="Salinity[pss] vs. Depth"
        )
        scatter_fig_sigma = make_depth_line_plot(
            df_latest,
            x="sigma",
            title="Sigma[kg/m^3] vs. Depth"
        )
        scatter_fig_ph_delta = make_depth_line_plot(
            df_latest,
            x="ph-delta",
            title="ΔpHinsitu[Total] (pHin - pHinCanyonB)  vs. Depth"
        )
        scatter_fig_ph_drift = make_depth_scatter_plot(
            df_ph_drift,
            x="divenumber", y="ph-delta",
            title=f"ΔpHinsitu[Total] (pHin - pHinCanyonB) at {ph_drift_depth}m vs. Dive Number"
        )
        scatter_fig_vrs = make_depth_line_plot(
            df_latest,
            x="vrse",
            title="VRS[Volts] vs. Depth"
        )
        scatter_fig_vrs_std = make_depth_line_plot(
            df_latest,
            x="vrse_std",
            title="VRS_STD[Volts] vs. Depth"
        )
        scatter_fig_vk = make_depth_line_plot(
            df_latest,
            x="vk",
            title="VK[Volts] vs. Depth"
        )
        scatter_fig_vk_std = make_depth_line_plot(
            df_latest,
            x="vk_std",
            title="VK_STD[Volts] vs. Depth"
        )
        scatter_fig_ik = make_depth_line_plot(
            df_latest,
            x="ik",
            title="IK[nA] vs. Depth"
        )
        scatter_fig_ib = make_depth_line_plot(
            df_latest,
            x="ib",
            title="Ib[nA] vs. Depth"
        )
    # Contour plot and Property plot dropdown options
    if len(df_latest) > 0:
        dropdown_options = [
            {'label': col, 'value': col}
            for col in df_latest.columns
        ]
        # Get unique unixtimes and corresponding datetimes
        if 'unixtime' in df_latest.columns:
            unix_vals = df_latest['unixtime'].values
            datetimes = df_latest['datetime'].dt.strftime('%m/%d %H:%M').values if 'datetime' in df_latest.columns else df_latest['unixtime'].values

            # Ensure unix_vals and datetimes have the same length
            min_len = min(len(unix_vals), len(datetimes))
            unix_vals = unix_vals[:min_len]
            datetimes = datetimes[:min_len]

            # Choose evenly spaced indices
            n_ticks = 5
            if min_len > n_ticks:
                idxs = np.linspace(0, min_len - 1, n_ticks, dtype=int)
                tickvals = unix_vals[idxs]
                ticktext = datetimes[idxs]
            else:
                tickvals = unix_vals
                ticktext = datetimes
        else:
            tickvals = []
            ticktext = []
    else:
        dropdown_options = []
        tickvals = []
        ticktext = []
        
    # Contour plot figure
    if selected_tab == 'tab-contour' and contour_z and len(df_latest) > 0:
        # Check if requested column exists
        if contour_z in df_latest.columns:
            try:
                fig_contour = make_contour_plot(
                    df_latest, 
                    contour_z,
                    title=f'{contour_z} Contour Plot'
                )
            except Exception as e:
                print(f"Error creating contour plot: {e}")
                fig_contour = go.Figure()
                fig_contour.add_annotation(
                    text=f"Error creating contour plot: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
        else:
            fig_contour = go.Figure()
            fig_contour.add_annotation(
                text="Selected column not available in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
    else:
        fig_contour = go.Figure()
        
    # Property plot figure
    if selected_tab == 'tab-property' and property_x and property_y and len(df_latest) > 0:
        # Check if requested columns exist
        if property_x in df_latest.columns and property_y in df_latest.columns:
            try:
                fig_property = px.scatter(
                    df_latest, x=property_x, y=property_y,
                    title=f'{property_x} vs. {property_y}',
                    template='plotly_white',
                    color='unixtime' if 'unixtime' in df_latest.columns else None,
                )
                if 'depth' in property_y.lower() or property_y == 'depth':
                    fig_property.update_yaxes(autorange="reversed")
                fig_property.update_traces(marker=dict(size=6))
                if 'unixtime' in df_latest.columns and len(tickvals) > 0:
                    fig_property.update_layout(
                        coloraxis_colorbar=dict(
                            len=1,
                            tickvals=tickvals,
                            ticktext=ticktext,
                            title='datetime'
                        )
                    )
            except Exception as e:
                print(f"Error creating property plot: {e}")
                fig_property = go.Figure()
                fig_property.add_annotation(
                    text=f"Error creating plot: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
        else:
            fig_property = go.Figure()
            fig_property.add_annotation(
                text="Selected columns not available in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
    else:
        fig_property = go.Figure()
    if selected_tab == 'tab-phin-phin-canyonb':
        return (
            map_fig, scatter_fig_pHin, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            scatter_fig_ph_delta,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-ph-drift':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            scatter_fig_ph_drift,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-temp-psal':
        return (
            map_fig, dash.no_update, dash.no_update,
            scatter_fig_temp, scatter_fig_salinity, dash.no_update,
            dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-oxy-sigma':
        return (
            map_fig, dash.no_update, scatter_fig_doxy,
            dash.no_update, dash.no_update, scatter_fig_sigma,
            dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-2':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
            scatter_fig_vrs, scatter_fig_vrs_std, scatter_fig_vk, scatter_fig_vk_std, scatter_fig_ik, scatter_fig_ib,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-phin-phin-delta':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            scatter_fig_ph_drift,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-contour':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            fig_contour, dropdown_options,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-property':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            fig_property,
            dropdown_options, dropdown_options
        )
    else:
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,  # contour plot outputs
            go.Figure(),
            dropdown_options, dropdown_options
        )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=str(port), debug=True)