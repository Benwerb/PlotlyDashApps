import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objects as go

# Hardcode path to files and create list of all missions
folder_path = r"\\atlas.shore.mbari.org\ProjectLibrary\901805_Coastal_Biogeochemical_Sensing\WireWalker\MBARI\data"
file_name = "WW_Upcast_1Hz.txt"
file_path = os.path.join(folder_path, file_name)

# Load and clean data
def load_latest_data(file_path, downsample_factor):
    """Loads the latest RT.txt file, cleans it, and returns a DataFrame."""
    df = pd.read_csv(file_path, delimiter=",",skiprows=lambda i: i != 0 and i % downsample_factor != 0)
    
    # Clean data
        # Add some QC data

    # Make datetime
    df['Date'] = pd.to_datetime(df['mm/dd/yyyy'], format='%m/%d/%Y')
    df['Datetime'] = pd.to_datetime(df['mm/dd/yyyy'] + ' ' + df['HH:MM:SS'], format='%m/%d/%Y %H:%M:%S')
    
    return df

def filter_dataframe(df, filter_method, station_range, start_date, end_date, profile_number, depth_range):
    # Filter by station range, date range, or profile number
    if filter_method == 'station':
        df = df[(df['Station'] >= station_range[0]) & (df['Station'] <= station_range[1])]
    elif filter_method == 'date':
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif filter_method == 'profile' and profile_number is not None:
        df = df[df['Station'] == profile_number]
    
    # Filter by depth range
    df = df[(df['Depth'] >= depth_range[0]) & (df['Depth'] <= depth_range[1])]
    
    return df

downsample_init = 4
downsample_factor = downsample_init # initially set to 2 for first load, 10 for testing
df = load_latest_data(file_path,downsample_factor)

# Define variable-specific percentile limits
def get_clim(df, color_column):
    if color_column == 'ChlorophyllA':
        lower, upper = np.percentile(df[color_column].dropna(), [5, 99])
    else:
        lower, upper = np.percentile(df[color_column].dropna(), [1, 99])

    step = max(round((upper - lower) / 100, 3), 0.001)

    return lower, upper, step

# Get min/max values for filters
station_min, station_max = df["Station"].min(), df["Station"].max()
date_min, date_max = df["Date"].min(), df["Date"].max() 

# Get min/max to initialize clim adjust
cmin, cmax, cstep = get_clim(df, 'pH')

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn

app.layout = dbc.Container([
    dbc.Row([
        # LEFT Column (Controls)
        dbc.Col([
            html.Div([
                html.H2('MBARI WireWalker', className='text-primary text-start',
                        style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '20px'}),
                # Controls 
                *[
                    dbc.Card([
                        dbc.CardBody([control], className="text-light")
                    ], color="secondary", outline=True, className="mb-3") for control in [
                        html.Div([
                            dbc.ButtonGroup([
                                dbc.Button("Refresh Data", id="refresh-btn", color="primary", className="flex-fill"),
                                html.A(
                                    dbc.Button("WireWalker Info", color="info", className="flex-fill"),
                                    href="https://www.delmarocean.com/applications-blog/coastal-mooring",
                                    target="_blank"
                                )
                            ], className="w-100 d-flex"),
                            
                            # Hidden store that caches your data
                            dcc.Store(id="data-store",data=df.to_dict('records'),storage_type='memory'),
                            dcc.Store(id="data-store-filtered",data=[],storage_type='memory'),
                        ]),
                        html.Div([
                            html.Label("Select Filter Method:", className="text-primary mb-2"),
                            dcc.RadioItems(
                                id='filter-method',
                                options=[
                                    {'label': 'Filter by Profile Range', 'value': 'station'},
                                    {'label': 'Filter by Profile', 'value': 'profile'},
                                    {'label': 'Filter by Date', 'value': 'date'}
                                ],
                                value='station',
                                style={'color': 'white'}
                            )
                        ]),
                        html.Div([
                            html.Label("Station Range:", className="text-primary mb-2"),
                            dcc.RangeSlider(
                                min=station_min,
                                max=station_max,
                                value=[station_max - 1000, station_max],
                                id='station-range-slider',
                                marks={},  # no marks
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'width': '90%', 'margin': 'auto'}),

                        html.Div([
                            html.Label("Profile:", className="text-primary mb-2"),
                            dcc.Input(
                                id='profile-number',
                                type='number',
                                min=station_min,
                                max=station_max,
                                placeholder=station_max,
                                value=station_max
                            )
                        ]),
                        html.Div([
                            html.Label("Date Range:", className="text-primary mb-2"),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=date_min,
                                max_date_allowed=date_max,
                                start_date=date_min,
                                end_date=date_max
                            )
                        ]),
                        html.Div([
                            html.Label("Select Color-axis:", className="text-primary mb-2"),
                            dcc.Dropdown(
                                id='color-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=False,
                                value="pH"
                            )
                        ]),
                        html.Div([
                            html.Label("Select X-axis:", className="text-primary mb-2"),
                            dcc.Dropdown(
                                id='x-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=True,
                                value="pH"
                            )
                        ]),
                        html.Div([
                            html.Label("Select Y-axis:", className="text-primary mb-2"),
                            dcc.Dropdown(
                                id='y-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=True,
                                value="Depth"
                            )
                        ]),
                        html.Div([
                            html.Label("Depth Range", className="text-primary mb-2"),
                            dcc.RangeSlider(
                                id='depth-slider',
                                min=0, max=100, step=10,
                                marks={i: str(i) for i in range(0, 101, 10)},
                                value=[0, 100]
                            )
                        ]),
                        html.Div([
                            html.Label("Select Color Scale:", className="text-primary mb-2"),
                            dcc.Dropdown(
                                id='color-scale-dropdown',
                                options=[
                                {'label': 'Viridis', 'value': 'Viridis'},
                                {'label': 'Plasma', 'value': 'Plasma'},
                                {'label': 'Cividis', 'value': 'Cividis'},
                                {'label': 'Algae', 'value': 'algae'},
                                {'label': 'Thermal', 'value': 'thermal'},
                                {'label': 'Balance', 'value': 'balance'},
                                {'label': 'Red Blue', 'value': 'rdbu'},
                                {'label': 'Blue Red', 'value': 'bluered'},
                                ],
                                value='Viridis'
                            )
                        ]),
                        html.Div([
                            html.Label("Select sample frq (1 sample / n seconds): ", className="text-primary mb-2"), # 1 is 1 sample per second, 2 is 1 sample per 2 seconds, etc...
                            dcc.Input(
                                id='downsample-factor',
                                type='number',
                                min=1,
                                max=1000,
                                placeholder=downsample_init,
                                value=downsample_init
                            )
                        ]),
                        html.Div([
                            html.Label("Manual Color Scale Range:", className="text-primary mb-2"),
                            dcc.RangeSlider(
                                id='clim-range-slider',
                                min=cmin,  # Placeholder, will be updated dynamically
                                max=cmax,
                                value=[cmin, cmax],
                                step=cstep,
                                tooltip={"placement": "bottom", "always_visible": True},
                                marks=None,
                                allowCross=False
                            )
                        ]),
                       html.Div([
                            html.Label("Manual Color Scale", className="text-primary mb-2"),
                            dcc.Checklist(
                                id='enable-clim-slider',
                                options=[{'label': 'Enable', 'value': 'enabled'}],
                                value=[],
                                labelStyle={'display': 'block'},
                                style={'color': 'white'}
                            )
                        ])
                    ]
                ],
            ], style={}) # style={'backgroundColor': '#21211F', 'padding': '10px', 'borderRadius': '10px'}) #e0f7fa
        ], width=3),

        # RIGHT Column (Plots)
        dbc.Col([
            html.Div([
                # Big Contour Plot
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='contour-plot',
                            style={'height': '800px', 'width': '100%'}
                        )
                    ])
                ], className="mb-3"),

                # Big XY Plot
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='scatter-plot-xy',
                            style={'height': '800px', 'width': '100%'}
                        )
                    ])
                ], className="mb-3"),

                # Map plot - Full width square
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='map-plot',
                            style={'height': '800px', 'width': '100%'}
                        )
                    ])
                ])
            ], style={
                # 'backgroundColor': '#e3f2fd',
                'backgroundColor': '#1e1e1e',
                'color': 'white',
                'padding': '10px',
                'borderRadius': '10px'
            })
        ], width=9)
    ])
], fluid=True, className='dashboard-container')

@app.callback(
    Output("data-store", "data"),
    [Input("refresh-btn", "n_clicks"),
    Input("downsample-factor", "value")],
    prevent_initial_call=True
)
def refresh_data(n_clicks, downsample_factor):
    if downsample_factor is None:
        raise PreventUpdate
    
    new_df = load_latest_data(file_path, downsample_factor)
    return new_df.to_dict('records')

@callback(
    Output('data-store-filtered', 'data'),
    [Input('filter-method', 'value'),
    Input('station-range-slider', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('profile-number', 'value'),
    Input('depth-slider', 'value'),
    Input('data-store', 'data')]
)
def update_filtered_data(filter_method, station_range, start_date, end_date, profile_number, depth_range, raw_data):
    if not raw_data:
        return []

    df = pd.DataFrame(raw_data)
    df_filtered = filter_dataframe(df, filter_method, station_range, start_date, end_date, profile_number, depth_range)
    
    return df_filtered.to_dict('records')


@callback(
    Output('clim-range-slider', 'disabled'),
    Input('enable-clim-slider', 'value')
)
def toggle_slider(enabled):
    return 'enabled' not in enabled

@callback(
    [Output('clim-range-slider', 'min'),
     Output('clim-range-slider', 'max'),
     Output('clim-range-slider', 'step'),
     Output('clim-range-slider', 'value')],
    [Input('data-store', 'data'),
     Input('color-axis-dropdown', 'value')]
)
def update_clim_slider(data, color_column):
    columns = {color_column}
    df = pd.DataFrame(data)[list(columns)]
    cmin, cmax, cstep = get_clim(df, color_column)
    return cmin, cmax, cstep, [cmin, cmax]

@callback(
    [Output('station-range-slider', 'disabled'),
     Output('date-picker-range', 'disabled')],
    Input('filter-method', 'value')
)
def toggle_filters(selected_filter):
    return selected_filter != 'station', selected_filter != 'date'

@callback(
    Output('contour-plot', 'figure'),
    [Input('data-store-filtered','data'),
     Input('color-axis-dropdown','value'),
     Input('color-scale-dropdown','value'),
     Input('clim-range-slider', 'value'),
     Input('enable-clim-slider', 'value')]
)

def update_Contour(data, color_column, color_scale, clims, enable_clim_slider):
    
    # columns = {color_column, "Datetime", "Depth", "Station"}
    # df = pd.DataFrame(data)[list(columns)]
    df = pd.DataFrame(data) 

    # Use manual slider values only if the checkbox is enabled
    if 'enabled' in enable_clim_slider and enable_clim_slider is not None:
        cmin, cmax = clims
    else:
        cmin, cmax, cstep = get_clim(df, color_column)

    Contour_Plot = px.scatter(
        df,
        x='Datetime',
        y='Depth',
        color=color_column,
        color_continuous_scale=color_scale,  # or other scale like 'Plasma', 'Cividis', 'Viridis'
        range_color=[cmin, cmax],
        labels={color_column},
        title=f'Depth vs Time Colored by {color_column}'
    )
    
   # Apply color limits and reverse depth axis
    Contour_Plot.update_layout(
        yaxis_autorange='reversed',
        coloraxis_colorbar=dict(title=color_column),
    )

    Contour_Plot.update_traces(marker=dict(cmin=cmin, cmax=cmax))
    Contour_Plot.update_layout(template="plotly_dark")
    

    return Contour_Plot

@callback(
     Output('scatter-plot-xy','figure'),
    [Input('data-store-filtered','data'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('color-scale-dropdown','value')]
)

def update_graph(data, x_column, y_column, color_scale):

    # df = load_latest_data(file_path, downsample_factor)

    columns = {x_column, y_column, "Datetime", "Depth", "Station"}
    df = pd.DataFrame(data)[list(columns)]
    
    scatter_fig_xy = go.Figure()

    # Ensure x_column and y_column are always lists
    if isinstance(x_column, str):
        x_columns = [x_column]  # Convert single selection to list
    else:
        x_columns = x_column  # Already a list

    if isinstance(y_column, str):
        y_columns = [y_column]  # Convert single selection to list
    else:
        y_columns = y_column  # Already a list

    # Filter out invalid columns
    valid_x_columns = [x for x in x_columns if x in df.columns]
    valid_y_columns = [y for y in y_columns if y in df.columns]

    if not valid_x_columns:
        empty_fig = go.Figure()
        return empty_fig
    if not valid_y_columns:
        empty_fig = go.Figure()
        return empty_fig

    # Iterate over valid x and y columns and add traces
    for i, x_col in enumerate(valid_x_columns):
        for j, y_col in enumerate(valid_y_columns):
            xaxis_name = "x" if i == 0 else f"x{i+1}"
            yaxis_name = "y" if j == 0 else f"y{j+1}"

            scatter_fig_xy.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(
                color=df['Station'],
                colorscale=color_scale,
                colorbar=dict(title='Station'),
                size=2,
                opacity=.8
            ),
            name=f"{x_col} vs {y_col}",
            xaxis=xaxis_name,
            yaxis=yaxis_name
        ))


    # Define layout with multiple x- and y-axes
    layout = {
        "title": f"{', '.join(valid_x_columns)} vs. {', '.join(valid_y_columns)}",
        "xaxis": {"title": valid_x_columns[0]},  # Primary x-axis
        "yaxis": {"title": valid_y_columns[0], "autorange": "reversed"},  # Primary y-axis
    }

    # Add additional x-axes dynamically
    for i, x_col in enumerate(valid_x_columns[1:], start=2):
        layout[f"xaxis{i}"] = {
            "title": x_col,
            "anchor": "free",
            "overlaying": "x",
            "side": "bottom",
            "position": i * 0.1,  # Offset each x-axis
            "showgrid": False,
            "tickmode": "sync",
        }

    # Add additional y-axes dynamically with proper spacing
    for j, y_col in enumerate(valid_y_columns[1:], start=2):
        layout[f"yaxis{j}"] = {
            "title": y_col,
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "position": 1 - (j - 1) * 0.1,  # Move each additional y-axis further right
        }

    scatter_fig_xy.update_layout(layout, template="plotly_dark")

    return scatter_fig_xy

@callback(
    Output('map-plot','figure'),
    Input('data-store-filtered','data')
)

def update_map(data):

    # df = load_latest_data(file_path, downsample_factor)

    columns = {"Datetime", "Depth", "Station", "Lat [°N]", "Lon [°E]"}
    df = pd.DataFrame(data)[list(columns)].iloc[[0]]  # All lat/lon are the same in this file

    # Create base figure
    map_fig = px.scatter_map(
        df, lat="Lat [°N]", lon="Lon [°E]",
        hover_name="Station",
        map_style="satellite",
        zoom=8,
        # color='Station',
        # labels={"Station": "Profile"}
    )
    map_fig.update_layout(template="plotly_dark")
    return map_fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)