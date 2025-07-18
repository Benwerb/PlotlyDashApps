import dash_bootstrap_components as dbc
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re

# Hardcode path to files and create list of all missions
folder_path = "https://www3.mbari.org/lobo/Data/GliderVizData/"

response = requests.get(folder_path)
soup = BeautifulSoup(response.text, 'html.parser')

# Get clean filenames without path or extension
# Get just the filename with extension (no path)
files = [
    os.path.basename(a['href'])
    for a in soup.find_all('a', href=True)
    if 'RT.txt' in a['href']
]

# Load and clean data
def load_latest_data(folder_path, selected_file=None):
    """Loads the latest RT.txt file, cleans it, and returns a DataFrame."""
    filename = selected_file if selected_file else files[-1]
    file_url = folder_path + filename
    file_response = requests.get(file_url)
    file_content = StringIO(file_response.text)
    df = pd.read_csv(file_content, delimiter="\t", skiprows=6)

    # Clean data
    df.columns = df.columns.str.replace('Â', '') # Issue when importing from html
    df.replace(-1e10, pd.NA, inplace=True)
    df.replace(-999, pd.NA, inplace=True)
    df['Date'] = pd.to_datetime(df['mon/day/yr'], format='%m/%d/%Y')
    df['Datetime'] = pd.to_datetime(df['mon/day/yr'] + ' ' + df['hh:mm'], format='%m/%d/%Y %H:%M')
    # Only calculate pHin_Canb_Delta if 'PHIN_CANYONB[Total]' exists
    if 'PHIN_CANYONB[Total]' in df.columns:
        df['pHin_Canb_Delta'] = df['pHinsitu[Total]'] - df['PHIN_CANYONB[Total]']
    else:
        df['pHin_Canb_Delta'] = pd.NA  # Optional: Add the column as all-NA if needed     
    return df

df = load_latest_data(folder_path)

# Get min/max values for filters
station_min, station_max = df["Station"].min(), df["Station"].max()
date_min, date_max = df["Date"].min(), df["Date"].max() 

url = 'https://ocean.weather.gov/gulf_stream_latest.txt'
def getGulfStreamCoords(url):
    print('running')
    # Step 1: Retrieve the data

    response = requests.get(url)
    data = response.text

    # Step 2: Locate the coordinates section
    match = re.search(r'RMKS/1\. GULF STREAM NORTH WALL DATA FOR.*?:\s*(.*?)(?:RMKS/|$)', data, re.S)
    if not match:
        raise ValueError('Could not locate the Gulf Stream coordinates section.')
    coordinates_text = match.group(1).strip()

    # Step 3: Extract coordinate pairs (format like 25.5N80.1W)
    coordinate_strings = re.findall(r'(\d+\.\d+N\d+\.\d+W)', coordinates_text)

    # Step 4: Convert to decimal degrees and store in lists
    latitudes = []
    longitudes = []

    for pair in coordinate_strings:
        lat_str, lon_str = re.match(r'(\d+\.\d+)N(\d+\.\d+)W', pair).groups()
        latitudes.append(float(lat_str))
        longitudes.append(-float(lon_str))  # W longitude is negative

    # Step 5: Create a pandas DataFrame
    gulfstreamcoords = pd.DataFrame({
        'Lat': latitudes,
        'Lon': longitudes
    })
    return gulfstreamcoords
gulfstreamcoords = getGulfStreamCoords(url)

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY ]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn

app.layout = dbc.Container([
    dbc.Row([
        # LEFT Column (Controls)
        dbc.Col([
            html.Div([
                html.H2('Glide App', className='text-info text-start',
                        style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '20px'}),
                
                # Controls
                *[
                    dbc.Card([
                        dbc.CardBody([control])
                    ], className="mb-3") for control in [
                        html.Div([
                            html.Label("Select Deployment:"),
                            dcc.Dropdown(files, files[-1], id='Deployment-Dropdown', clearable=True),
                            html.Div(id="file-output")
                        ]),
                        html.Div([
                            html.Label("Select Filter Method:"),
                            dcc.RadioItems(
                                id='filter-method',
                                options=[
                                    {'label': 'Filter by Profile Range', 'value': 'station'},
                                    {'label': 'Filter by Profile', 'value': 'profile'},
                                    {'label': 'Filter by Date', 'value': 'date'}
                                ],
                                value='station'
                            )
                        ]),
                        html.Div([
                            html.Label("Data Quality:"),
                            dcc.RadioItems(
                                id='data-quality',
                                options=[
                                    {'label': 'All', 'value': 'all'},
                                    {'label': 'Good', 'value': 'good'},
                                    {'label': 'Good + Questionable', 'value': 'good_questionable'}
                                ],
                                value='all'
                            )
                        ]),
                        html.Div([
                            html.Label("Cast Direction:"),
                            dcc.RadioItems(
                                id='cast-direction',
                                options=[
                                    {'label': 'Raw profile', 'value': 'Both'},
                                    {'label': 'Upcast only', 'value': 'Upcast'},
                                    {'label': 'Downcast only', 'value': 'Downcast'}
                                ],
                                value='Both'
                            )
                        ]),
                        html.Div([
                            html.Label("Station Range:"),
                            dcc.RangeSlider(
                                min=station_min,
                                max=station_max,
                                value=[station_min, station_max],
                                id='station-range-slider',
                                marks = {},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ]),
                        html.Div([
                            html.Label("Profile:"),
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
                            html.Label("Date Range:"),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=date_min,
                                max_date_allowed=date_max,
                                start_date=date_min,
                                end_date=date_max
                            )
                        ]),
                        html.Div([
                            html.Label("Select X-axis:"),
                            dcc.Dropdown(
                                id='x-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=True,
                                value="pH25C_1atm[Total]"
                            )
                        ]),
                        html.Div([
                            html.Label("Select Y-axis:"),
                            dcc.Dropdown(
                                id='y-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=True,
                                value="Depth[m]"
                            )
                        ]),
                        html.Div([
                            html.Label("Depth Range"),
                            dcc.RangeSlider(
                                id='depth-slider',
                                min=0, max=1000, step=50,
                                marks={i: str(i) for i in range(0, 1001, 200)},
                                value=[0, 1000]
                            )
                        ]),
                        html.Div([
                            html.Label("Map Options:"),
                            dcc.Checklist(
                                id='overlay-gulf-stream',
                                options=[{'label': 'Overlay Gulf Stream Bounds', 'value': 'gulf_stream'}],
                                value=[],
                                labelStyle={'display': 'block'}
                            )
                        ])
                    ]
                ],
            ], style={'backgroundColor': '#e0f7fa', 'padding': '10px', 'borderRadius': '10px'})
        ], width=3),

        # RIGHT Column (Plots)
        dbc.Col([
            html.Div([
                # Big XY Plot
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='scatter-plot-xy',
                            style={'height': '800px', 'width': '800px', 'margin': '0 auto'}
                        )
                    ])
                ], className="mb-3"),

                # First row of 3 square plots
                dbc.Row([
                    dbc.Col(dcc.Graph(id='scatter-plot-pH', style={'height': '400px', 'width': '400px'}), width=4),
                    dbc.Col(dcc.Graph(id='scatter-plot-pH-delta', style={'height': '400px', 'width': '400px'}), width=4),
                    dbc.Col(dcc.Graph(id='scatter-plot-chla', style={'height': '400px', 'width': '400px'}), width=4),
                ], className="mb-3"),

                # Second row of 3 square plots
                dbc.Row([
                    dbc.Col(dcc.Graph(id='scatter-plot-Temperature', style={'height': '400px', 'width': '400px'}), width=4),
                    dbc.Col(dcc.Graph(id='scatter-plot-Salinity', style={'height': '400px', 'width': '400px'}), width=4),
                    dbc.Col(dcc.Graph(id='scatter-plot-Doxy', style={'height': '400px', 'width': '400px'}), width=4),
                ], className="mb-3"),

                # Map plot - Full width square
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='map-plot',
                            style={'height': '800px', 'width': '800px', 'margin': '0 auto'}
                        )
                    ])
                ])
            ], style={'backgroundColor': '#e3f2fd', 'padding': '10px', 'borderRadius': '10px'})
        ], width=9)
    ])
], fluid=True, className='dashboard-container')

# Dynamically load file
@callback(
    [Output('station-range-slider', 'min'),
     Output('station-range-slider', 'max'),
     Output('station-range-slider', 'value'),
     Output('date-picker-range', 'min_date_allowed'),
     Output('date-picker-range', 'max_date_allowed'),
     Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date')],
    Input('Deployment-Dropdown', 'value')
)
def update_file(selected_file):

    df = load_latest_data(folder_path,selected_file)

    # Get new min/max values for filters
    station_min, station_max = df["Station"].min(), df["Station"].max()
    date_min, date_max = df["Date"].min(), df["Date"].max()
    # new_marks = {int(i): str(int(i)) for i in range(int(station_min), int(station_max) + 1, 5)}
    # Return updated values for filters
    return station_min, station_max, [max(station_min, station_max - 10), station_max], date_min, date_max, date_min, date_max


@callback(
    [Output('station-range-slider', 'disabled'),
     Output('date-picker-range', 'disabled')],
    Input('filter-method', 'value')
)
def toggle_filters(selected_filter):
    return selected_filter != 'station', selected_filter != 'date'

@callback(
    Output('profile-number', 'value'),
    Output('profile-number','min'),
    Output('profile-number','max'),
    Input('Deployment-Dropdown', 'value')
)
def update_profile_number(selected_file):
    df = load_latest_data(folder_path, selected_file)
    
    if df.empty:
        return None

    station_max = df["Station"].max()
    station_min = df["Station"].min()
    return station_max, station_min, station_max  # Set to the max station of the new file

def toggle_profile_input(selected_filter):
    return selected_filter != 'profile'  # Disable input unless 'profile' is selected


@callback(
    [Output('map-plot','figure'),
     Output('scatter-plot-pH','figure'),
     Output('scatter-plot-pH-delta','figure'),
     Output('scatter-plot-chla','figure'),
     Output('scatter-plot-Temperature','figure'),
     Output('scatter-plot-Salinity','figure'),
     Output('scatter-plot-Doxy','figure'),
     Output('scatter-plot-xy','figure')],
    [Input('filter-method', 'value'),
     Input('station-range-slider', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('profile-number', 'value'),
     Input('data-quality', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('depth-slider','value'),
     Input('overlay-gulf-stream','value'),
     Input('cast-direction','value'),
     State('Deployment-Dropdown', 'value')]
)

def update_graph(filter_method, station_range, start_date, end_date, profile_number, data_quality, x_column, y_column, depth_range, gulf_stream_check, cast_direction, selected_file):

    df = load_latest_data(folder_path, selected_file)

    # Identify QF columns (columns immediately following measured values)
    qf_columns = [col for col in df.columns if 'QF' in col]

    # Apply Data Quality filter
    if data_quality == "good":
        df = df[(df[qf_columns] == 0).all(axis=1)]
    elif data_quality == "good_questionable":
        df = df[(df[qf_columns].isin([0, 4])).all(axis=1)]

    # Apply filter based on the selected method
    if filter_method == 'station':  # Profile Range
        filtered_df = df[(df["Station"] >= station_range[0]) & (df["Station"] <= station_range[1])]
    elif filter_method == 'date':  # Date Range
        filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    else:  # Profile number
        filtered_df = df[df["Station"] == profile_number] if profile_number is not None else df

    # Apply filter based on cast direction
    if cast_direction == 'Upcast':
        filtered_df = filtered_df.copy()
        filtered_df["depth_diff"] = filtered_df["Depth[m]"].diff()
        filtered_df = filtered_df[filtered_df["depth_diff"] < 0]
        filtered_df.drop(columns="depth_diff", inplace=True)

    elif cast_direction == 'Downcast':
        filtered_df = filtered_df.copy()
        filtered_df["depth_diff"] = filtered_df["Depth[m]"].diff()
        filtered_df = filtered_df[filtered_df["depth_diff"] > 0]
        filtered_df.drop(columns="depth_diff", inplace=True)
    # If 'Both', no further filtering needed

    
    depth_min, depth_max = depth_range  # Unpack values
    filtered_df = filtered_df[(filtered_df['Depth[m]'] > depth_min) & (filtered_df['Depth[m]'] < depth_max)]

    # Create base figure
    map_fig = px.scatter_map(
        filtered_df, lat="Lat [°N]", lon="Lon [°E]",
        hover_name="Station",
        map_style="satellite",
        zoom=8,
        color='Station',
        labels={"Station": "Profile"}
    )

    # If Gulf Stream overlay is checked, add trace
    if 'gulf_stream' in gulf_stream_check:
        map_fig.add_trace(go.Scattermap(
            lat=gulfstreamcoords['Lat'],
            lon=gulfstreamcoords['Lon'],
            mode='lines+markers',
            name='Gulf Stream',
            marker=dict(size=6, color='deepskyblue'),
            line=dict(width=2, color='deepskyblue')
        ))

    # Scatter Plot pH25atm
    scatter_fig_pHin = px.scatter(
        filtered_df, x="pHinsitu[Total]", y="Depth[m]",
        labels={"pHinsitu[Total]", "Depth[m]", "Profile"},
        title=f"pHinsitu[Total] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_pHin.update_yaxes(autorange="reversed")
    # scatter_fig.update_layout(height=1000, width=1000)

    scatter_fig_pHin_delta = px.scatter(
        filtered_df, x="pHin_Canb_Delta", y="Depth[m]",
        labels={"pHin - pHin_Canb", "Depth[m]", "Profile"},
        title=f"pHin - pHin_Canb vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_pHin_delta.update_yaxes(autorange="reversed")
    x_max = max(abs(filtered_df["pHin_Canb_Delta"].max()), abs(filtered_df["pHin_Canb_Delta"].min()))
    scatter_fig_pHin_delta.update_xaxes(range=[-x_max, x_max])

    scatter_fig_Chla = px.scatter(
        filtered_df, x="Chl_a[mg/m^3]", y="Depth[m]",
        labels={"Chl_a[mg/m^3]", "Depth[m]", "Profile"},
        title=f"Chl_a[mg/m^3] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_Chla.update_yaxes(autorange="reversed")


    scatter_fig_Temperature = px.scatter(
        filtered_df, x="Temperature[°C]", y="Depth[m]",
        labels={"Temperature[°C]", "Depth[m]", "Profile"},
        title=f"Temperature[°C] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_Temperature.update_yaxes(autorange="reversed")

    scatter_fig_Salinity = px.scatter(
        filtered_df, x="Salinity[pss]", y="Depth[m]",
        labels={"Salinity[pss]", "Depth[m]", "Profile"},
        title=f"Salinity[pss] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_Salinity.update_yaxes(autorange="reversed")

    scatter_fig_Doxy = px.scatter(
        filtered_df, x="Oxygen[µmol/kg]", y="Depth[m]",
        labels={"Oxygen[µmol/kg]", "Depth[m]", "Profile"},
        title=f"Oxygen[µmol/kg] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_Doxy.update_yaxes(autorange="reversed")

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
    valid_x_columns = [x for x in x_columns if x in filtered_df.columns]
    valid_y_columns = [y for y in y_columns if y in filtered_df.columns]

    if not valid_x_columns:
        empty_fig = go.Figure()
        return map_fig, scatter_fig_pHin, scatter_fig_pHin_delta, scatter_fig_Chla, scatter_fig_Temperature, scatter_fig_Salinity, scatter_fig_Doxy, empty_fig
    if not valid_y_columns:
        empty_fig = go.Figure()
        return map_fig, scatter_fig_pHin, scatter_fig_pHin_delta, scatter_fig_Chla, scatter_fig_Temperature, scatter_fig_Salinity, scatter_fig_Doxy, empty_fig

    # Iterate over valid x and y columns and add traces
    for i, x_col in enumerate(valid_x_columns):
        for j, y_col in enumerate(valid_y_columns):
            xaxis_name = "x" if i == 0 else f"x{i+1}"
            yaxis_name = "y" if j == 0 else f"y{j+1}"

            scatter_fig_xy.add_trace(go.Scatter(
                x=filtered_df[x_col],
                y=filtered_df[y_col],
                mode='markers',
                name=f"{x_col} vs {y_col}",
                xaxis=xaxis_name,
                yaxis=yaxis_name,
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

    scatter_fig_xy.update_layout(layout, template="plotly_white")

    return map_fig, scatter_fig_pHin, scatter_fig_pHin_delta, scatter_fig_Chla, scatter_fig_Temperature, scatter_fig_Salinity, scatter_fig_Doxy, scatter_fig_xy

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port) # Build command: gunicorn GlideApp:server
