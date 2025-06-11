import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go

# Hardcode path to files and create list of all missions
folder_path = r"\\atlas.shore.mbari.org\ProjectLibrary\901805_Coastal_Biogeochemical_Sensing\WireWalker\MBARI\data"
file_name = "WW_Upcast_1Hz.txt"
file_path = os.path.join(folder_path, file_name)

# Load and clean data
def load_latest_data(file_path):
    """Loads the latest RT.txt file, cleans it, and returns a DataFrame."""
    df = pd.read_csv(file_path, delimiter=",")

    # Clean data
    df['Date'] = pd.to_datetime(df['mm/dd/yyyy'], format='%m/%d/%Y')
    df['Datetime'] = pd.to_datetime(df['mm/dd/yyyy'] + ' ' + df['HH:MM:SS'], format='%m/%d/%Y %H:%M:%S')
    
    return df

df = load_latest_data(file_path)

# Get min/max values for filters
station_min, station_max = df["Station"].min(), df["Station"].max()
date_min, date_max = df["Date"].min(), df["Date"].max() 

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.FLATLY ]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn

app.layout = dbc.Container([
    dbc.Row([
        # LEFT Column (Controls)
        dbc.Col([
            html.Div([
                html.H2('MBARI WireWalker', className='text-info text-start',
                        style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '20px'}),
                
                # Controls
                *[
                    dbc.Card([
                        dbc.CardBody([control])
                    ], className="mb-3") for control in [
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
                            html.Label("Station Range:"),
                            dcc.RangeSlider(
                                min=station_min,
                                max=station_max,
                                value=[station_max - 500, station_max],
                                id='station-range-slider',
                                marks={},  # <- removes marks
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'width': '90%', 'margin': 'auto'}),

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
                                value="pH"
                            )
                        ]),
                        html.Div([
                            html.Label("Select Y-axis:"),
                            dcc.Dropdown(
                                id='y-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=True,
                                value="Depth"
                            )
                        ]),
                        html.Div([
                            html.Label("Select Color-axis:"),
                            dcc.Dropdown(
                                id='color-axis-dropdown',
                                options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                                multi=False,
                                value="pH"
                            )
                        ]),
                        html.Div([
                            html.Label("Depth Range"),
                            dcc.RangeSlider(
                                id='depth-slider',
                                min=0, max=100, step=10,
                                marks={i: str(i) for i in range(0, 101, 10)},
                                value=[0, 100]
                            )
                        ]),
                    ]
                ],
            ], style={'backgroundColor': '#e0f7fa', 'padding': '10px', 'borderRadius': '10px'})
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
                'backgroundColor': '#e3f2fd',
                'padding': '10px',
                'borderRadius': '10px'
            })
        ], width=9)
    ])
], fluid=True, className='dashboard-container')

@callback(
    [Output('station-range-slider', 'disabled'),
     Output('date-picker-range', 'disabled')],
    Input('filter-method', 'value')
)
def toggle_filters(selected_filter):
    return selected_filter != 'station', selected_filter != 'date'

# @callback(
#     Output('profile-number', 'value'),
#     Output('profile-number','min'),
#     Output('profile-number','max'),
#     Input('Deployment-Dropdown', 'value')
# )
# def update_profile_number(selected_file):
#     df = load_latest_data(folder_path, selected_file)
    
#     if df.empty:
#         return None

#     station_max = df["Station"].max()
#     station_min = df["Station"].min()
#     return station_max, station_min, station_max  # Set to the max station of the new file

# def toggle_profile_input(selected_filter):
#     return selected_filter != 'profile'  # Disable input unless 'profile' is selected


@callback(
    [Output('map-plot','figure'),
     Output('scatter-plot-xy','figure'),
     Output('contour-plot', 'figure')],
    [Input('filter-method', 'value'),
     Input('station-range-slider', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('profile-number', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('color-axis-dropdown','value'),
     Input('depth-slider','value'),]
)

def update_graph(filter_method, station_range, start_date, end_date, profile_number, x_column, y_column, color_column, depth_range):

    df = load_latest_data(file_path)

    # Apply filter based on the selected method
    if filter_method == 'station':  # Profile Range
        filtered_df = df[(df["Station"] >= station_range[0]) & (df["Station"] <= station_range[1])]
    elif filter_method == 'date':  # Date Range
        filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    else:  # Profile number
        filtered_df = df[df["Station"] == profile_number] if profile_number is not None else df
    
    depth_min, depth_max = depth_range  # Unpack values
    
    filtered_df = filtered_df[(filtered_df['Depth'] > depth_min) & (filtered_df['Depth'] < depth_max)]

    # Create base figure
    map_fig = px.scatter_map(
        filtered_df, lat="Lat [°N]", lon="Lon [°E]",
        hover_name="Station",
        map_style="satellite",
        zoom=8,
        color='Station',
        labels={"Station": "Profile"}
    )

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
        return map_fig, empty_fig
    if not valid_y_columns:
        empty_fig = go.Figure()
        return map_fig, empty_fig

    # Iterate over valid x and y columns and add traces
    for i, x_col in enumerate(valid_x_columns):
        for j, y_col in enumerate(valid_y_columns):
            xaxis_name = "x" if i == 0 else f"x{i+1}"
            yaxis_name = "y" if j == 0 else f"y{j+1}"

            scatter_fig_xy.add_trace(go.Scatter(
            x=filtered_df[x_col],
            y=filtered_df[y_col],
            mode='markers',
            marker=dict(
                color=filtered_df['Station'],     # Use filtered_df, not df
                colorscale='Viridis',
                colorbar=dict(title='Station'),   # Update title if desired
                size=2,
                opacity=.8
            ),
            name=f"{x_col} vs {y_col}",           # Comma was missing here
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

    scatter_fig_xy.update_layout(layout, template="plotly_white")


    Contour_Plot = px.scatter(
        filtered_df,
        x='Datetime',
        y='Depth',
        color=color_column,
        color_continuous_scale='Viridis',  # or other scale like 'Plasma', 'Cividis'
        labels={'pH': 'pH (total scale)'},
        title='Depth vs Time Colored by pH'
    )

    Contour_Plot.update_layout(yaxis_autorange='reversed')  # Depth increases downward

    return map_fig, scatter_fig_xy, Contour_Plot

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port,debug=True,dev_tools_hot_reload=True)
