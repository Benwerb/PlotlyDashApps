# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import os

folder_path = r"\\sirocco\wwwroot\lobo\Data\GliderVizData"
files = [f for f in os.listdir(folder_path) if 'RT.txt' in f]


# Load and clean data
def load_latest_data(folder_path, selected_file=None):
    """Loads the latest RT.txt file, cleans it, and returns a DataFrame."""

    filename = os.path.join(folder_path, folder_path, selected_file if selected_file else files[-1])
    df = pd.read_csv(filename, delimiter="\t", skiprows=6)

    # Clean data
    df.replace(-1e10, pd.NA, inplace=True)
    df['Date'] = pd.to_datetime(df['mon/day/yr'], format='%m/%d/%Y')

    return df

df = load_latest_data(folder_path)

# files = [f for f in os.listdir(folder_path) if 'RT.txt' in f]

# # Need a better method for initializing the app that uses the function below
# filename = os.path.join(folder_path, files[-1])

# df = pd.read_csv(filename, delimiter="\t", skiprows=6)

# # Replace missing values
# df.replace(-1e10, pd.NA, inplace=True)

# # Convert date column to datetime
# df['Date'] = pd.to_datetime(df['mon/day/yr'], format='%m/%d/%Y')

# Get min/max values for filters
station_min, station_max = df["Station"].min(), df["Station"].max()
date_min, date_max = df["Date"].min(), df["Date"].max() 

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Predefined dropdown options
dropdown_options = [
    {"label": "Temperature (°C)", "value": "Temperature[°C]"},
    {"label": "Salinity (pss)", "value": "Salinity[pss]"},
    {"label": "Sigma Theta (kg/m³)", "value": "Sigma_theta[kg/m^3]"},
    {"label": "Oxygen (µmol/kg)", "value": "Oxygen[µmol/kg]"},
    {"label": "Oxygen Saturation (%)", "value": "OxygenSat[%]"},
    {"label": "Chlorophyll-a (µg/L)", "value": "Chl_a[µg/l]"},
    {"label": "pH in situ (Total)", "value": "pHinsitu[Total]"},
    {"label": "pH at 25°C, 1 atm (Total)", "value": "pH25C_1atm[Total]"},
    {"label": "Total Alkalinity (µmol/kg)", "value": "TALK_CANYONB[µmol/kg]"},
    {"label": "DIC (µmol/kg)", "value": "DIC_CANYONB[µmol/kg]"},
    {"label": "pCO₂ (µatm)", "value": "pCO2_CANYONB[µatm]"},
    {"label": "Saturation State AR", "value": "SAT_AR_CANYONB[]"},
    {"label": "Chlorophyll-a (corrected) (mg/m³)", "value": "Chl_a_corr[mg/m^3]"},
    {"label": "Backscatter (corrected) (1/m)", "value": "b_bp_corr[1/m]"},
    {"label": "POC (mmol/m³)", "value": "POC[mmol/m^3]"},
    {"label": "Downwelling Irradiance 380 nm (W/m²/nm)", "value": "DOWN_IRRAD380[W/m^2/nm]"},
    {"label": "Downwelling Irradiance 412 nm (W/m²/nm)", "value": "DOWN_IRRAD412[W/m^2/nm]"},
    {"label": "Downwelling Irradiance 490 nm (W/m²/nm)", "value": "DOWN_IRRAD490[W/m^2/nm]"},
    {"label": "Downwelling PAR (µmol Quanta/m²/sec)", "value": "DOWNWELL_PAR[µmol Quanta/m^2/sec]"},
    {"label": "VRS (Volts)", "value": "VRS[Volts]"},
    {"label": "VRS STD (Volts)", "value": "VRS_STD[Volts]"},
    {"label": "VK (Volts)", "value": "VK[Volts]"},
    {"label": "VK STD (Volts)", "value": "VK_STD[Volts]"},
    {"label": "IK (nA)", "value": "IK[nA]"},
    {"label": "Ib (nA)", "value": "Ib[nA]"},
    {"label": "Depth[m]", "value": "Depth[m]"},
    {"label": "Pressure[dbar]", "value": "Pressure[dbar]"},
]

# App layout
app.layout = dbc.Container([
    
    dbc.Row([html.H3('Glide App', className="text-primary text-center")]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Deployment:"),
            dcc.Dropdown(files, files[-1], id='Deployment-Dropdown', clearable=True)
        ], width=3)
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col(html.Div(id="file-output"))  # Output container
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Filter Method:"),
            dcc.RadioItems(
                id='filter-method',
                options=[
                    {'label': 'Filter by Profile Range', 'value': 'station'},
                    {'label': 'Filter by Date', 'value': 'date'},
                    {'label': 'Filter by Profile', 'value': 'profile'}
                ],
                value='station',
                inline=False
            )
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Station Range:"),
            dcc.RangeSlider(
                min=station_min, max=station_max, step=1,
                marks={int(i): str(int(i)) for i in range(int(station_min), int(station_max) + 1, 5)},
                value=[station_min, station_max],
                id='station-range-slider'
            ),
            html.Br(),
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=date_min,
                max_date_allowed=date_max,
                start_date=date_min,
                end_date=date_max
            ),
            html.Br(),
            html.Label("Profile"),
            dcc.Input(
                id='profile-number',
                type='number',
                min=station_min,
                max=station_max,
                placeholder=station_max
                )

        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Select X-axis:"),
            dcc.Dropdown(id='x-axis-dropdown', options=dropdown_options, value="Temperature[\u00b0C]", clearable=False)
        ], width=3),
        dbc.Col([
            html.Label("Select Y-axis:"),
            dcc.Dropdown(id='y-axis-dropdown', options=dropdown_options, value="Depth[m]", clearable=False)
        ], width=3)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='scatter-plot', style={'height': 'auto', 'width': 'auto'})
        ], width=6),
        dbc.Col([
            dcc.Graph(id='map-plot', style={'height': 'auto', 'width': 'auto'})
        ], width=6)
    ]),

], fluid=True)

# Dynamically load file
@callback(
    [Output('file-output', 'children'),
     Output('station-range-slider', 'min'),
     Output('station-range-slider', 'max'),
     Output('station-range-slider', 'value'),
     Output('date-picker-range', 'min_date_allowed'),
     Output('date-picker-range', 'max_date_allowed'),
     Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date')],
    Input('Deployment-Dropdown', 'value')
)
def update_file(selected_file):
    # # Construct the full file path
    # filename = os.path.join(folder_path, selected_file)
    # df = pd.read_csv(filename, delimiter="\t", skiprows=6)
    # # Replace missing values
    # df.replace(-1e10, pd.NA, inplace=True)
    # # Convert date column to datetime
    # df['Date'] = pd.to_datetime(df['mon/day/yr'], format='%m/%d/%Y')

    df = load_latest_data(folder_path,selected_file)

    # Get new min/max values for filters
    station_min, station_max = df["Station"].min(), df["Station"].max()
    date_min, date_max = df["Date"].min(), df["Date"].max()

    # Return updated values for filters
    return f"Loaded: {selected_file}", station_min, station_max, [station_min, station_max], date_min, date_max, date_min, date_max


@callback(
    [Output('station-range-slider', 'disabled'),
     Output('date-picker-range', 'disabled')],
    Input('filter-method', 'value')
)
def toggle_filters(selected_filter):
    return selected_filter != 'station', selected_filter != 'date'

@callback(
    Output('profile-number', 'disabled'),
    Input('filter-method', 'value')
)
def toggle_profile_input(selected_filter):
    return selected_filter != 'profile'  # Disable input unless 'profile' is selected


@callback(
    [Output('scatter-plot', 'figure'), Output('map-plot', 'figure')],
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('filter-method', 'value'),
     Input('station-range-slider', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('profile-number', 'value'),
     State('Deployment-Dropdown', 'value')]
)
def update_graph(x_column, y_column, filter_method, station_range, start_date, end_date, profile_number, selected_file):
    # filename = os.path.join(folder_path, selected_file)
    # df = pd.read_csv(filename, delimiter="\t", skiprows=6)
    # df.replace(-1e10, pd.NA, inplace=True)
    # df['Date'] = pd.to_datetime(df['mon/day/yr'], format='%m/%d/%Y')

    df = load_latest_data(folder_path,selected_file)

    # Apply filter based on the selected method
    if filter_method == 'station':  # Profile Range
        filtered_df = df[(df["Station"] >= station_range[0]) & (df["Station"] <= station_range[1])]
    elif filter_method == 'date':  # Date Range
        filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    else:  # Profile number
        filtered_df = df[df["Station"] == profile_number] if profile_number is not None else df

    # Scatter Plot
    scatter_fig = px.scatter(
        filtered_df, x=x_column, y=y_column,
        labels={x_column: x_column, y_column: y_column},
        title=f"{x_column} vs. {y_column}",
        template="plotly_white"
    )
    scatter_fig.update_yaxes(autorange="reversed")
    scatter_fig.update_layout(height=600, width=600)

    # Map Plot
    map_fig = px.scatter_map(
        filtered_df, lat="Lat [°N]", lon="Lon [°E]",
        hover_name="Station",
        map_style="light",
        zoom=6,
    )
    map_fig.update_layout(height=600, width=600)

    return scatter_fig, map_fig

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8050, debug=True)
