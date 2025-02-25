import dash_bootstrap_components as dbc
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import os

# Hardcode path to files and create list
folder_path = r"\\sirocco\wwwroot\lobo\Data\GliderVizData"
# folder_path = r"https://github.com/Benwerb/PlotlyDashApps/tree/3401150177166a2a0234c9339782415f27aa7a5b/TestApp"
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

# Layout with dbc.Cards
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H3('Glide App', className="text-primary text-center"))),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Select Deployment:"),
                dcc.Dropdown(files, files[-1], id='Deployment-Dropdown', clearable=True),
                html.Div(id="file-output")  # Output container
            ])
        ]), width=4)
    ], className="mb-3"),

    # dbc.Row([
    #     dbc.Col(
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.Div(id="file-output")  # Output container
    #             ])
    #         ])
    #     )
    # ]),

    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Select Filter Method:"),
                dcc.RadioItems(
                    id='filter-method',
                    options=[
                        {'label': 'Filter by Profile', 'value': 'profile'},
                        {'label': 'Filter by Profile Range', 'value': 'station'},
                        {'label': 'Filter by Date', 'value': 'date'}
                    ],
                    value='profile',
                    inline=False
                )
            ])
        ]), width=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Data Quality:"),
                dcc.RadioItems(
                    id='data-quality',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Good', 'value': 'good'},
                        {'label': 'Good + Questionable', 'value': 'good_questionable'}
                    ],
                    value='all',
                    inline=False
                )
            ])
        ]), width=4)
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Profile:"),
                dcc.Input(
                    id='profile-number',
                    type='number',
                    min=station_min,
                    max=station_max,
                    placeholder=station_max,
                    value=station_max
                )
            ])
        ]), width=8)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Station Range:"),
                dcc.RangeSlider(
                    min=station_min, max=station_max, step=1,
                    marks={int(i): str(int(i)) for i in range(int(station_min), int(station_max) + 1, 5)},
                    value=[station_min, station_max],
                    id='station-range-slider'
                )
            ])
        ]), width=8)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=date_min,
                    max_date_allowed=date_max,
                    start_date=date_min,
                    end_date=date_max
                )
            ])
        ]), width=8)
    ], className="mb-3"),
    
    
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='map-plot', style={'height': '500px'})])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-pH', style={'height': '500px'})])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-chla', style={'height': '500px'})])]), width=4),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-Temperature', style={'height': '500px'})])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-Salinity', style={'height': '500px'})])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-Doxy', style={'height': '500px'})])]), width=4),
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Select X-axis:"),
                dcc.Dropdown(id='x-axis-dropdown', options=dropdown_options, value="pH25C_1atm[Total]", clearable=False)
            ])
        ]), width=4),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Select Y-axis:"),
                dcc.Dropdown(id='y-axis-dropdown', options=dropdown_options, value="Depth[m]", clearable=False)
            ])
        ]), width=4)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-xy', style={'height': '1500px', 'width': '1500px'})])]), width=12),
    ]),

], fluid=True)





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

    # Return updated values for filters
    return station_min, station_max, [station_min, station_max], date_min, date_max, date_min, date_max


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
    [Output('map-plot','figure'),
     Output('scatter-plot-pH','figure'),
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
     State('Deployment-Dropdown', 'value')]
)

def update_graph(filter_method, station_range, start_date, end_date, profile_number, data_quality, x_column, y_column, selected_file):

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


    # Map Plot
    map_fig = px.scatter_map(
        filtered_df, lat="Lat [°N]", lon="Lon [°E]",
        hover_name="Station",
        map_style="satellite",
        zoom=8,
        color='Station'
        # labels={"Station": "Profile"}
    )
    # map_fig.update_layout(height=500, width=500)

    # Scatter Plot pH25 - CanB
    scatter_fig_pH25 = px.scatter(
        filtered_df, x="pH25C_1atm[Total]", y="Depth[m]",
        labels={"pH25C_1atm[Total]", "Depth[m]", "Profile"},
        title=f"pH25C_1atm[Total] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_pH25.update_yaxes(autorange="reversed")
    # scatter_fig.update_layout(height=1000, width=1000)


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

    scatter_fig_xy = px.scatter(
        filtered_df, x=x_column, y=y_column,
        labels={x_column: x_column, y_column: y_column, "Station": "Profile"},
        title=f"{x_column} vs. {y_column}",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_xy.update_yaxes(autorange="reversed")

    return map_fig, scatter_fig_pH25, scatter_fig_Chla, scatter_fig_Temperature, scatter_fig_Salinity, scatter_fig_Doxy, scatter_fig_xy

# @callback(
#     [Output('scatter-plot', 'figure'), 
#      Output('map-plot', 'figure'),
#      Output('contour-plot', 'figure')],
#     [Input('x-axis-dropdown', 'value'),
#      Input('y-axis-dropdown', 'value'),
#      Input('filter-method', 'value'),
#      Input('station-range-slider', 'value'),
#      Input('date-picker-range', 'start_date'),
#      Input('date-picker-range', 'end_date'),
#      Input('profile-number', 'value'),
#      Input('data-quality', 'value'),
#      State('Deployment-Dropdown', 'value')]
# )
# def update_graph(x_column, y_column, filter_method, station_range, start_date, end_date, profile_number, data_quality, selected_file):

#     df = load_latest_data(folder_path, selected_file)

#     # Identify QF columns (columns immediately following measured values)
#     qf_columns = [col for col in df.columns if 'QF' in col]

#     # Apply Data Quality filter
#     if data_quality == "good":
#         df = df[(df[qf_columns] == 0).all(axis=1)]
#     elif data_quality == "good_questionable":
#         df = df[(df[qf_columns].isin([0, 4])).all(axis=1)]

#     # Apply filter based on the selected method
#     if filter_method == 'station':  # Profile Range
#         filtered_df = df[(df["Station"] >= station_range[0]) & (df["Station"] <= station_range[1])]
#     elif filter_method == 'date':  # Date Range
#         filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
#     else:  # Profile number
#         filtered_df = df[df["Station"] == profile_number] if profile_number is not None else df

#     # Scatter Plot
#     scatter_fig = px.scatter(
#         filtered_df, x=x_column, y=y_column,
#         labels={x_column: x_column, y_column: y_column, "Station": "Profile"},
#         title=f"{x_column} vs. {y_column}",
#         template="plotly_white",
#         color='Station',
#     )
#     scatter_fig.update_yaxes(autorange="reversed")
#     scatter_fig.update_layout(height=1000, width=1000)

#     # Map Plot
#     map_fig = px.scatter_map(
#         filtered_df, lat="Lat [°N]", lon="Lon [°E]",
#         hover_name="Station",
#         map_style="satellite",
#         zoom=10,
#         color='Station',
#         labels={"Station": "Profile"}
#     )
#     map_fig.update_layout(height=1000, width=1000)

#     # Contour Figure
#     contour_fig = px.density_contour(
#         filtered_df, x="Date", y=y_column, z=x_column,
#         labels={"Date": "Date", y_column: y_column, x_column: x_column},
#         title=f"Contour Plot: {x_column} vs {y_column} over Time",
#         template="plotly_white",
#     )
#     contour_fig.update_yaxes(autorange="reversed")
#     contour_fig.update_layout(height=1000, width=1000)

#     # Add depth filter and fix the contour plot

#     return scatter_fig, map_fig, contour_fig

if __name__ == '__main__':
    app.run(debug=True)