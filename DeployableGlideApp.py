import dash_bootstrap_components as dbc
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go
from sqlalchemy import create_engine 
from sqlalchemy import inspect
import psycopg2

db_url = "postgresql://spraydabase_user:8pgy9Sba79ETgds8QcaycQj0U6uIhhwQ@dpg-cur2o7lds78s7384jthg-a.oregon-postgres.render.com/spraydabase"
engine = create_engine(db_url)
inspector = inspect(engine)
files = inspector.get_table_names()
latest_table = f'"{files[-1]}"'  # Add double quotes for case-sensitive or numeric table names
query = f'SELECT * FROM {latest_table}'
df = pd.read_sql(query, engine) #double quotes needed for case-sensitive or numeric names

# Get min/max values for filters
station_min, station_max = df["Station"].min(), df["Station"].max()
date_min, date_max = df["Date"].min(), df["Date"].max() 


# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn

# Predefined dropdown options
dropdown_options = [
  {"label": "Profile", "value": "Station"},
  {"label": "Date", "value": "Date"},
  {"label": "Datetime", "value": "Datetime"},
  {"label": "hh:mm", "value": "hh:mm"},
  {"label": "Lon [°E]", "value": "Lon [°E]"},
  {"label": "Lat [°N]", "value": "Lat [°N]"},
  {"label": "Pressure[dbar]", "value": "Pressure[dbar]"},
  {"label": "Temperature[°C]", "value": "Temperature[°C]"},
  {"label": "Salinity[pss]", "value": "Salinity[pss]"},
  {"label": "Sigma_theta[kg/m^3]", "value": "Sigma_theta[kg/m^3]"},
  {"label": "Depth[m]", "value": "Depth[m]"},
  {"label": "Oxygen[µmol/kg]", "value": "Oxygen[µmol/kg]"},
  {"label": "OxygenSat[%]", "value": "OxygenSat[%]"},
  {"label": "Nitrate[µmol/kg]", "value": "Nitrate[µmol/kg]"},
  {"label": "Chl_a[mg/m^3]", "value": "Chl_a[mg/m^3]"},
  {"label": "b_bp700[1/m]", "value": "b_bp700[1/m]"},
  {"label": "pHinsitu[Total]", "value": "pHinsitu[Total]"},
  {"label": "b_bp532[1/m]", "value": "b_bp532[1/m]"},
  {"label": "CDOM[ppb]", "value": "CDOM[ppb]"},
  {"label": "TALK_CANYONB[µmol/kg]", "value": "TALK_CANYONB[µmol/kg]"},
  {"label": "DIC_CANYONB[µmol/kg]", "value": "DIC_CANYONB[µmol/kg]"},
  {"label": "pCO2_CANYONB[µatm]", "value": "pCO2_CANYONB[µatm]"},
  {"label": "SAT_AR_CANYONB[]", "value": "SAT_AR_CANYONB[]"},
  {"label": "pH25C_1atm[Total]", "value": "pH25C_1atm[Total]"},
  {"label": "DOWNWELL_PAR[µmol Quanta/m^2/sec]", "value": "DOWNWELL_PAR[µmol Quanta/m^2/sec]"},
  {"label": "DOWN_IRRAD380[W/m^2/nm]", "value": "DOWN_IRRAD380[W/m^2/nm]"},
  {"label": "DOWN_IRRAD443[W/m^2/nm]", "value": "DOWN_IRRAD443[W/m^2/nm]"},
  {"label": "DOWN_IRRAD490[W/m^2/nm]", "value": "DOWN_IRRAD490[W/m^2/nm]"},
  {"label": "VRS[Volts]", "value": "VRS[Volts]"},
  {"label": "VRS_STD[Volts]", "value": "VRS_STD[Volts]"},
  {"label": "VK[Volts]", "value": "VK[Volts]"},
  {"label": "VK_STD[Volts]", "value": "VK_STD[Volts]"},
  {"label": "IK[nA]", "value": "IK[nA]"},
  {"label": "Ib[nA]", "value": "Ib[nA]"}
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
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='map-plot', style={'height': '1000px','width': '1000px'})])]), width=8),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-pH', style={'height': '500px'})])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([dcc.Graph(id='scatter-plot-pH-delta', style={'height': '500px'})])]), width=4),
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
                dcc.Dropdown(
                    id='x-axis-dropdown', 
                    options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col],
                    multi=True, 
                    value="pH25C_1atm[Total]", 
                    clearable=False)
            ])
        ]), width=4),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Label("Select Y-axis:"),
                dcc.Dropdown(
                    id='y-axis-dropdown', 
                    options=[{'label': col, 'value': col} for col in df.columns if 'QF' not in col], 
                    # multi=True,
                    value="Depth[m]", 
                    clearable=False)
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

    # df = load_latest_data(folder_path,selected_file)
    df = pd.read_sql(query, engine) #double quotes needed for case-sensitive or numeric names. Change to selected file.


    # Get new min/max values for filters
    station_min, station_max = df["Station"].min(), df["Station"].max()
    date_min, date_max = df["Date"].min(), df["Date"].max()

    # Return updated values for filters
    return station_min, station_max, [station_min, station_max], date_min, date_max, date_min, date_max


@callback(
    [Output('station-range-slider', 'enabled'),
     Output('date-picker-range', 'enabled')],
    Input('filter-method', 'value')
)
def toggle_filters(selected_filter):
    return selected_filter != 'station', selected_filter != 'date'

@callback(
    Output('profile-number', 'value'),
    Input('Deployment-Dropdown', 'value')
)
def update_profile_number(selected_file):
    # df = load_latest_data(folder_path, selected_file)
    df = pd.read_sql(query, engine) #double quotes needed for case-sensitive or numeric names

    
    if df.empty:
        return None

    station_max = df["Station"].max()
    return station_max  # Set to the max station of the new file

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

    # Scatter Plot pH25atm
    scatter_fig_pH25 = px.scatter(
        filtered_df, x="pH25C_1atm[Total]", y="Depth[m]",
        labels={"pH25C_1atm[Total]", "Depth[m]", "Profile"},
        title=f"pH25C_1atm[Total] vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_pH25.update_yaxes(autorange="reversed")
    # scatter_fig.update_layout(height=1000, width=1000)

    scatter_fig_pHin_delta = px.scatter(
        filtered_df, x="pHin_Canb_Delta", y="Depth[m]",
        labels={"pHin - pHin_Canb", "Depth[m]", "Profile"},
        title=f"pHin - pHin_Canb vs. Depth[m]",
        template="plotly_white",
        color='Station'
    )
    scatter_fig_pH25.update_yaxes(autorange="reversed")
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

    # scatter_fig_xy = px.scatter(
    #     filtered_df, x=x_column, y=y_column,
    #     labels={x_column: x_column, y_column: y_column, "Station": "Profile"},
    #     title=f"{x_column} vs. {y_column}",
    #     template="plotly_white",
    #     color='Station'
    # )
    # scatter_fig_xy.update_yaxes(autorange="reversed")

    scatter_fig_xy = go.Figure()

    # Ensure x_column is always a list
    if isinstance(x_column, str):
        x_columns = [x_column]  # Convert single selection to list
    else:
        x_columns = x_column  # Already a list

    # Filter out any x-columns not in the DataFrame
    valid_x_columns = [x for x in x_columns if x in filtered_df.columns]

    if not valid_x_columns:
        raise ValueError("No valid x-axis columns found in the selected data.")

    # Iterate over valid x-columns and add traces with separate x-axes
    for i, x_col in enumerate(valid_x_columns):
        xaxis_name = "x" if i == 0 else f"x{i+1}"  # First axis is "x", others are "x2", "x3", etc.

        scatter_fig_xy.add_trace(go.Scatter(
            x=filtered_df[x_col],
            y=filtered_df[y_column],
            mode='markers',
            name=x_col,  # Legend entry for each x-column
            xaxis=xaxis_name  # Assign to respective x-axis
        ))

    # Define layout with multiple x-axes
    layout = {
        "title": f"{', '.join(valid_x_columns)} vs. {y_column}",
        "yaxis": {"title": y_column, "autorange": "reversed"},
        "xaxis": {"title": valid_x_columns[0]},  # Primary x-axis
    }

    # Add additional x-axes dynamically
    for i, x_col in enumerate(valid_x_columns[1:], start=2):
        layout[f"xaxis{i}"] = {
            "title": x_col,
            "anchor":"free",
            "overlaying": "x",  # Overlay on the same plot
            # "side": "top" if i % 2 == 0 else "bottom",  # Alternate positions
            "side": "bottom",
            "position": i * .1,  # Offset each x-axis by 5% of the plot width
            "showgrid": False,  # Hide grid for additional x-axes
            # "tickangle": 45 if i % 2 == 0 else -45  # Tilt ticks for legibility
            "tickmode":"sync",
        }

    scatter_fig_xy.update_layout(layout, template="plotly_white")




    return map_fig, scatter_fig_pH25, scatter_fig_pHin_delta, scatter_fig_Chla, scatter_fig_Temperature, scatter_fig_Salinity, scatter_fig_Doxy, scatter_fig_xy

if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(host='0.0.0.0', port=8050, debug=True)
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=port)