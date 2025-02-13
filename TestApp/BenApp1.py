# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

# Load the data
file_path = "25202901RT.txt"
df = pd.read_csv(file_path, delimiter="\t", skiprows=6)

# Replace missing values
df.replace(-1e10, pd.NA, inplace=True)

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Available x-axis options
x_options = [
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

# Available y-axis options
y_options = [
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

# Update the layout to include the scatter plot
app.layout = dbc.Container([
    dbc.Row([
        html.H3('Variable vs. Variable Plot', className="text-primary text-center")
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select X-axis:"),
            dcc.Dropdown(
                options=x_options,
                value="Temperature[°C]",  # Default selection
                clearable=False,
                searchable=True,
                id='x-axis-dropdown'
            )
        ], width=4),

        dbc.Col([
            html.Label("Select Y-axis:"),
            dcc.Dropdown(
                options=y_options,
                value="Depth[m]",  # Default selection
                clearable=False,
                searchable=True,
                id='y-axis-dropdown'
            )
        ], width=4)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='scatter-plot',style={'height': '1000px', 'width': '1000px'})
        ], width=24),
    ]),

], fluid=True)

# Callback to update the graph
@callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_graph(x_column, y_column):
    fig = px.scatter(df, x=x_column, y=y_column, 
                     labels={x_column: x_column, y_column: y_column},
                     title=f"{x_column} vs. {y_column}",
                     template="plotly_white")
    
    # Reverse the y-axis for proper depth representation
    fig.update_yaxes(autorange="reversed")

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
