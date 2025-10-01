# Import packages
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import os
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# Hardcode path to files and create list
folder_path = r"\\atlas.shore.mbari.org\ProjectLibrary\901805_Coastal_Biogeochemical_Sensing\Wetlab_Sensor_Calibration\NanoFet\K0"

files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Get sensor IDs from the data file
def get_sensor_ids(folder_path, selected_file=None):
    """
    Get unique sensor IDs from the CSV file
    Returns:
    - sensor_ids: List of unique sensor IDs
    """
    filename = os.path.join(folder_path, selected_file if selected_file else files[-1])
    df = pd.read_csv(filename)
    sensor_ids = df['SensorID'].unique().tolist()
    return sensor_ids

# Load and clean data
def load_latest_data(folder_path, selected_file=None):
    """Loads the latest CSV file, cleans it, and returns a DataFrame."""
    filename = os.path.join(folder_path, selected_file if selected_file else files[-1])
    df = pd.read_csv(filename)
    
    # Convert DateTime to datetime if needed
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    return df

# Get initial sensor IDs
sensor_ids = get_sensor_ids(folder_path)

# Initial data load
df = load_latest_data(folder_path)

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Dynamically create layout with fixed 4 sensor plots
def create_layout(df):
    """
    Create fixed layout with exactly 4 sensor plots plus magnitude comparison
    """
    # Fixed layout with 4 sensor plots in 2x2 grid
    layout = dbc.Container([
        dbc.Row([html.H3('WetViz', className="text-primary text-center")]),

        dbc.Row([
            dbc.Col([
                html.Label("Select file:"),
                dcc.Dropdown(files, files[-1], id='Deployment-Dropdown', clearable=True)
            ], width=3)
        ], className="mb-3"),

        # Fixed 2x2 grid for sensor plots
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='sensor-plot-1', style={'height': '400px'}),
            ], width=6),
            dbc.Col([
                dcc.Graph(id='sensor-plot-2', style={'height': '400px'}),
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='sensor-plot-3', style={'height': '400px'}),
            ], width=6),
            dbc.Col([
                dcc.Graph(id='sensor-plot-4', style={'height': '400px'}),
            ], width=6)
        ]),

        # Magnitude comparison plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='magnitude-comparison-plot', style={'height': '400px'}),
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Label("Select X-axis:"),
                    dcc.Dropdown(
                        id='x-axis-dropdown', 
                        options=[{'label': col, 'value': col} for col in df.columns if col not in ['COM', 'SensorID']],
                        value="DateTime", 
                        clearable=False)
                ])
            ]), width=4),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Label("Select Y-axis:"),
                    dcc.Dropdown(
                        id='y-axis-dropdown', 
                        options=[{'label': col, 'value': col} for col in df.columns if col not in ['COM', 'SensorID']],
                        value="Vrse", 
                        clearable=False)
                ])
            ]), width=4)
        ]),
    ], fluid=True)

    return layout

# Set the app layout with fixed 4 plots
app.layout = create_layout(df)

# Create single callback for all plots
@callback(
    [Output('sensor-plot-1', 'figure'),
     Output('sensor-plot-2', 'figure'),
     Output('sensor-plot-3', 'figure'),
     Output('sensor-plot-4', 'figure'),
     Output('magnitude-comparison-plot', 'figure')],
    [Input('Deployment-Dropdown', 'value'),
     Input('x-axis-dropdown','value'),
     Input('y-axis-dropdown', 'value')]
)
def update_all_plots(selected_file, x_column, y_column):
    """Update all plots with data from selected file"""
    # Get current sensor IDs and load data
    current_sensor_ids = get_sensor_ids(folder_path, selected_file)
    df = load_latest_data(folder_path, selected_file)
    
    # Create plots for each sensor slot (up to 4)
    figures = []
    for i in range(4):
        if i < len(current_sensor_ids):
            # Sensor exists in current file
            sensor = current_sensor_ids[i]
            df_sensor = df[df['SensorID'] == sensor]
            
            fig = px.scatter(
                df_sensor, 
                x=x_column, 
                y=y_column, 
                title=f'{sensor}: {x_column} vs. {y_column}',
                labels={x_column: x_column, y_column: y_column}
            )
        else:
            # No sensor for this slot
            fig = go.Figure()
            fig.update_layout(
                title=f'Sensor Slot {i+1}: No data available',
                xaxis_title=x_column,
                yaxis_title=y_column,
                annotations=[{
                    'text': 'No sensor data for this slot',
                    'x': 0.5, 'y': 0.5,
                    'xref': 'paper', 'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
        figures.append(fig)
    
    # Create magnitude comparison plot
    magnitude_fig = create_magnitude_comparison(df, y_column, current_sensor_ids)
    figures.append(magnitude_fig)
    
    return figures

def create_magnitude_comparison(df, y_column, sensor_ids):
    """
    Create an offset comparison plot to compare magnitude of Vrse changes across all sensors
    """
    # Prepare figure
    fig = go.Figure()
    
    # Apply offset for each sensor to align starting points
    for sensor in sensor_ids:
        df_sensor = df[df['SensorID'] == sensor].sort_values('DateTime')  # Sort by DateTime
        
        # x_data is always DateTime (no conversion needed)
        x_data = df_sensor['DateTime']
        y_data = pd.to_numeric(df_sensor[y_column], errors='coerce')
        
        # Only process if we have data
        if not x_data.empty and not y_data.empty:
            # Calculate offset to align all sensors at the same starting point
            first_value = y_data.iloc[0]  # Get first measurement for this sensor
            target_start = 0  # Or choose a common starting value
            offset = target_start - first_value
            
            y_offset = y_data + offset  # Apply offset to align starting points
            
            # Add trace for this sensor
            fig.add_trace(go.Scatter(
                x=x_data, 
                y=y_offset,
                mode='markers+lines',  # Add lines to better see trends
                name=sensor,
                opacity=0.7
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Magnitude Comparison (Offset-Aligned): DateTime vs. {y_column}',
        xaxis_title='DateTime',
        yaxis_title=f'{y_column} (Offset-Aligned)',
        legend_title_text='Sensor IDs'
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)