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
folder_path = r"\\atlas.shore.mbari.org\ProjectLibrary\901805_Coastal_Biogeochemical_Sensing\Wetlab_Sensor_Calibration\NanoFet\k0"

files = [f for f in os.listdir(folder_path) if '.csv']

# Parse metadata from the first CSV file
def parse_metadata(folder_path):
    """
    Parse metadata about COM ports and NanoFet IDs
    Returns:
    - com_dict: Dictionary mapping COM ports to NanoFet IDs
    - nanofet_ids: List of NanoFet IDs
    - com_ports: List of corresponding COM ports
    """
    filename = os.path.join(folder_path, files[-1])
    with open(filename, 'r') as f:
        # Read first few lines to extract metadata
        metadata_lines = [next(f).strip() for _ in range(3)]
    
    # Parse COM ports and NanoFet IDs
    com_ports = []
    nanofet_ids = []
    
    for line in metadata_lines:
        if line.startswith('NanoFetIDs:'):
            nanofet_ids = [id.strip() for id in line.split(':')[1].split(';')]
        elif line.startswith('NanoFets:'):
            com_ports = [port.strip() for port in line.split(':')[1].split(';')]
    
    # Create mapping between COM ports and NanoFet IDs
    com_dict = dict(zip(com_ports, nanofet_ids))
    
    return com_dict, nanofet_ids, com_ports

# Load and clean data
def load_latest_data(folder_path, selected_file=None, com_dict=None):
    """Loads the latest CSV file, cleans it, and returns a DataFrame."""
    filename = os.path.join(folder_path, selected_file if selected_file else files[-1])
    df = pd.read_csv(filename, delimiter=",", skiprows=4)
    
    # Clean data
    # df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M')
    
    # Replace COM with NanoFet ID if metadata is available
    if com_dict:
        df['NanoFet'] = df['COM'].map(com_dict)
    
    return df

# Parse initial metadata
com_dict, nanofet_ids, com_ports = parse_metadata(folder_path)

# Initial data load
df = load_latest_data(folder_path, com_dict=com_dict)

# Initialize the app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Dynamically create layout based on number of NanoFets
def create_layout(nanofet_ids, df):
    """
    Dynamically create app layout based on number of NanoFets
    """
    # Determine number of columns based on number of NanoFets
    num_nanofets = len(nanofet_ids)
    cols_per_row = 2  # Can be adjusted
    rows = (num_nanofets + cols_per_row - 1) // cols_per_row

    # Create graph rows dynamically
    graph_rows = []
    for r in range(rows):
        row_cols = []
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < num_nanofets:
                nanofet = nanofet_ids[idx]
                col = dbc.Col([
                    dcc.Graph(id=f'scatter-plot-{nanofet}', 
                              style={'height': 'auto', 'width': 'auto'}),
                ], width=6)
                row_cols.append(col)
        graph_rows.append(dbc.Row(row_cols))

    # Magnitude comparison plot row
    graph_rows.append(
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='magnitude-comparison-plot', 
                          style={'height': 'auto', 'width': 'auto'}),
            ], width=12)
        ])
    )

    # Full layout
    layout = dbc.Container([
        dbc.Row([html.H3('WetViz', className="text-primary text-center")]),

        dbc.Row([
            dbc.Col([
                html.Label("Select file:"),
                dcc.Dropdown(files, files[-1], id='Deployment-Dropdown', clearable=True)
            ], width=3)
        ], className="mb-3"),

        *graph_rows,  # Unpack dynamic graph rows

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Label("Select X-axis:"),
                    dcc.Dropdown(
                        id='x-axis-dropdown', 
                        options=[{'label': col, 'value': col} for col in df.columns if col not in ['COM', 'NanoFet']],
                        value="DateTime", 
                        clearable=False)
                ])
            ]), width=4),
            
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Label("Select Y-axis:"),
                    dcc.Dropdown(
                        id='y-axis-dropdown', 
                        options=[{'label': col, 'value': col} for col in df.columns if col not in ['COM', 'NanoFet']],
                        value="Vrse", 
                        clearable=False)
                ])
            ]), width=4)
        ]),
    ], fluid=True)

    return layout

# Set the app layout dynamically
app.layout = create_layout(nanofet_ids, df)

# Create dynamic callback for multiple NanoFet graphs
def create_multi_output_callback(nanofet_ids):
    """
    Dynamically create callback outputs for multiple NanoFet graphs
    """
    @callback(
        [Output(f'scatter-plot-{nanofet}', 'figure') for nanofet in nanofet_ids] + 
        [Output('magnitude-comparison-plot', 'figure')],
        [Input('Deployment-Dropdown', 'value'),
         Input('x-axis-dropdown','value'),
         Input('y-axis-dropdown', 'value')]
    )
    def update_graphs(selected_file, x_column, y_column):
        # Re-parse metadata and load data each time to handle dynamic changes
        current_com_dict, current_nanofet_ids, _ = parse_metadata(folder_path)
        
        # Load data with NanoFet ID mapping
        df = load_latest_data(folder_path, selected_file, current_com_dict)
        
        # Create plots for each NanoFet
        figures = []
        for nanofet in current_nanofet_ids:
            df_nanofet = df[df['NanoFet'] == nanofet]
            
            fig = px.scatter(
                df_nanofet, 
                x=x_column, 
                y=y_column, 
                title=f'{nanofet}: {x_column} vs. {y_column}',
                labels={x_column: x_column, y_column: y_column}
            )
            figures.append(fig)
        
        # Create magnitude comparison plot
        magnitude_fig = create_magnitude_comparison(df, y_column, nanofet_ids)
        figures.append(magnitude_fig)
        
        return figures

    return update_graphs

def create_magnitude_comparison(df, y_column, nanofet_ids):
    """
    Create an offset comparison plot to compare magnitude of Vrse changes across all NanoFets
    """
    # Prepare figure
    fig = go.Figure()
    
    # Apply offset for each NanoFet to align starting points
    for nanofet in nanofet_ids:
        df_nanofet = df[df['NanoFet'] == nanofet].sort_values('DateTime')  # Sort by DateTime
        
        # x_data is always DateTime (no conversion needed)
        x_data = df_nanofet['DateTime']
        y_data = pd.to_numeric(df_nanofet[y_column], errors='coerce')
        
        # Only process if we have data
        if not x_data.empty and not y_data.empty:
            # Calculate offset to align all sensors at the same starting point
            first_value = y_data.iloc[0]  # Get first measurement for this sensor
            target_start = 0  # Or choose a common starting value
            offset = target_start - first_value
            
            y_offset = y_data + offset  # Apply offset to align starting points
            
            # Add trace for this NanoFet
            fig.add_trace(go.Scatter(
                x=x_data, 
                y=y_offset,
                mode='markers+lines',  # Add lines to better see trends
                name=nanofet,
                opacity=0.7
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Magnitude Comparison (Offset-Aligned): DateTime vs. {y_column}',
        xaxis_title='DateTime',
        yaxis_title=f'{y_column} (Offset-Aligned)',
        legend_title_text='NanoFet IDs'
    )
    
    return fig

# Apply the dynamic callback
app.callback = create_multi_output_callback(nanofet_ids)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)