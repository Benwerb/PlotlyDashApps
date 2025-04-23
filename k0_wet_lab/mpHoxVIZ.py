import os
import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc


# App initialization
external_stylesheets = [dbc.themes.CERULEAN]
app = dash.Dash(__name__, title="mpHoxVIZ",external_stylesheets=external_stylesheets)
server = app.server

# Hardcode path to files and create list
folder_path = r"\\atlas.shore.mbari.org\ProjectLibrary\901805_Coastal_Biogeochemical_Sensing\Wetlab_Sensor_Calibration\mpHox"
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# App layout
app.layout = html.Div([
    html.H1("mpHoxVIZ", style={'textAlign': 'center', 'marginBottom': '30px', 'marginTop': '20px'}),
    
    html.Div([
        html.Div([
            html.Label('Select CSV File:'),
            dcc.Dropdown(
                id='file-dropdown',
                options=[{'label': file, 'value': file} for file in files],
                value=files[0] if files else None,
                clearable=False
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('X-Axis:'),
            dcc.Dropdown(
                id='x-axis-dropdown',
                value='DateTime',
                clearable=False
            )
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Y-Axis:'),
            dcc.Dropdown(
                id='y-axis-dropdown',
                value='Vrse',
                clearable=False
            )
        ], style={'width': '33%', 'display': 'inline-block'})
    ], style={'marginBottom': '20px'}),
    
    dcc.Graph(id='data-plot', style={'height': '70vh'}),
    
    html.Div(id='data-info', style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
])

# Update dropdown options based on loaded file
@callback(
    [Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('data-info', 'children')],
    [Input('file-dropdown', 'value')]
)
def update_dropdowns(selected_file):
    if not selected_file:
        return [], [], "No file selected"
    
    try:
        # Load the data
        file_path = os.path.join(folder_path, selected_file)
        df = pd.read_csv(file_path)
        
        # Get column options
        columns = df.columns.tolist()
        options = [{'label': col, 'value': col} for col in columns]
        
        # Create info text
        info_text = [
            html.P(f"File: {selected_file}"),
            html.P(f"Rows: {len(df)}"),
            html.P(f"Columns: {len(columns)}")
        ]
        
        return options, options, info_text
    
    except Exception as e:
        return [], [], f"Error loading file: {str(e)}"

# Update the plot based on selections
@callback(
    Output('data-plot', 'figure'),
    [Input('file-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_plot(selected_file, x_axis, y_axis):
    if not selected_file or not x_axis or not y_axis:
        return px.scatter(title="Please select valid parameters")
    
    try:
        # Load the data
        file_path = os.path.join(folder_path, selected_file)
        df = pd.read_csv(file_path)
        
        # Handle DateTime conversion if it's the x-axis
        if x_axis == 'DateTime':
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            except:
                pass
        
        # Create the plot
        fig = px.scatter(
            df, 
            x=x_axis, 
            y=y_axis,
            title=f"{y_axis} vs {x_axis} from {selected_file}",
            hover_data=df.columns[:5].tolist()  # Show the first few columns on hover
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    except Exception as e:
        # Return an empty figure with error message
        fig = px.scatter(title=f"Error: {str(e)}")
        return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)