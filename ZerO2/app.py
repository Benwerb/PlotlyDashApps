"""
Plotly Dash dashboard for O2 sensor data visualization.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from pathlib import Path
from data_parser import load_data_file, apply_filter
from gsw import O2sol

# Data source configuration
# TODO: Replace with FTP connection when available
DATA_FOLDER = Path('data')

# External stylesheet from Dash Bootstrap Components
import dash_bootstrap_components as dbc

# Initialize Dash app with Journal theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])

# Get list of available data files, sorted by modification time
def get_data_files():
    """Get list of data files sorted by modification time (newest first)."""
    if not DATA_FOLDER.exists():
        return []
    
    files = []
    for file in DATA_FOLDER.glob('*.txt'):
        files.append({
            'label': file.name,
            'value': str(file)
        })
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x['value']), reverse=True)
    return files


# App layout
app.layout = html.Div([
    html.H1("O2 Sensor Data Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # File selection section
    html.Div([
        html.Div([
            html.Label("File 1:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='file1-dropdown',
                options=get_data_files(),
                value=None,
                placeholder="Select file 1",
                style={'width': '100%'}
            )
        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),
        
        html.Div([
            html.Label("File 2:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='file2-dropdown',
                options=get_data_files(),
                value=None,
                placeholder="Select file 2",
                style={'width': '100%'}
            )
        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),
        
        html.Div([
            html.Label("File 3:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='file3-dropdown',
                options=get_data_files(),
                value=None,
                placeholder="Select file 3",
                style={'width': '100%'}
            )
        ], style={'width': '32%', 'display': 'inline-block'})
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f0f0f0'}),
    
    # O2 type selection
    html.Div([
        html.Label("O2 Display Type:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.RadioItems(
            id='o2-type-radio',
            options=[
                {'label': 'O2 Concentration', 'value': 'concentration'},
                {'label': 'O2 Saturation', 'value': 'saturation'}
            ],
            value='concentration',
            inline=True,
            style={'marginLeft': '10px'}
        )
    ], style={'marginBottom': '20px', 'padding': '10px'}),
    
    # Time series plot
    html.Div([
        html.H3("Time Series", style={'textAlign': 'center'}),
        dcc.Graph(id='time-series-plot')
    ], style={'marginBottom': '30px'}),
    
    # Filtered O2 vs Temperature plots
    html.Div([
        html.H3("O2 vs Temperature (Filtered)", style={'textAlign': 'center'}),
        dcc.Graph(id='filtered-plots')
    ])
])


@callback(
    [Output('file1-dropdown', 'value'),
     Output('file2-dropdown', 'value'),
     Output('file3-dropdown', 'value')],
    Input('file1-dropdown', 'options'),
    prevent_initial_call=False
)
def set_default_files(options):
    """Set default files to last 3 modified files."""
    if options and len(options) >= 3:
        return options[0]['value'], options[1]['value'], options[2]['value']
    elif options and len(options) == 2:
        return options[0]['value'], options[1]['value'], None
    elif options and len(options) == 1:
        return options[0]['value'], None, None
    else:
        return None, None, None


@callback(
    Output('time-series-plot', 'figure'),
    [Input('file1-dropdown', 'value'),
     Input('file2-dropdown', 'value'),
     Input('file3-dropdown', 'value'),
     Input('o2-type-radio', 'value')]
)
def update_time_series(file1, file2, file3, o2_type):
    """Update time series plot with separate subplots for each file, each with dual y-axes."""
    files = [f for f in [file1, file2, file3] if f is not None]
    
    if not files:
        # Return empty figure if no files selected
        fig = go.Figure()
        fig.add_annotation(
            text="No files selected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Determine which O2 column to use
    o2_column = 'o2_concentration' if o2_type == 'concentration' else 'o2_saturation'
    o2_label = 'O2 Concentration (uM)' if o2_type == 'concentration' else 'O2 Saturation (%)'
    
    # Create subplots (one per file) with secondary y-axis for each
    num_files = len(files)
    subplot_specs = [[{"secondary_y": True}] for _ in range(num_files)]
    fig = make_subplots(
        rows=num_files,
        cols=1,
        subplot_titles=[os.path.basename(f) for f in files],
        specs=subplot_specs,
        vertical_spacing=0.15
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, filepath in enumerate(files):
        if filepath and os.path.exists(filepath):
            try:
                df = load_data_file(filepath)
                filename = os.path.basename(filepath)
                
                # Add O2 trace (left y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[o2_column],
                        name=o2_label,
                        mode='lines',
                        line=dict(color=colors[idx % len(colors)], width=1.5),
                        legendgroup=f'file{idx}',
                        showlegend=(idx == 0)  # Only show legend for first file
                    ),
                    row=idx+1,
                    col=1,
                    secondary_y=False
                )
                
                # Add Temperature trace (right y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['temperature'],
                        name='Temperature',
                        mode='lines',
                        line=dict(color=colors[idx % len(colors)], width=1.5, dash='dash'),
                        legendgroup=f'file{idx}',
                        showlegend=(idx == 0)  # Only show legend for first file
                    ),
                    row=idx+1,
                    col=1,
                    secondary_y=True
                )
                
                # Set axis labels for each subplot (no x-axis label)
                fig.update_xaxes(title_text="", row=idx+1, col=1)
                fig.update_yaxes(title_text=o2_label, secondary_y=False, row=idx+1, col=1)
                fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True, row=idx+1, col=1)
                
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
    
    fig.update_layout(
        height=400 * num_files,
        hovermode='x unified',
        title="Time Series: O2 and Temperature"
    )
    
    return fig


@callback(
    Output('filtered-plots', 'figure'),
    [Input('file1-dropdown', 'value'),
     Input('file2-dropdown', 'value'),
     Input('file3-dropdown', 'value'),
     Input('o2-type-radio', 'value')]
)
def update_filtered_plots(file1, file2, file3, o2_type):
    """Update filtered O2 vs Temperature plots (one subplot per file)."""
    files = [f for f in [file1, file2, file3] if f is not None]
    
    if not files:
        # Return empty figure if no files selected
        fig = go.Figure()
        fig.add_annotation(
            text="No files selected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create subplots (one per file)
    num_files = len(files)
    fig = make_subplots(
        rows=num_files,
        cols=1,
        subplot_titles=[os.path.basename(f) for f in files],
        vertical_spacing=0.15
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    o2_column = 'o2_concentration' if o2_type == 'concentration' else 'o2_saturation'
    o2_label = 'O2 Concentration (uM)' if o2_type == 'concentration' else 'O2 Saturation (%)'
    
    for idx, filepath in enumerate(files):
        if filepath and os.path.exists(filepath):
            try:
                df = load_data_file(filepath)
                
                # Apply filter
                filter_mask = apply_filter(df)
                df_filtered = df[filter_mask].copy()
                
                # Always add a trace, even if empty (to ensure subplot renders)
                if len(df_filtered) > 0:
                    # Add scatter plot for filtered data
                    fig.add_trace(
                        go.Scatter(
                            x=df_filtered['temperature'],
                            y=df_filtered[o2_column],
                            mode='markers',
                            name=f'{os.path.basename(filepath)}',
                            marker=dict(
                                color=colors[idx % len(colors)],
                                size=4,
                                opacity=0.6
                            ),
                            showlegend=False
                        ),
                        row=idx+1,
                        col=1
                    )
                else:
                    # Add empty trace to ensure subplot is visible
                    fig.add_trace(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode='markers',
                            name=f'{os.path.basename(filepath)}',
                            showlegend=False
                        ),
                        row=idx+1,
                        col=1
                    )
                
                # Set axis labels for each subplot
                fig.update_xaxes(title_text="Temperature (°C)", row=idx+1, col=1)
                fig.update_yaxes(title_text=o2_label, row=idx+1, col=1)
                
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
                # Add empty trace even if there's an error to ensure subplot exists
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        showlegend=False
                    ),
                    row=idx+1,
                    col=1
                )
                fig.update_xaxes(title_text="Temperature (°C)", row=idx+1, col=1)
                fig.update_yaxes(title_text=o2_label, row=idx+1, col=1)
    
    fig.update_layout(
        height=300 * num_files,
        title_text="O2 vs Temperature (Filtered: O2Sat < 9 & |diff(O2Sat)| < 0.0005)"
    )
    
    return fig


if __name__ == '__main__':
    # Run server accessible on network
    # Set host='0.0.0.0' to make it accessible from other machines on the network
    # Set debug=False for production use
    app.run(debug=True, host='0.0.0.0', port=8050)

