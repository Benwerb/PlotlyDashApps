#!/usr/bin/env python3
"""
Spatial Interpolation Function for Nessie.py Integration
This function provides spatial interpolation capabilities that can be easily integrated
into your existing Nessie dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def create_spatial_interpolation(df, parameter='rhodamine', hours_back=24, 
                               platform_filter=None, layer_filter=None, 
                               grid_resolution=40, method='linear',
                               nan_filter_parameters=None):
    """
    Create a spatial interpolation plot for integration into Nessie.py
    
    Args:
        df (pd.DataFrame): Input dataframe with columns: lat, lon, datetime, and parameter
        parameter (str): Parameter name to interpolate (e.g., 'rhodamine', 'pHin', 'MLD')
        hours_back (int): Number of hours to look back from latest time
        platform_filter (str or list): Platform(s) to include (e.g., 'Ship', 'Glider', None for all)
        layer_filter (str or list): Layer(s) to include (e.g., 'Surface', 'MLD', None for all)
        grid_resolution (int): Grid resolution for interpolation (default: 80)
        method (str): Interpolation method ('linear', 'cubic', 'nearest')
        nan_filter_parameters (list): Parameters to check for NaN values (None for defaults)
        
    Returns:
        go.Figure: Plotly figure object ready for display
        dict: Metadata about the interpolation (data points, time range, etc.)
    """
    
    # Input validation
    if df is None or len(df) == 0:
        print("Error: No data provided")
        return None, None
    
    required_columns = ['lat', 'lon', 'Datetime', parameter]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None, None
    
    # Create a copy to avoid modifying original dataframe
    df_work = df.copy()
    
    # Apply filters
    df_filtered = apply_filters(df_work, hours_back, platform_filter, layer_filter, nan_filter_parameters)
    if df_filtered is None or len(df_filtered) == 0:
        print("No data available after filtering")
        return None, None
    
    # Create interpolation grid
    lat_grid, lon_grid, lat_mesh, lon_mesh = create_interpolation_grid(df_filtered, grid_resolution)
    if lat_grid is None:
        return None, None
    
    # Perform interpolation
    interpolated_values = interpolate_parameter(df_filtered, lat_mesh, lon_mesh, parameter, method)
    if interpolated_values is None:
        return None, None
    
    # Create the plot
    fig = create_interpolation_plot(df_filtered, lat_grid, lon_grid, interpolated_values, parameter, method)
    
    # Prepare metadata
    metadata = {
        'parameter': parameter,
        'data_points': len(df_filtered),
        'time_range': f"{df_filtered['Datetime'].min().strftime('%Y-%m-%d %H:%M')} to {df_filtered['Datetime'].max().strftime('%Y-%m-%d %H:%M')}",
        'spatial_bounds': {
            'lat_min': df_filtered['lat'].min(),
            'lat_max': df_filtered['lat'].max(),
            'lon_min': df_filtered['lon'].min(),
            'lon_max': df_filtered['lon'].max()
        },
        'grid_resolution': grid_resolution,
        'method': method,
        'platform_filter': platform_filter,
        'layer_filter': layer_filter,
        'hours_back': hours_back,
        'nan_filter_parameters': nan_filter_parameters
    }
    
    return fig, metadata

def filter_nan_values(df, parameters=None):
    """
    Filter out rows where specified parameters have NaN values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        parameters (list): List of parameter names to check for NaN values
        
    Returns:
        pd.DataFrame: Filtered dataframe with no NaN values in specified parameters
    """
    if df is None or len(df) == 0:
        return df
    
    if parameters is None:
        # Default to common oceanographic parameters
        parameters = ['rhodamine', 'pHin', 'MLD', 'temperature', 'salinity']
    
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    
    for param in parameters:
        if param in df_filtered.columns:
            before_count = len(df_filtered)
            df_filtered = df_filtered[~df_filtered[param].isna()].copy()
            after_count = len(df_filtered)
            removed = before_count - after_count
            if removed > 0:
                print(f"  Filtered out {removed} rows with NaN {param} values")
    
    total_removed = initial_count - len(df_filtered)
    if total_removed > 0:
        print(f"  Total rows removed due to NaN values: {total_removed}")
        print(f"  Remaining data points: {len(df_filtered)}")
    
    return df_filtered

def apply_filters(df, hours_back, platform_filter=None, layer_filter=None, nan_filter_parameters=None):
    """Apply time, platform, and layer filters to the dataframe"""
    
    df_filtered = df.copy()
    
    # Apply platform filter
    if platform_filter is not None:
        if isinstance(platform_filter, str):
            platform_filter = [platform_filter]
        if 'Platform' in df_filtered.columns:
            initial_count = len(df_filtered)
            df_filtered = df_filtered[df_filtered['Platform'].isin(platform_filter)].copy()
            print(f"Platform filter: {len(df_filtered)} points after filtering")
    
    # Apply layer filter
    if layer_filter is not None:
        if isinstance(layer_filter, str):
            layer_filter = [layer_filter]
        if 'Layer' in df_filtered.columns:
            initial_count = len(df_filtered)
            df_filtered = df_filtered[df_filtered['Layer'].isin(layer_filter)].copy()
            print(f"Layer filter: {len(df_filtered)} points after filtering")
    
    # Apply time filter
    if 'Datetime' in df_filtered.columns:
        latest_time = df_filtered['Datetime'].max()
        time_window = timedelta(hours=hours_back)
        start_time = latest_time - time_window
        
        initial_count = len(df_filtered)
        df_filtered = df_filtered[df_filtered['Datetime'] >= start_time].copy()
        print(f"Time filter: {len(df_filtered)} points in last {hours_back} hours")
    
    # Apply NaN value filter
    if nan_filter_parameters is not None:
        df_filtered = filter_nan_values(df_filtered, nan_filter_parameters)
    
    return df_filtered

def create_interpolation_grid(df_filtered, grid_resolution):
    """Create a regular grid for interpolation"""
    
    if df_filtered is None or len(df_filtered) == 0:
        return None, None, None, None
    
    # Get spatial bounds
    lat_min, lat_max = df_filtered['lat'].min(), df_filtered['lat'].max()
    lon_min, lon_max = df_filtered['lon'].min(), df_filtered['lon'].max()
    
    # Add small buffer to avoid edge effects
    lat_buffer = (lat_max - lat_min) * 0.05
    lon_buffer = (lon_max - lon_min) * 0.05
    
    lat_min -= lat_buffer
    lat_max += lat_buffer
    lon_min -= lon_buffer
    lon_max += lon_buffer
    
    # Create regular grid
    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    return lat_grid, lon_grid, lat_mesh, lon_mesh

def interpolate_parameter(df_filtered, lat_mesh, lon_mesh, parameter, method='linear'):
    """Interpolate a parameter over the spatial grid"""
    
    if df_filtered is None or len(df_filtered) == 0:
        return None
    
    # Prepare data for interpolation
    points = df_filtered[['lon', 'lat']].values
    values = df_filtered[parameter].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(values)
    
    if not valid_mask.any():
        print(f"Warning: No valid values for parameter '{parameter}'")
        return None
    
    if valid_mask.sum() < 3:
        print(f"Warning: Only {valid_mask.sum()} valid points for '{parameter}', need at least 3")
        return None
    
    points_valid = points[valid_mask]
    values_valid = values[valid_mask]
    
    # Perform interpolation
    try:
        interpolated_values = griddata(
            points_valid, 
            values_valid, 
            (lon_mesh, lat_mesh), 
            method=method, 
            fill_value=np.nan
        )
        return interpolated_values
        
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None

def create_interpolation_plot(df_filtered, lat_grid, lon_grid, interpolated_values, parameter, method):
    """Create a Plotly figure showing the interpolated data"""
    
    if interpolated_values is None:
        return None
    
    # Create the main figure
    fig = go.Figure()
    
    # Add contour plot for interpolated values
    colorscale = 'Reds' if parameter == 'rhodamine' else 'Blues'
    
    fig.add_trace(go.Contour(
        z=interpolated_values,
        x=lon_grid,
        y=lat_grid,
        colorscale=colorscale,
        name=f'{parameter} Interpolation',
        showscale=True,
        colorbar=dict(
            title=f'{parameter}',
            len=0.5,  # Shorter colorbar
            thickness=15,
            x=1.02,  # Position outside the plot on the right
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=10, color='white')
        )
    ))
    
    # Add original data points with symbols based on Cruise IDs
    if len(df_filtered) > 0:
        hover_text = df_filtered['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get unique cruise IDs and assign symbols
        if 'Cruise' in df_filtered.columns:
            unique_cruises = df_filtered['Cruise'].unique()
            # Define a set of distinct symbols
            symbols = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 
                      'star', 'cross', 'x', 'hexagon', 'pentagon', 'octagon']
            
            # Create a mapping of cruise to symbol
            cruise_symbol_map = {}
            for i, cruise in enumerate(unique_cruises):
                cruise_symbol_map[cruise] = symbols[i % len(symbols)]
            
            # Add data points grouped by cruise
            for cruise in unique_cruises:
                cruise_data = df_filtered[df_filtered['Cruise'] == cruise]
                
                fig.add_trace(go.Scatter(
                    x=cruise_data['lon'],
                    y=cruise_data['lat'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cruise_data[parameter] if parameter in cruise_data.columns else 'black',
                        colorscale=colorscale,
                        showscale=False,
                        line=dict(color='black', width=1),
                        symbol=cruise_symbol_map[cruise]
                    ),
                    name=f'{cruise} Data Points',
                    text=cruise_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    hovertemplate='<b>Cruise:</b> ' + cruise + '<br>' +
                                 '<b>Time:</b> %{text}<br>' +
                                 '<b>Lat:</b> %{y:.4f}°N<br>' +
                                 '<b>Lon:</b> %{x:.4f}°W<br>' +
                                 f'<b>{parameter}:</b> %{{marker.color:.3f}}<extra></extra>'
                ))
        else:
            # Fallback if no Cruise column - use default markers
            fig.add_trace(go.Scatter(
                x=df_filtered['lon'],
                y=df_filtered['lat'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_filtered[parameter] if parameter in df_filtered.columns else 'black',
                    colorscale=colorscale,
                    showscale=False,
                    line=dict(color='black', width=1)
                ),
                name='Original Data Points',
                text=hover_text,
                hovertemplate='<b>Time:</b> %{text}<br>' +
                             '<b>Lat:</b> %{y:.4f}°N<br>' +
                             '<b>Lon:</b> %{x:.4f}°W<br>' +
                             f'<b>{parameter}:</b> %{{marker.color:.3f}}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=f'{parameter} Spatial Interpolation ({method} method)<br>' +
              f'Data from {df_filtered["Datetime"].min().strftime("%Y-%m-%d %H:%M")} to ' +
              f'{df_filtered["Datetime"].max().strftime("%Y-%m-%d %H:%M")}',
        xaxis_title='Longitude (°W)',
        yaxis_title='Latitude (°N)',
        width=900,
        height=700,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,  # Move legend outside to the right
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
            bordercolor='black',
            borderwidth=1,
        )
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig