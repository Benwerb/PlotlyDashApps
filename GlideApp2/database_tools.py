import pandas as pd
import sqlalchemy
from sqlalchemy import text

def create_engine():
    return sqlalchemy.create_engine("postgresql://glidata_user:IoFUTBeaQDppSYcmBebA4rV8SJOEMCFI@dpg-d2jobg3e5dus738ce5vg-a.oregon-postgres.render.com/glidata")

def get_mission_metadata() -> pd.DataFrame:
    """
    Fetch metadata for all missions including:
    - mission_id
    - start_date (earliest timestamp)
    - end_date (latest timestamp)
    - total_dives (count of unique dive numbers)
    - latest_timestamp (for sorting by most recent data)
    
    Returns sorted by most recent data first.
    """
    engine = create_engine()
    sql = """
        SELECT 
            mission_id,
            MIN(unixtime) as start_timestamp,
            MAX(unixtime) as end_timestamp,
            COUNT(DISTINCT divenumber) as total_dives
        FROM public.real_time_binned
        GROUP BY mission_id
        ORDER BY MAX(unixtime) DESC
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
        # Convert timestamps to datetime
        df['start_date'] = pd.to_datetime(df['start_timestamp'], unit='s')
        df['end_date'] = pd.to_datetime(df['end_timestamp'], unit='s')
        return df

def get_available_missions() -> list:
    """
    Fetch all unique mission_ids from the database, ordered by most recent data first.
    Returns list of mission_ids.
    """
    metadata = get_mission_metadata()
    return metadata['mission_id'].tolist()

def get_dive_range(mission_id: str) -> tuple:
    """
    Get the min and max dive numbers for a mission without loading all the data.
    
    Returns:
        tuple: (min_dive, max_dive)
    """
    engine = create_engine()
    sql = """
        SELECT 
            MIN(divenumber) as min_dive,
            MAX(divenumber) as max_dive
        FROM public.real_time_binned
        WHERE mission_id = :mission_id
    """
    with engine.connect() as conn:
        result = pd.read_sql_query(text(sql), conn, params={"mission_id": mission_id})
        if len(result) > 0:
            return int(result['min_dive'].iloc[0]), int(result['max_dive'].iloc[0])
        return 1, 10  # default fallback

def get_dives_data(mission_id: str, min_dive: int = None, max_dive: int = None, columns: list = None) -> pd.DataFrame:
    """
    Return columns for divenumbers in the specified range for a given mission_id.
    If min_dive and max_dive are None, returns last 10 dives with valid data.
    Ordered by divenumber DESC, depth ASC.
    
    Parameters:
    -----------
    mission_id : str
        The mission ID to query
    min_dive : int, optional
        Minimum dive number
    max_dive : int, optional
        Maximum dive number
    columns : list, optional
        List of column names to select. If None, selects all columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with requested columns. Returns empty DataFrame if no valid data found.
    """
    engine = create_engine()
    
    # Build column selection
    if columns is None:
        column_str = "*"
    else:
        column_str = ", ".join(columns)
    
    if min_dive is None and max_dive is None:
        # Default behavior: last 10 dives with data
        # Use COALESCE to prioritize dives with non-null critical columns
        sql = f"""
            SELECT {column_str}
            FROM public.real_time_binned
            WHERE mission_id = :mission_id
              AND divenumber IN (
                  SELECT DISTINCT divenumber
                  FROM public.real_time_binned
                  WHERE mission_id = :mission_id
                    AND (lat IS NOT NULL OR lon IS NOT NULL OR depth IS NOT NULL)
                  ORDER BY divenumber DESC
                  LIMIT 10
              )
            ORDER BY divenumber DESC, depth ASC
        """
        params = {"mission_id": mission_id}
    else:
        # Range-based query
        sql = f"""
            SELECT {column_str}
            FROM public.real_time_binned
            WHERE mission_id = :mission_id
              AND divenumber >= :min_dive
              AND divenumber <= :max_dive
            ORDER BY divenumber DESC, depth ASC
        """
        params = {"mission_id": mission_id, "min_dive": min_dive, "max_dive": max_dive}
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn, params=params)
            
            # Return empty DataFrame if no data
            if len(df) == 0:
                return pd.DataFrame()
            
            # Add computed columns safely
            if 'phin' in df.columns and 'phin_canb' in df.columns:
                df['ph-delta'] = df['phin'] - df['phin_canb']
            else:
                df['ph-delta'] = pd.NA
                
            if 'phin_corr' in df.columns and 'phin_canb' in df.columns:
                df['ph-corr-delta'] = df['phin_corr'] - df['phin_canb']
            else:
                df['ph-corr-delta'] = pd.NA
                
            if 'unixtime' in df.columns:
                df['datetime'] = pd.to_datetime(df['unixtime'], unit='s', errors='coerce')
            else:
                df['datetime'] = pd.NaT
                
            return df
    except Exception as e:
        print(f"Error loading dives data: {e}")
        return pd.DataFrame()

def get_map_data(mission_id: str, min_dive: int = None, max_dive: int = None, depth: int = 0, columns: list = None) -> pd.DataFrame:
    """
    Return surface data (depth=0) for divenumbers in the specified range for a given mission_id.
    If min_dive and max_dive are None, returns last 10 dives with valid location data.
    Ordered by divenumber DESC, depth ASC.
    
    Parameters:
    -----------
    mission_id : str
        The mission ID to query
    min_dive : int, optional
        Minimum dive number
    max_dive : int, optional
        Maximum dive number
    depth : int, optional
        Depth to filter by (default: 0 for surface)
    columns : list, optional
        List of column names to select. If None, selects all columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with requested columns. Returns empty DataFrame if no valid data found.
    """
    engine = create_engine()
    
    # Build column selection
    if columns is None:
        column_str = "*"
    else:
        column_str = ", ".join(columns)
    
    if min_dive is None and max_dive is None:
        # Default behavior: last 10 dives with valid location data
        sql = f"""
            SELECT {column_str}
            FROM public.real_time_binned
            WHERE mission_id = :mission_id
            AND divenumber IN (
                SELECT DISTINCT divenumber
                FROM public.real_time_binned
                WHERE mission_id = :mission_id
                  AND depth = :depth
                  AND lat IS NOT NULL 
                  AND lon IS NOT NULL
                ORDER BY divenumber DESC
                LIMIT 10
            )
            AND depth = :depth
            ORDER BY divenumber DESC, depth ASC
        """
        params = {"mission_id": mission_id, "depth": depth}
    else:
        # Range-based query
        sql = f"""
            SELECT {column_str}
            FROM public.real_time_binned
            WHERE mission_id = :mission_id
              AND divenumber >= :min_dive
              AND divenumber <= :max_dive
              AND depth = :depth
            ORDER BY divenumber DESC, depth ASC
        """
        params = {"mission_id": mission_id, "min_dive": min_dive, "max_dive": max_dive, "depth": depth}
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn, params=params)
            
            # Return empty DataFrame if no data
            if len(df) == 0:
                return pd.DataFrame()
            
            # Add computed columns safely
            if 'phin' in df.columns and 'phin_canb' in df.columns:
                df['ph-delta'] = df['phin'] - df['phin_canb']
            else:
                df['ph-delta'] = pd.NA
                
            if 'phin_corr' in df.columns and 'phin_canb' in df.columns:
                df['ph-corr-delta'] = df['phin_corr'] - df['phin_canb']
            else:
                df['ph-corr-delta'] = pd.NA
                
            if 'unixtime' in df.columns:
                df['datetime'] = pd.to_datetime(df['unixtime'], unit='s', errors='coerce')
            else:
                df['datetime'] = pd.NaT
                
            return df
    except Exception as e:
        print(f"Error loading map data: {e}")
        return pd.DataFrame()

def get_ph_drift_data(mission_id: str, min_dive: int = None, max_dive: int = None, depth: int = 450, columns: list = None) -> pd.DataFrame:
    """
    Return rows for a mission where depth equals specified depth (default 450m)
    across all divenumbers, or limited to a range if min/max are provided.
    
    Parameters:
    -----------
    mission_id : str
        The mission ID to query
    min_dive : int, optional
        Minimum dive number
    max_dive : int, optional
        Maximum dive number
    depth : int, optional
        Depth to filter by (default: 450)
    columns : list, optional
        List of column names to select. If None, selects all columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with requested columns. Returns empty DataFrame if no valid data found.
    """
    engine = create_engine()
    
    # Build column selection
    if columns is None:
        column_str = "*"
    else:
        column_str = ", ".join(columns)
    
    if min_dive is None and max_dive is None:
        sql = f"""
            SELECT {column_str}
            FROM public.real_time_binned
            WHERE mission_id = :mission_id
              AND depth = :depth
            ORDER BY divenumber DESC, depth ASC
        """
        params = {"mission_id": mission_id, "depth": depth}
    else:
        sql = f"""
            SELECT {column_str}
            FROM public.real_time_binned
            WHERE mission_id = :mission_id
              AND divenumber >= :min_dive
              AND divenumber <= :max_dive
              AND depth = :depth
            ORDER BY divenumber DESC, depth ASC
        """
        params = {"mission_id": mission_id, "min_dive": min_dive, "max_dive": max_dive, "depth": depth}

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn, params=params)
            
            # Return empty DataFrame if no data
            if len(df) == 0:
                return pd.DataFrame()
            
            # Add computed columns safely
            if 'phin' in df.columns and 'phin_canb' in df.columns:
                df['ph-delta'] = df['phin'] - df['phin_canb']
            else:
                df['ph-delta'] = pd.NA
                
            if 'phin_corr' in df.columns and 'phin_canb' in df.columns:
                df['ph-corr-delta'] = df['phin_corr'] - df['phin_canb']
            else:
                df['ph-corr-delta'] = pd.NA
                
            if 'unixtime' in df.columns:
                df['datetime'] = pd.to_datetime(df['unixtime'], unit='s', errors='coerce')
            else:
                df['datetime'] = pd.NaT
                
            return df
    except Exception as e:
        print(f"Error loading pH drift data: {e}")
        return pd.DataFrame()