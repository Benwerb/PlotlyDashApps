import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from sqlalchemy import text
from database_tools import create_engine
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union

OCEAN_TILES_URL = (
    'https://server.arcgisonline.com/ArcGIS/rest/services'
    '/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}.jpg'
)
OSU_MISSIONS = {'25B68601'}
EXCLUDED_MISSIONS = {'25720901', '25821101', '25821001'}
EAST_COAST_LON_MIN = -85
INSTITUTION_COLORS = {'SIO': '#1f77b4', 'WHOI': '#d62728', 'OSU': '#2ca02c'}
CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'map_data_cache.parquet')


def mission_institution(mission_id, mission_mean_lon):
    if mission_id in OSU_MISSIONS:
        return 'OSU'
    return 'WHOI' if mission_mean_lon >= EAST_COAST_LON_MIN else 'SIO'


def filter_ocean_points(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where lat/lon falls on land."""
    land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    land_geom = unary_union(list(shpreader.Reader(land_shp).geometries()))
    mask = [not land_geom.contains(Point(lon, lat)) for lon, lat in zip(df['lon'], df['lat'])]
    filtered = df[mask]
    n_removed = len(df) - len(filtered)
    if n_removed:
        print(f"Filtered out {n_removed} land points ({len(filtered)} remain)")
    return filtered


def query_all_lat_lon():
    engine = create_engine()
    sql = """
        SELECT lat, lon, pHin, mission_id, divenumber, depth, unixtime
        FROM public.real_time_binned
        WHERE lat IS NOT NULL AND lon IS NOT NULL AND pHin BETWEEN 6.5 AND 9
        ORDER BY unixtime ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    print(f"Retrieved {len(df)} rows, {df['mission_id'].nunique()} missions")
    return df


def create_panel_map(df: pd.DataFrame, lon_min, lon_max, lat_min, lat_max,
                     title: str, output_file: str, legend: bool = True,
                     figsize: tuple = (14, 7)):
    missions = list(df['mission_id'].unique()) if 'mission_id' in df.columns else None

    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.Mercator())
    ax.add_image(cimgt.GoogleTiles(url=OCEAN_TILES_URL), 6)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    plot_kwargs = dict(s=12, alpha=0.6, edgecolors='none', zorder=10,
                       transform=ccrs.PlateCarree())

    panel_df = df[(df['lon'] >= lon_min) & (df['lon'] <= lon_max)]

    legend_handles = {}
    if missions:
        for mission in missions:
            mdf = panel_df[panel_df['mission_id'] == mission]
            if mdf.empty:
                continue
            institution = mission_institution(mission, mdf['lon'].mean())
            color = INSTITUTION_COLORS[institution]
            ax.scatter(mdf['lon'], mdf['lat'], c=color, marker='o', **plot_kwargs)
            if institution not in legend_handles:
                legend_handles[institution] = plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=color, markersize=10, label=institution
                )
    else:
        ax.scatter(panel_df['lon'], panel_df['lat'], c='blue', **plot_kwargs)

    if legend and legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc='upper right',
                  fontsize=12, handletextpad=0.5)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Re-query the database')
    args = parser.parse_args()

    cache_path = os.path.normpath(CACHE_FILE)

    if not args.refresh and os.path.exists(cache_path):
        print(f"Loading cache from {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"Loaded {len(df)} rows")
    else:
        df = query_all_lat_lon()
        if len(df) > 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path, index=False)
            print(f"Cached to {cache_path}")

    if len(df) == 0:
        print("No data found.")
    else:
        df = df[~df['mission_id'].isin(EXCLUDED_MISSIONS)]
        print("Filtering land points...")
        df = filter_ocean_points(df)

        out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
        os.makedirs(out_dir, exist_ok=True)

        create_panel_map(df, -130, -65, 22, 50, "US Glider Deployments",
                         os.path.join(out_dir, "static_map.png"))

        create_panel_map(df, -130, -105, 28, 50, "West Coast Deployments",
                         os.path.join(out_dir, "static_map_west.png"), legend=False,
                         figsize=(16/3, 9))

        create_panel_map(df, -85, -60, 24, 46, "East Coast Deployments",
                         os.path.join(out_dir, "static_map_east.png"), legend=False,
                         figsize=(16/3, 9))
