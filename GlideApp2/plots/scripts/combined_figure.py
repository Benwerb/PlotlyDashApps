import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sqlalchemy import text
from database_tools import create_engine

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
    land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    land_geom = unary_union(list(shpreader.Reader(land_shp).geometries()))
    mask = [not land_geom.contains(Point(lon, lat)) for lon, lat in zip(df['lon'], df['lat'])]
    filtered = df[mask]
    n_removed = len(df) - len(filtered)
    if n_removed:
        print(f"Filtered out {n_removed} land points ({len(filtered)} remain)")
    return filtered


def query_map_data():
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


def query_ph_data(start='2025-04-01', end='2026-05-01'):
    engine = create_engine()
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    sql = f"""
        SELECT unixtime, mission_id, pHin, lon
        FROM public.real_time_binned
        WHERE pHin BETWEEN 6.5 AND 9
          AND unixtime >= {start_ts}
          AND unixtime < {end_ts}
        ORDER BY unixtime ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df


def compute_monthly_mission_stack(df: pd.DataFrame, start='2025-04-01', end='2026-04-01'):
    if df is None or df.empty:
        all_months = pd.date_range(start=start, end=end, freq="MS")
        return pd.DataFrame(index=all_months)

    df["datetime"] = pd.to_datetime(df["unixtime"], unit="s")
    df["date"] = pd.to_datetime(df["datetime"].dt.date)

    presence = df[["date", "mission_id"]].drop_duplicates()
    presence["present"] = 1

    daily_presence = presence.pivot_table(
        index="date", columns="mission_id", values="present", fill_value=0, aggfunc="sum"
    )
    daily_presence["month"] = daily_presence.index.to_period("M").to_timestamp()
    monthly = daily_presence.groupby("month").sum()

    all_months = pd.date_range(start=start, end=end, freq="MS")
    monthly = monthly.reindex(all_months, fill_value=0)
    monthly.columns = [str(c) for c in monthly.columns]

    return monthly


def create_combined_figure(map_df: pd.DataFrame, ph_df: pd.DataFrame,
                           monthly_df: pd.DataFrame, output_file: str,
                           total_glider_days: int = None):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.suptitle("US glider deployments", fontsize=15, y=1.01)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1, 1], wspace=0.05)

    # --- Left: bar chart ---
    ax_bar = fig.add_subplot(gs[0, 0])

    months = monthly_df.index
    x = np.arange(len(months))
    bottoms = np.zeros(len(months))
    mission_order = monthly_df.sum(axis=0).sort_values(ascending=False).index.tolist()

    first_lon = ph_df.groupby('mission_id')['lon'].first()
    legend_handles = {}

    for mission in mission_order:
        vals = monthly_df[mission].values
        institution = mission_institution(mission, first_lon[mission]) if mission in first_lon.index else 'SIO'
        color = INSTITUTION_COLORS[institution]
        ax_bar.bar(x, vals, bottom=bottoms, color=color)
        bottoms = bottoms + vals
        if institution not in legend_handles:
            legend_handles[institution] = mpatches.Patch(color=color, label=institution)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([ts.strftime("%b %Y") for ts in months], rotation=45, ha="right", fontsize=11)
    ax_bar.set_ylabel("Total glider-days", fontsize=13)
    total_label = f" (total: {total_glider_days:,})" if total_glider_days is not None else ""
    ax_bar.set_title(f"pH Glider Days Per Month{total_label}", fontsize=12)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.tick_params(axis='y', labelsize=11)

    ax_bar.legend(
        handles=list(legend_handles.values()),
        bbox_to_anchor=(0, 1.01),
        loc='lower left',
        fontsize=12,
        frameon=False,
        handletextpad=0.5,
        ncol=len(legend_handles),
        columnspacing=1.0,
    )

    # --- Center: Pacific map (SIO + OSU) ---
    plot_kwargs = dict(s=8, alpha=0.6, edgecolors='none', zorder=10,
                       transform=ccrs.PlateCarree())

    ax_west = fig.add_subplot(gs[0, 1], projection=ccrs.Mercator())
    ax_west.add_image(cimgt.GoogleTiles(url=OCEAN_TILES_URL), 6)
    ax_west.set_extent([-131, -114, 30, 50], crs=ccrs.PlateCarree())
    ax_west.set_title("West Coast", fontsize=12)

    # --- Right: Atlantic map (WHOI) ---
    ax_east = fig.add_subplot(gs[0, 2], projection=ccrs.Mercator())
    ax_east.add_image(cimgt.GoogleTiles(url=OCEAN_TILES_URL), 6)
    ax_east.set_extent([-82, -65, 26, 46], crs=ccrs.PlateCarree())
    ax_east.set_title("East Coast", fontsize=12)

    for mission in map_df['mission_id'].unique():
        mdf = map_df[map_df['mission_id'] == mission]
        if mdf.empty:
            continue
        institution = mission_institution(mission, mdf['lon'].mean())
        color = INSTITUTION_COLORS[institution]
        ax = ax_east if institution == 'WHOI' else ax_west
        ax.scatter(mdf['lon'], mdf['lat'], c=color, marker='o', **plot_kwargs)

    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Re-query the database')
    args = parser.parse_args()

    cache_path = os.path.normpath(CACHE_FILE)
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(out_dir, exist_ok=True)

    if not args.refresh and os.path.exists(cache_path):
        print(f"Loading map cache from {cache_path}")
        map_df = pd.read_parquet(cache_path)
        print(f"Loaded {len(map_df)} rows")
    else:
        map_df = query_map_data()
        if len(map_df) > 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            map_df.to_parquet(cache_path, index=False)
            print(f"Cached to {cache_path}")

    print("Querying pH data for bar chart...")
    ph_df = query_ph_data(start='2025-04-01', end='2026-05-01')
    print(f"Retrieved {len(ph_df)} rows")

    map_df = map_df[~map_df['mission_id'].isin(EXCLUDED_MISSIONS)]
    ph_df = ph_df[~ph_df['mission_id'].isin(EXCLUDED_MISSIONS)]

    print("Filtering land points...")
    map_df = filter_ocean_points(map_df)

    monthly_df = compute_monthly_mission_stack(ph_df, start='2025-04-01', end='2026-04-01')
    if monthly_df.empty:
        print("No monthly data computed.")
    else:
        total_glider_days = int(monthly_df.values.sum())
        print(f"Total glider-days (Apr 2025 – Apr 2026): {total_glider_days}")
        output_file = os.path.join(out_dir, 'combined_figure.png')
        create_combined_figure(map_df, ph_df, monthly_df, output_file, total_glider_days=total_glider_days)
