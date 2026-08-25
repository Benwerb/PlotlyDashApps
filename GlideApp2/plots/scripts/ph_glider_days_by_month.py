import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sqlalchemy import text
from database_tools import create_engine

OSU_MISSIONS = {'25B68601'}
EXCLUDED_MISSIONS = {'25720901', '25821101', '25821001'}
EAST_COAST_LON_MAX = -85
INSTITUTION_COLORS = {'SIO': '#1f77b4', 'WHOI': '#d62728', 'OSU': '#2ca02c'}

def mission_institution(mission_id, first_lon):
    if mission_id in OSU_MISSIONS:
        return 'OSU'
    return 'WHOI' if first_lon >= EAST_COAST_LON_MAX else 'SIO'


def query_good_ph(start='2025-04-01', end='2026-05-01'):
    engine = create_engine()

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    sql = f"""
        SELECT
            unixtime,
            mission_id,
            pHin,
            lon
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
    """
    Compute total glider-days per month for each mission.

    For each calendar day, a mission contributes 1 if it has any good-pH record that day.
    For each month, sum those daily counts per mission.
    Returns a DataFrame indexed by month (Timestamp) with columns per `mission_id`.
    """
    if df is None or df.empty:
        all_months = pd.date_range(start=start, end=end, freq="MS")
        return pd.DataFrame(index=all_months)

    df["datetime"] = pd.to_datetime(df["unixtime"], unit="s")
    df["date"] = pd.to_datetime(df["datetime"].dt.date)

    # presence: unique mission per date
    presence = df[["date", "mission_id"]].drop_duplicates()
    presence["present"] = 1

    # pivot to have one column per mission, rows = dates, values 0/1 for presence
    daily_presence = presence.pivot_table(index="date", columns="mission_id", values="present", fill_value=0, aggfunc="sum")

    # convert index to month period and sum days per month
    daily_presence["month"] = daily_presence.index.to_period("M").to_timestamp()
    monthly = daily_presence.groupby("month").sum()

    # Ensure months for full range are present
    all_months = pd.date_range(start=start, end=end, freq="MS")
    monthly = monthly.reindex(all_months, fill_value=0)

    # columns to strings for safe plotting/legend
    monthly.columns = [str(c) for c in monthly.columns]

    return monthly


def plot_monthly_stacked(monthly_df: pd.DataFrame, df_raw: pd.DataFrame,
                         output_file: str = "plots/outputs/ph_glider_days_by_month_2025_2026_stacked.png"):
    if monthly_df is None or monthly_df.empty:
        print("No data to plot.")
        return

    import matplotlib.patches as mpatches

    # Build mission -> institution and start date labels
    first_lon = df_raw.groupby('mission_id')['lon'].first()

    fig, ax = plt.subplots(figsize=(14, 7))
    months = monthly_df.index
    x = np.arange(len(months))

    bottoms = np.zeros(len(months))

    mission_order = monthly_df.sum(axis=0).sort_values(ascending=False).index.tolist()

    legend_handles = {}
    for mission in mission_order:
        vals = monthly_df[mission].values
        institution = mission_institution(mission, first_lon[mission]) if mission in first_lon.index else 'SIO'
        color = INSTITUTION_COLORS[institution]
        ax.bar(x, vals, bottom=bottoms, color=color)
        bottoms = bottoms + vals

        if institution not in legend_handles:
            legend_handles[institution] = mpatches.Patch(color=color, label=institution)

    ax.set_xticks(x)
    ax.set_xticklabels([ts.strftime("%b %Y") for ts in months], rotation=45, ha="right", fontsize=20)
    ax.set_ylabel("Total glider-days", fontsize=22)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(handles=list(legend_handles.values()), bbox_to_anchor=(1.05, 1),
              loc="upper left", fontsize=20, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, transparent=True)
    plt.close()
    print(f"Saved stacked bar chart to: {output_file}")


if __name__ == "__main__":
    print("Querying database for good pH points (6.5–9) Apr 2025–Apr 2026...")
    df = query_good_ph(start='2025-04-01', end='2026-05-01')
    print(f"Retrieved {len(df)} rows")

    df = df[~df['mission_id'].isin(EXCLUDED_MISSIONS)]
    monthly_df = compute_monthly_mission_stack(df, start='2025-04-01', end='2026-04-01')
    if monthly_df.empty:
        print("No monthly data computed.")
    else:
        print("Monthly contributions (first 10 missions):")
        print(monthly_df.iloc[:, :10].head(12))

    plot_monthly_stacked(monthly_df, df)
