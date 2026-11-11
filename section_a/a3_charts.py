"""Section A3: Comparative data visualization with Matplotlib and Seaborn.

Generates hourly revenue multi-bar charts and telemetry correlation heatmaps,
contrasting low-level procedural canvas control with declarative visualization.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

SECTION_B = PROJECT_ROOT / "section_b"  # holds data_pipeline.py
if str(SECTION_B) not in sys.path:
    sys.path.insert(0, str(SECTION_B))

from data_pipeline import prepare_ev_data  # noqa: E402


def plot_bar_matplotlib(unstacked_rev: pd.DataFrame, out_path: Path):
    """Render multi-bar plot of hourly revenue using procedural Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hours, width = np.arange(24), 0.38
    ax.bar(hours - width / 2, unstacked_rev["AC"], width=width, label="AC", color="#1f77b4")
    ax.bar(hours + width / 2, unstacked_rev["DC"], width=width, label="DC", color="#ff7f0e")
    ax.set_title("Hourly EV Revenue by Charger Type (Matplotlib)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Start Hour of Day (0-23)", fontsize=10)
    ax.set_ylabel("Total Net Revenue (MYR)", fontsize=10)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right", fontsize=8)
    ax.legend(title="Charger Type")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar_seaborn(grouped_rev: pd.DataFrame, out_path: Path):
    """Render multi-bar plot of hourly revenue using declarative Seaborn."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=grouped_rev, x="Start_Hour", y="Net_Billable_MYR", hue="Charger_Type",
        palette={"AC": "#1f77b4", "DC": "#ff7f0e"}, ax=ax,
    )
    ax.set_title("Hourly EV Revenue by Charger Type (Seaborn)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Start Hour of Day (0-23)", fontsize=10)
    ax.set_ylabel("Total Net Revenue (MYR)", fontsize=10)
    ax.legend(title="Charger Type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_matplotlib(corr: pd.DataFrame, out_path: Path):
    """Render correlation heatmap using Matplotlib imshow and text annotations."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(False)
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=25, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")
    ax.set_title("Telemetry Correlation Heatmap (Matplotlib)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Telemetry Metrics", fontsize=10)
    ax.set_ylabel("Telemetry Metrics", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_seaborn(corr: pd.DataFrame, out_path: Path):
    """Render correlation heatmap with declarative annotations using Seaborn."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
        cbar_kws={"label": "Pearson Correlation"}, ax=ax,
    )
    ax.set_title("Telemetry Correlation Heatmap (Seaborn)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Telemetry Metrics", fontsize=10)
    ax.set_ylabel("Telemetry Metrics", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Load data, compute metrics, print correlations, and save figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ev_df = prepare_ev_data()
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading prepared EV data: {err}", file=sys.stderr)
        raise

    grouped = (
        ev_df.groupby(["Start_Hour", "Charger_Type"], as_index=False)["Net_Billable_MYR"]
        .sum()
    )
    unstacked = (
        grouped.pivot(index="Start_Hour", columns="Charger_Type", values="Net_Billable_MYR")
        .reindex(range(24), fill_value=0.0)
    )

    corr_cols = ["Battery_Temp_C", "Duration_Mins", "Power_Rate_kW", "Energy_kWh"]
    corr_matrix = ev_df[corr_cols].corr().round(4)
    print("=" * 60 + "\nTelemetry Pearson Correlation Matrix\n" + "=" * 60)
    print(corr_matrix)

    plot_bar_matplotlib(unstacked, FIGURES_DIR / "a3_bar_matplotlib.png")
    plot_bar_seaborn(grouped, FIGURES_DIR / "a3_bar_seaborn.png")
    plot_heatmap_matplotlib(corr_matrix, FIGURES_DIR / "a3_heatmap_matplotlib.png")
    plot_heatmap_seaborn(corr_matrix, FIGURES_DIR / "a3_heatmap_seaborn.png")
    print(f"\nSuccessfully generated 4 charts in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
