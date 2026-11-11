"""Visualisation module for the Smart Campus Energy & EV Fleet platform (Section B2).

Builds three panels from the cleaned charging sessions:
1. hourly grid demand against a modelled solar generation curve,
2. session cost by user type (box plot),
3. battery temperature against energy delivered, with trendlines.
Each panel is saved on its own and as one combined figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from data_pipeline import prepare_ev_data  # noqa: E402

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
SOLAR_CAPACITY_KW = 50.0  # assumed rooftop PV array size
SUNRISE, SUNSET = 7, 19
CHARGER_COLOURS = {"AC": "#2a9d8f", "DC": "#e76f51"}
USER_ORDER = ["Student", "Staff", "Visitor"]


def hourly_grid_demand(ev):
    """Spread each session's energy over the clock hours it covers and sum per hour (kWh)."""
    demand = np.zeros(24)
    for start, minutes, energy in ev[["Start_Hour", "Duration_Mins", "Energy_kWh"]].to_numpy():
        minute_index = np.arange(int(round(minutes)))
        clock_hours = ((start * 60 + minute_index) // 60 % 24).astype(int)
        np.add.at(demand, clock_hours, energy / len(minute_index))
    return demand


def modelled_solar_curve(hours):
    """Return a half-sine solar output (kW) between sunrise and sunset, zero at night."""
    daylight = (hours >= SUNRISE) & (hours < SUNSET)
    phase = np.pi * (hours + 0.5 - SUNRISE) / (SUNSET - SUNRISE)
    return np.where(daylight, SOLAR_CAPACITY_KW * np.sin(phase), 0.0)


def plot_demand_vs_solar(ev, ax):
    """Panel 1: time-series line plot of grid demand and the solar curve."""
    hours = np.arange(24)
    demand = hourly_grid_demand(ev)
    ax.plot(hours, demand, marker="o", color="#264653", label="Grid demand from EV charging")
    ax.axvspan(12, 17, color="#f4a261", alpha=0.18, label="Peak window 12:00-17:00")
    peak_hour = int(demand.argmax())
    ax.annotate(f"Peak {demand[peak_hour]:.0f} kWh", (peak_hour, demand[peak_hour]),
                xytext=(0.5, demand[peak_hour] * 0.88),
                arrowprops={"arrowstyle": "->", "color": "#264653"})
    ax.set(xlabel="Hour of day", ylabel="Energy drawn (kWh, all sessions)",
           title="Panel 1: Hourly grid demand vs modelled solar generation", xticks=range(0, 24, 2))
    solar_ax = ax.twinx()
    solar_ax.plot(hours, modelled_solar_curve(hours), linestyle="--", color="#e9c46a",
                  label=f"Modelled solar output ({SOLAR_CAPACITY_KW:.0f} kWp)")
    solar_ax.set_ylabel("Solar output (kW)")
    solar_ax.grid(False)
    lines = ax.get_legend_handles_labels()
    solar_lines = solar_ax.get_legend_handles_labels()
    ax.legend(lines[0] + solar_lines[0], lines[1] + solar_lines[1], loc="upper right", fontsize=8)


def plot_cost_by_user(ev, ax):
    """Panel 2: box plot of net billable amount for each user type."""
    sns.boxplot(data=ev, x="User_Type", y="Net_Billable_MYR", hue="User_Type", order=USER_ORDER,
                palette="Set2", legend=False, ax=ax)
    medians = ev.groupby("User_Type")["Net_Billable_MYR"].median()
    for position, user in enumerate(USER_ORDER):
        ax.text(position, medians[user], f"median {medians[user]:.2f}", ha="center",
                va="bottom", fontsize=8)
    ax.set(xlabel="User type", ylabel="Net billable amount (MYR)",
           title="Panel 2: Charging session cost by user type")


def plot_temp_vs_energy(ev, ax):
    """Panel 3: scatter of battery temperature vs energy with a trendline per charger type."""
    for charger, group in ev.groupby("Charger_Type"):
        colour = CHARGER_COLOURS[charger]
        ax.scatter(group["Battery_Temp_C"], group["Energy_kWh"], s=12, alpha=0.45,
                   color=colour, label=f"{charger} sessions")
        slope, intercept = np.polyfit(group["Battery_Temp_C"], group["Energy_kWh"], 1)
        x_line = np.linspace(group["Battery_Temp_C"].min(), group["Battery_Temp_C"].max(), 50)
        ax.plot(x_line, slope * x_line + intercept, color=colour, linewidth=2,
                label=f"{charger} trend ({slope:+.2f} kWh/°C)")
    ax.axvline(45, color="grey", linestyle=":", label="45 °C caution line")
    ax.set(xlabel="Battery temperature (°C)", ylabel="Energy delivered (kWh)",
           title="Panel 3: Battery temperature vs energy delivered")
    ax.legend(fontsize=8)


PANELS = [
    ("b2_panel1_demand_vs_solar.png", plot_demand_vs_solar),
    ("b2_panel2_cost_by_user.png", plot_cost_by_user),
    ("b2_panel3_temp_vs_energy.png", plot_temp_vs_energy),
]


def save_figures(ev):
    """Save each panel on its own and all three together in one figure."""
    FIGURE_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    for file_name, draw in PANELS:
        fig, ax = plt.subplots(figsize=(9, 5))
        draw(ev, ax)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / file_name, dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ax, (_, draw) in zip(axes, PANELS):
        draw(ev, ax)
    fig.suptitle("Smart Campus EV Charging - Visual Summary", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "b2_combined_dashboard.png", dpi=200)
    plt.close(fig)


def main():
    """Prepare the data, draw the panels and report where they were saved."""
    try:
        ev = prepare_ev_data()
        save_figures(ev)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Visualisation stopped: {error}")
        raise SystemExit(1) from error
    demand = hourly_grid_demand(ev)
    print(f"Busiest grid hour: {int(demand.argmax())}:00 ({demand.max():.1f} kWh)")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
