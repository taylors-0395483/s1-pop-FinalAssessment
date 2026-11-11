"""Multipage dashboard for campus charging sessions and retail energy sales."""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime

os.environ.setdefault("MPLBACKEND", "Agg")
# Make data_pipeline.py (same folder) importable whatever the working directory is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

from data_pipeline import (  # noqa: E402
    OVERSTAY_LIMIT_MINS,
    TEMP_LIMIT_C,
    prepare_ev_data,
    prepare_sales_data,
)

TEMP_KEY = "alert_temp_threshold"
EV_COLUMNS = (
    "Charger_Type", "User_Type", "Start_Hour", "Energy_kWh", "Overstay_Mins",
    "Battery_Temp_C", "Carbon_Savings_kgCO2", "Net_Billable_MYR",
)
SALES_COLUMNS = ("Transaction_Date", "Total_Revenue_MYR")


def require_frame(frame, columns, loader_name):
    """Return a copy of a pipeline table, or raise when the shape is wrong."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{loader_name} did not return a DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{loader_name} is missing columns: {', '.join(missing)}")
    return frame.copy()


@st.cache_data(show_spinner=False)
def load_ev_frame():
    """Load cleaned charging sessions from the data pipeline."""
    return require_frame(prepare_ev_data(), EV_COLUMNS, "prepare_ev_data()")


@st.cache_data(show_spinner=False)
def load_sales_frame():
    """Load cleaned sales transactions from the data pipeline."""
    return require_frame(prepare_sales_data(), SALES_COLUMNS, "prepare_sales_data()")


def load_dashboard_data():
    """Load both tables, or stop the app when the pipeline cannot be read."""
    try:
        return load_ev_frame(), load_sales_frame()
    except Exception as exc:
        st.error(f"Could not load the dashboard data: {exc}")
        st.stop()


def transaction_days(sales):
    """Return each sale's calendar day at midnight."""
    parsed = pd.to_datetime(sales["Transaction_Date"], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def date_bounds(value, first, last):
    """Return an inclusive start and end from a date-range widget."""
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value, value
    chosen = []
    for item in value or ():
        if item is not None:
            chosen.append(item.date() if isinstance(item, datetime) else item)
    if len(chosen) >= 2:
        start, end = chosen[0], chosen[1]
        return (end, start) if start > end else (start, end)
    if len(chosen) == 1:
        return chosen[0], chosen[0]
    return first, last


def ordered_labels(values, preferred):
    """List categories with the preferred names first, then any extras."""
    present = {str(item) for item in pd.Series(values).dropna().unique()}
    ordered = [label for label in preferred if label in present]
    return ordered + sorted(present.difference(ordered))


def numeric_sum(frame, column):
    """Sum a column, treating non-numeric values as zero."""
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def percent_label(current, baseline):
    """Format the percentage change from baseline to current."""
    if baseline == 0 or not math.isfinite(baseline) or not math.isfinite(current):
        return None
    return f"{(current - baseline) / baseline * 100:+.1f}%"


def mean_delta(filtered, overall):
    """Percent change of the filtered per-session mean versus the full mean."""
    current = pd.to_numeric(filtered, errors="coerce").dropna()
    base = pd.to_numeric(overall, errors="coerce").dropna()
    if current.empty or base.empty:
        return None
    return percent_label(float(current.mean()), float(base.mean()))


def sales_between(sales, start, end):
    """Keep sales inside an inclusive calendar-date range."""
    days = transaction_days(sales)
    mask = days.ge(pd.Timestamp(start)) & days.le(pd.Timestamp(end))
    return sales.loc[mask.fillna(False)].copy()


def apply_filters(sessions, sales, selection):
    """Apply the sidebar selection to charging sessions and sales."""
    energy = pd.to_numeric(sessions["Energy_kWh"], errors="coerce")
    chosen = set(selection["charger_types"])
    mask = energy.ge(selection["minimum_energy"])
    mask &= sessions["Charger_Type"].astype(str).isin(chosen)
    filtered_sessions = sessions.loc[mask.fillna(False)].copy()
    filtered_sales = sales_between(sales, selection["start"], selection["end"])
    return filtered_sessions, filtered_sales


def render_sidebar(sessions, sales):
    """Draw the sales and charging filters and return the selected values."""
    days = transaction_days(sales).dropna()
    if days.empty:
        st.error("Sales data has no valid transaction dates.")
        st.stop()
    first_day, last_day = days.min().date(), days.max().date()
    st.sidebar.subheader("Sales filters")
    raw_range = st.sidebar.date_input(
        "Date range", value=(first_day, last_day), min_value=first_day,
        max_value=last_day, key="sales_date_range",
    )
    observed_max = pd.to_numeric(sessions["Energy_kWh"], errors="coerce").max()
    max_energy = 1.0 if pd.isna(observed_max) or float(observed_max) <= 0 else float(observed_max)
    st.sidebar.subheader("Charging session filters")
    minimum_energy = st.sidebar.slider(
        "Minimum energy (kWh)", min_value=0.0, max_value=max_energy, value=0.0,
        step=0.1, key="minimum_energy_kwh", format="%.1f",
    )
    options = ordered_labels(sessions["Charger_Type"], ("AC", "DC"))
    selected = st.sidebar.multiselect(
        "Charger type", options=options, default=options, key="charger_types",
    )
    start, end = date_bounds(raw_range, first_day, last_day)
    return {
        "start": start, "end": end, "minimum_energy": float(minimum_energy),
        "charger_types": tuple(selected),
    }


def load_filtered_view():
    """Load both datasets, draw the sidebar, and apply the current filters."""
    sessions, sales = load_dashboard_data()
    selection = render_sidebar(sessions, sales)
    filtered_sessions, filtered_sales = apply_filters(sessions, sales, selection)
    return sessions, sales, filtered_sessions, filtered_sales, selection


def alert_mask(sessions, threshold):
    """Mark sessions above the temperature threshold that also overstay."""
    temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
    overstay = pd.to_numeric(sessions["Overstay_Mins"], errors="coerce")
    return temperature.gt(threshold) & overstay.gt(OVERSTAY_LIMIT_MINS)


def render_kpi_header(sessions, filtered_sessions, sales, filtered_sales):
    """Show the four executive metrics shared by every page."""
    revenue_total = numeric_sum(filtered_sales, "Total_Revenue_MYR")
    revenue_total += numeric_sum(filtered_sessions, "Net_Billable_MYR")
    threshold = float(st.session_state.get(TEMP_KEY, TEMP_LIMIT_C))
    alerts = int(alert_mask(filtered_sessions, threshold).fillna(False).sum())
    baseline = int(alert_mask(filtered_sessions, TEMP_LIMIT_C).fillna(False).sum())
    # Value deltas compare the filtered average per record with the full dataset;
    # the alert delta compares the current threshold with the default 40 °C rule.
    with st.container(border=True):
        energy, carbon, revenue, alert_card = st.columns(4)
        energy.metric(
            "Total Energy Delivered (kWh)",
            f"{numeric_sum(filtered_sessions, 'Energy_kWh'):,.2f}",
            delta=mean_delta(filtered_sessions["Energy_kWh"], sessions["Energy_kWh"]),
            help="Delta: average energy per filtered session vs all sessions.",
        )
        carbon.metric(
            "Carbon Offset (kg CO2)",
            f"{numeric_sum(filtered_sessions, 'Carbon_Savings_kgCO2'):,.2f}",
            delta=mean_delta(
                filtered_sessions["Carbon_Savings_kgCO2"], sessions["Carbon_Savings_kgCO2"],
            ),
            help="Delta: average carbon saving per filtered session vs all sessions.",
        )
        revenue.metric(
            "Total Revenue (MYR)",
            f"{revenue_total:,.2f}",
            delta=mean_delta(filtered_sales["Total_Revenue_MYR"], sales["Total_Revenue_MYR"]),
            help=(
                "Filtered sales revenue plus filtered charging Net Billable. "
                "Delta: average sales transaction value vs all transactions."
            ),
        )
        alert_card.metric(
            "Active Alert Count",
            f"{alerts:,}",
            delta=alerts - baseline,
            delta_color="inverse",
            help=(
                "Sessions above the alert temperature with more than 20 minutes overstay. "
                "Delta: change versus the default 40 °C threshold."
            ),
        )


def style_axis(axis, xlabel, ylabel, title):
    """Label an axes and hide the top and right spines."""
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axis.set_axisbelow(True)


@contextmanager
def saved_figure(figsize):
    """Yield a Matplotlib axes, then render it in Streamlit and close it."""
    figure, axis = plt.subplots(figsize=figsize)
    try:
        yield axis
        figure.tight_layout()
        st.pyplot(figure)
    finally:
        plt.close(figure)


def draw_hourly_demand(sessions):
    """Draw total energy delivered by charging start hour."""
    usable = pd.DataFrame({
        "hour": pd.to_numeric(sessions["Start_Hour"], errors="coerce"),
        "energy": pd.to_numeric(sessions["Energy_kWh"], errors="coerce"),
    }).dropna()
    if usable.empty:
        st.info("No charging sessions match the current filters.")
        return
    grouped = usable.groupby(usable["hour"].astype(int))["energy"].sum().sort_index()
    if grouped.index.min() >= 0 and grouped.index.max() <= 23:
        grouped = grouped.reindex(range(24), fill_value=0.0)
    with saved_figure((6.6, 3.7)) as axis:
        axis.plot(grouped.index, grouped.to_numpy(), color="#0F6E56", marker="o", linewidth=1.8)
        style_axis(axis, "Start hour", "Energy delivered (kWh)", "Energy demand by start hour")
        axis.set_ylim(bottom=0)


def draw_average_billable(sessions):
    """Draw average net billable amount by charger type and user type."""
    plot_data = sessions.copy()
    plot_data["Net_Billable_MYR"] = pd.to_numeric(plot_data["Net_Billable_MYR"], errors="coerce")
    plot_data = plot_data.dropna(subset=["Charger_Type", "User_Type", "Net_Billable_MYR"])
    if plot_data.empty:
        st.info("No charging sessions match the current filters.")
        return
    with saved_figure((6.6, 3.7)) as axis:
        sns.barplot(
            data=plot_data, x="Charger_Type", y="Net_Billable_MYR", hue="User_Type",
            order=ordered_labels(plot_data["Charger_Type"], ("AC", "DC")),
            hue_order=ordered_labels(plot_data["User_Type"], ("Student", "Staff", "Visitor")),
            errorbar=None, ax=axis,
        )
        style_axis(
            axis, "Charger type", "Average net billable (MYR)",
            "Average net billable by charger and user",
        )
        # Extra headroom keeps the legend clear of the tallest bar.
        axis.set_ylim(top=axis.get_ylim()[1] * 1.3)
        axis.legend(title="User type", frameon=False, ncol=3, loc="upper left")


def draw_sales_revenue(sales):
    """Draw filtered sales revenue totaled by transaction date."""
    usable = pd.DataFrame({
        "day": transaction_days(sales),
        "revenue": pd.to_numeric(sales["Total_Revenue_MYR"], errors="coerce"),
    }).dropna()
    if usable.empty:
        st.info("No sales transactions match the current filters.")
        return
    daily = usable.groupby("day")["revenue"].sum().sort_index()
    with saved_figure((10.5, 3.6)) as axis:
        axis.plot(daily.index, daily.to_numpy(), color="#175CD3", marker="o", linewidth=1.8)
        style_axis(axis, "Transaction date", "Revenue (MYR)", "Sales revenue by date")
        axis.set_ylim(bottom=0)
        axis.tick_params(axis="x", labelrotation=30)


def page_executive_overview():
    """Executive overview with shared KPIs and three filtered charts."""
    sessions, sales, filtered_sessions, filtered_sales, _ = load_filtered_view()
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)
    st.header("Executive Overview")
    left, right = st.columns(2)
    for panel, title, drawer, frame in (
        (left, "hourly energy demand", draw_hourly_demand, filtered_sessions),
        (right, "average net billable", draw_average_billable, filtered_sessions),
    ):
        with panel:
            try:
                drawer(frame)
            except Exception as exc:
                st.error(f"Could not draw {title}: {exc}")
    try:
        draw_sales_revenue(filtered_sales)
    except Exception as exc:
        st.error(f"Could not draw sales revenue: {exc}")


def render_alert_banners(sessions, threshold):
    """Show an error, warning, or success banner for the current threshold."""
    temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
    hot = temperature.gt(threshold).fillna(False)
    risky = alert_mask(sessions, threshold).fillna(False)
    risky_count, warm_count, total = int(risky.sum()), int((hot & ~risky).sum()), len(sessions)
    risky_share = 0.0 if total == 0 else risky_count / total * 100
    warm_share = 0.0 if total == 0 else warm_count / total * 100
    limit = OVERSTAY_LIMIT_MINS
    if risky_count:
        st.error(
            f"{risky_count:,} high-risk sessions ({risky_share:.1f}% of filtered sessions). "
            f"Battery temperature is above {threshold:.1f}°C and overstay is above {limit} minutes."
        )
    if warm_count:
        st.warning(
            f"{warm_count:,} sessions are above {threshold:.1f}°C without an overstay above "
            f"{limit} minutes ({warm_share:.1f}% of filtered sessions)."
        )
    if not risky_count and not warm_count:
        st.success(f"No filtered sessions are above the {threshold:.1f}°C temperature threshold.")


def draw_temperature_scatter(sessions, threshold):
    """Plot battery temperature against overstay and mark the threshold."""
    temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
    overstay = pd.to_numeric(sessions["Overstay_Mins"], errors="coerce")
    valid = temperature.notna() & overstay.notna()
    risky = valid & alert_mask(sessions, threshold).fillna(False)
    with saved_figure((8.8, 4.3)) as axis:
        if (valid & ~risky).any():
            axis.scatter(
                overstay[valid & ~risky], temperature[valid & ~risky], s=18, alpha=0.7,
                c="#98A2B3", linewidths=0, label="Other sessions",
            )
        if risky.any():
            axis.scatter(
                overstay[risky], temperature[risky], s=28, alpha=0.9,
                c="#B42318", linewidths=0, label="High-risk sessions",
            )
        axis.axhline(
            threshold, color="#B42318", linestyle="--", linewidth=1.2,
            label=f"Temperature threshold ({threshold:.1f}°C)",
        )
        axis.axvline(
            OVERSTAY_LIMIT_MINS, color="#667085", linestyle=":", linewidth=1.1,
            label=f"Overstay limit ({OVERSTAY_LIMIT_MINS:g} min)",
        )
        style_axis(axis, "Overstay (minutes)", "Battery temperature (°C)",
                   "Battery temperature vs overstay")
        temps = temperature[valid]
        stays = overstay[valid]
        ymin = min(threshold, float(temps.min()) if valid.any() else threshold)
        ymax = max(threshold, float(temps.max()) if valid.any() else threshold)
        xmin = min(0, float(stays.min()) if valid.any() else 0)
        xmax = max(OVERSTAY_LIMIT_MINS, float(stays.max()) if valid.any() else OVERSTAY_LIMIT_MINS)
        axis.set_ylim(ymin - max((ymax - ymin) * 0.08, 1), ymax + max((ymax - ymin) * 0.08, 1))
        axis.set_xlim(xmin - max((xmax - xmin) * 0.08, 1), xmax + max((xmax - xmin) * 0.08, 1))
        axis.legend(frameon=False, fontsize=8)


def page_alert_center():
    """Alert center for over-temperature and high-risk overstay sessions."""
    sessions, sales, filtered_sessions, filtered_sales, _ = load_filtered_view()
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)
    st.header("Alert Center")
    threshold = float(st.slider(
        "Temperature threshold (°C)", min_value=30.0, max_value=60.0,
        value=float(TEMP_LIMIT_C), step=0.5, format="%.1f", key=TEMP_KEY,
        help=(
            "High-risk sessions are hotter than this value and overstay "
            f"by more than {OVERSTAY_LIMIT_MINS} minutes."
        ),
    ))
    render_alert_banners(filtered_sessions, threshold)
    st.subheader("High-risk sessions")
    risky = alert_mask(filtered_sessions, threshold).fillna(False)
    table = filtered_sessions.loc[risky].sort_values("Battery_Temp_C", ascending=False)
    st.dataframe(table, width="stretch", hide_index=True)
    st.subheader("Battery temperature vs overstay")
    try:
        draw_temperature_scatter(filtered_sessions, threshold)
    except Exception as exc:
        st.error(f"Could not draw the alert scatter: {exc}")


def page_data_viewer():
    """Show the filtered charging and sales tables with CSV downloads."""
    sessions, sales, filtered_sessions, filtered_sales, _ = load_filtered_view()
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)
    st.header("Data Viewer")
    charging_tab, sales_tab = st.tabs(["Charging sessions", "Sales transactions"])
    views = (
        (charging_tab, filtered_sessions, len(sessions), "charging sessions",
         "filtered_charging_sessions.csv", "download_charging_sessions"),
        (sales_tab, filtered_sales, len(sales), "sales transactions",
         "filtered_sales_transactions.csv", "download_sales_transactions"),
    )
    for tab, filtered, total, noun, file_name, key in views:
        with tab:
            st.caption(f"{len(filtered):,} of {total:,} {noun}")
            st.dataframe(filtered, width="stretch", hide_index=True)
            st.download_button(
                f"Download filtered {noun}", data=filtered.to_csv(index=False).encode("utf-8"),
                file_name=file_name, mime="text/csv", key=key,
            )


def main():
    """Configure the app and run the selected page."""
    st.set_page_config(page_title="Smart Campus Energy & EV Dashboard", layout="wide")
    selected = st.navigation([
        st.Page(
            page_executive_overview, title="Executive Overview",
            url_path="overview", default=True,
        ),
        st.Page(page_alert_center, title="Alert Center", url_path="alerts"),
        st.Page(page_data_viewer, title="Data Viewer", url_path="data"),
    ])
    selected.run()


if __name__ == "__main__":
    main()
