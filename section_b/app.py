"""Multipage dashboard for campus charging sessions and retail energy sales."""

from __future__ import annotations

import math
import os
import sys
from datetime import date, datetime

# Make data_pipeline.py (same folder) importable whatever the working directory is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from data_pipeline import (  # noqa: E402
    OVERSTAY_LIMIT_MINS,
    TEMP_LIMIT_C,
    prepare_ev_data,
    prepare_sales_data,
)

TEMP_KEY = "alert_temp_threshold"
EV_COLUMNS = (
    "Session_ID", "Charger_Type", "User_Type", "Start_Hour", "Energy_kWh",
    "Overstay_Mins", "Battery_Temp_C", "Carbon_Savings_kgCO2", "Net_Billable_MYR",
)
SALES_COLUMNS = ("Transaction_Date", "Total_Revenue_MYR")

COLOR_PRIMARY_BLUE = "#2a78d6"
COLOR_AXIS_DOMAIN = "#c3c2b7"
COLOR_AXIS_GRID = "#e1e0d9"
COLOR_MUTED_TEXT = "#52514e"
COLOR_CHART_BG = "#fcfcfb"

USER_TYPE_COLORS = {
    "domain": ["Student", "Staff", "Visitor"],
    "range": ["#2a78d6", "#eb6834", "#1baf7a"],
}

STATUS_COLORS = {
    "domain": ["High risk", "High temp (no overstay)", "Normal"],
    "range": ["#d03b3b", "#fab219", "#c3c2b7"],
}

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


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
    """Draw the brand header, sales and charging filters, and return selection."""
    st.sidebar.markdown("### ⚡ Smart Campus EV")
    st.sidebar.caption("Energy & fleet dashboard")

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


def render_page_header(title: str, sessions, filtered_sessions, selection):
    """Display the top page title and an active filter summary caption."""
    st.title(title)
    chargers_text = (
        ", ".join(selection["charger_types"]) if selection["charger_types"] else "None"
    )
    summary = (
        f"{len(filtered_sessions):,} of {len(sessions):,} sessions · "
        f"Sales {selection['start']} → {selection['end']} · "
        f"Chargers: {chargers_text}"
    )
    st.caption(summary)


def render_kpi_header(sessions, filtered_sessions, sales, filtered_sales):
    """Show four executive KPI metric cards inside bordered column containers."""
    energy_sum = numeric_sum(filtered_sessions, "Energy_kWh")
    carbon_sum = numeric_sum(filtered_sessions, "Carbon_Savings_kgCO2")
    revenue_total = numeric_sum(filtered_sales, "Total_Revenue_MYR")
    revenue_total += numeric_sum(filtered_sessions, "Net_Billable_MYR")
    threshold = float(st.session_state.get(TEMP_KEY, TEMP_LIMIT_C))
    alerts = int(alert_mask(filtered_sessions, threshold).fillna(False).sum())
    baseline = int(alert_mask(filtered_sessions, TEMP_LIMIT_C).fillna(False).sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric(
                "Energy delivered",
                f"{energy_sum:,.0f} kWh",
                delta=mean_delta(filtered_sessions["Energy_kWh"], sessions["Energy_kWh"]),
                help="Delta: average energy per filtered session vs all sessions.",
            )
    with col2:
        with st.container(border=True):
            st.metric(
                "Carbon offset",
                f"{carbon_sum:,.0f} kg CO₂",
                delta=mean_delta(
                    filtered_sessions["Carbon_Savings_kgCO2"],
                    sessions["Carbon_Savings_kgCO2"],
                ),
                help="Delta: average carbon saving per filtered session vs all sessions.",
            )
    with col3:
        with st.container(border=True):
            st.metric(
                "Total revenue",
                f"MYR {revenue_total:,.0f}",
                delta=mean_delta(filtered_sales["Total_Revenue_MYR"], sales["Total_Revenue_MYR"]),
                help=(
                    "Filtered sales revenue plus filtered charging Net Billable. "
                    "Delta: average sales transaction value vs all transactions."
                ),
            )
    with col4:
        with st.container(border=True):
            st.metric(
                "Active alerts",
                f"{alerts:,}",
                delta=alerts - baseline,
                delta_color="inverse",
                help=(
                    "Sessions above the alert temperature with more than 20 minutes overstay. "
                    "Delta: change versus the default 40 °C threshold."
                ),
            )


def base_config(chart: alt.Chart) -> alt.Chart:
    """Apply standard clean styling and typography to an Altair chart."""
    return (
        chart.properties(background=COLOR_CHART_BG)
        .configure_view(strokeWidth=0)
        .configure_axis(
            domainColor=COLOR_AXIS_DOMAIN,
            gridColor=COLOR_AXIS_GRID,
            gridWidth=0.6,
            labelColor=COLOR_MUTED_TEXT,
            titleColor=COLOR_MUTED_TEXT,
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(
            orient="top",
            labelColor=COLOR_MUTED_TEXT,
            titleColor=COLOR_MUTED_TEXT,
        )
    )


def draw_hourly_demand(sessions):
    """Draw total energy delivered by charging start hour inside a bordered container."""
    with st.container(border=True):
        st.markdown("**Energy demand by start hour**")
        st.caption("Total energy delivered across campus by session start hour.")
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
        chart_data = pd.DataFrame({
            "Start_Hour": grouped.index.astype(int),
            "Energy_kWh": grouped.values,
        })
        base = alt.Chart(chart_data).encode(
            x=alt.X(
                "Start_Hour:Q",
                title="Start hour",
                scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(tickMinStep=2, format="d"),
            ),
            y=alt.Y(
                "Energy_kWh:Q",
                title="Energy delivered (kWh)",
                scale=alt.Scale(zero=True),
            ),
            tooltip=[
                alt.Tooltip("Start_Hour:Q", title="Start hour", format="d"),
                alt.Tooltip("Energy_kWh:Q", title="Energy delivered (kWh)", format=",.1f"),
            ],
        )
        line = base.mark_line(color=COLOR_PRIMARY_BLUE, strokeWidth=2)
        points = base.mark_point(color=COLOR_PRIMARY_BLUE, size=40, filled=True)
        chart = base_config(line + points)
        st.altair_chart(chart, width="stretch")


def draw_average_billable(sessions):
    """Draw average net billable amount by charger and user type."""
    with st.container(border=True):
        st.markdown("**Average net bill by charger and user type**")
        st.caption("Mean billable revenue comparing AC and DC chargers across user groups.")
        plot_data = sessions.copy()
        plot_data["Net_Billable_MYR"] = pd.to_numeric(
            plot_data["Net_Billable_MYR"], errors="coerce"
        )
        plot_data = plot_data.dropna(subset=["Charger_Type", "User_Type", "Net_Billable_MYR"])
        if plot_data.empty:
            st.info("No charging sessions match the current filters.")
            return
        grouped = (
            plot_data.groupby(["Charger_Type", "User_Type"], as_index=False)["Net_Billable_MYR"]
            .mean()
        )
        chart = alt.Chart(grouped).mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        ).encode(
            x=alt.X("Charger_Type:N", title="Charger type", sort=["AC", "DC"],
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("User_Type:N", sort=USER_TYPE_COLORS["domain"]),
            y=alt.Y("Net_Billable_MYR:Q", title="Average net billable (MYR)"),
            color=alt.Color(
                "User_Type:N",
                title="User type",
                scale=alt.Scale(
                    domain=USER_TYPE_COLORS["domain"],
                    range=USER_TYPE_COLORS["range"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Charger_Type:N", title="Charger type"),
                alt.Tooltip("User_Type:N", title="User type"),
                alt.Tooltip("Net_Billable_MYR:Q", title="Average net bill (MYR)", format=",.2f"),
            ],
        )
        st.altair_chart(base_config(chart), width="stretch")


def draw_sales_revenue(sales):
    """Draw filtered sales revenue totaled by transaction date."""
    with st.container(border=True):
        st.markdown("**Daily sales revenue**")
        st.caption("Aggregated daily revenue from campus retail and energy services.")
        usable = pd.DataFrame({
            "day": transaction_days(sales),
            "revenue": pd.to_numeric(sales["Total_Revenue_MYR"], errors="coerce"),
        }).dropna()
        if usable.empty:
            st.info("No sales transactions match the current filters.")
            return
        daily = usable.groupby("day", as_index=False)["revenue"].sum().sort_values("day")
        base = alt.Chart(daily).encode(
            x=alt.X("day:T", title="Transaction date", axis=alt.Axis(format="%Y-%m-%d")),
            y=alt.Y("revenue:Q", title="Revenue (MYR)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("day:T", title="Transaction date", format="%Y-%m-%d"),
                alt.Tooltip("revenue:Q", title="Revenue (MYR)", format=",.2f"),
            ],
        )
        line = base.mark_line(color=COLOR_PRIMARY_BLUE, strokeWidth=2)
        points = base.mark_point(color=COLOR_PRIMARY_BLUE, size=40, filled=True)
        chart = base_config(line + points)
        st.altair_chart(chart, width="stretch")


def draw_temperature_scatter(sessions, threshold):
    """Plot battery temperature against overstay with safety reference rules."""
    with st.container(border=True):
        st.markdown("**Battery temperature vs overstay**")
        st.caption(
            "Scatter plot of battery temperature versus overstay minutes with safety thresholds."
        )
        temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
        overstay = pd.to_numeric(sessions["Overstay_Mins"], errors="coerce")
        valid = temperature.notna() & overstay.notna()
        plot_data = sessions.loc[valid].copy()
        if plot_data.empty:
            st.info("No valid temperature and overstay data to display.")
            return

        plot_data["Battery_Temp_C"] = temperature[valid]
        plot_data["Overstay_Mins"] = overstay[valid]
        is_hot = plot_data["Battery_Temp_C"] > threshold
        is_overstay = plot_data["Overstay_Mins"] > OVERSTAY_LIMIT_MINS
        conditions = [
            is_hot & is_overstay,
            is_hot & ~is_overstay,
        ]
        choices = ["High risk", "High temp (no overstay)"]
        plot_data["Status"] = np.select(conditions, choices, default="Normal")

        scatter = alt.Chart(plot_data).mark_point(
            size=40, opacity=0.75, filled=True
        ).encode(
            x=alt.X("Battery_Temp_C:Q", title="Battery temperature (°C)",
                    scale=alt.Scale(zero=False)),
            y=alt.Y("Overstay_Mins:Q", title="Overstay (minutes)"),
            color=alt.Color(
                "Status:N",
                title="Status",
                scale=alt.Scale(
                    domain=STATUS_COLORS["domain"],
                    range=STATUS_COLORS["range"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Session_ID:N", title="Session ID"),
                alt.Tooltip("Charger_Type:N", title="Charger type"),
                alt.Tooltip("Battery_Temp_C:Q", title="Battery temp (°C)", format=".1f"),
                alt.Tooltip("Overstay_Mins:Q", title="Overstay (mins)", format=".1f"),
                alt.Tooltip("Status:N", title="Status"),
            ],
        )
        rule_x = alt.Chart(pd.DataFrame({"threshold": [float(threshold)]})).mark_rule(
            color=COLOR_MUTED_TEXT, strokeDash=[4, 4], strokeWidth=1.2
        ).encode(x="threshold:Q")
        rule_y = alt.Chart(pd.DataFrame({"limit": [float(OVERSTAY_LIMIT_MINS)]})).mark_rule(
            color=COLOR_MUTED_TEXT, strokeDash=[4, 4], strokeWidth=1.2
        ).encode(y="limit:Q")

        chart = base_config(scatter + rule_x + rule_y)
        st.altair_chart(chart, width="stretch")


def page_executive_overview():
    """Executive overview with shared KPIs and three filtered charts."""
    sessions, sales, filtered_sessions, filtered_sales, selection = load_filtered_view()
    render_page_header("Executive Overview", sessions, filtered_sessions, selection)
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)

    left, right = st.columns(2)
    with left:
        try:
            draw_hourly_demand(filtered_sessions)
        except Exception as exc:
            st.error(f"Could not draw hourly energy demand: {exc}")
    with right:
        try:
            draw_average_billable(filtered_sessions)
        except Exception as exc:
            st.error(f"Could not draw average net billable: {exc}")

    try:
        draw_sales_revenue(filtered_sales)
    except Exception as exc:
        st.error(f"Could not draw sales revenue: {exc}")


def render_alert_banners(sessions, threshold):
    """Show an error, warning, or success banner for the current threshold."""
    temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
    hot = temperature.gt(threshold).fillna(False)
    risky = alert_mask(sessions, threshold).fillna(False)
    risky_count = int(risky.sum())
    warm_count = int((hot & ~risky).sum())
    total = len(sessions)
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


def page_alert_center():
    """Alert center for over-temperature and high-risk overstay sessions."""
    sessions, sales, filtered_sessions, filtered_sales, selection = load_filtered_view()
    render_page_header("Alert Center", sessions, filtered_sessions, selection)
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)

    threshold = float(st.slider(
        "Temperature threshold (°C)", min_value=30.0, max_value=60.0,
        value=float(TEMP_LIMIT_C), step=0.5, format="%.1f", key=TEMP_KEY,
        help=(
            "High-risk sessions are hotter than this value and overstay "
            f"by more than {OVERSTAY_LIMIT_MINS} minutes."
        ),
    ))
    render_alert_banners(filtered_sessions, threshold)

    st.markdown("**High-risk sessions**")
    st.caption("Sessions above the threshold with an overstay above 20 minutes, hottest first.")
    risky = alert_mask(filtered_sessions, threshold).fillna(False)
    table = filtered_sessions.loc[risky].sort_values("Battery_Temp_C", ascending=False)
    session_config = {
        "Battery_Temp_C": st.column_config.NumberColumn("Battery Temp (°C)", format="%.1f"),
        "Overstay_Mins": st.column_config.NumberColumn("Overstay (mins)", format="%.1f"),
        "Energy_kWh": st.column_config.NumberColumn("Energy (kWh)", format="%.2f"),
        "Carbon_Savings_kgCO2": st.column_config.NumberColumn(
            "Carbon Savings (kg CO₂)", format="%.2f"
        ),
        "Net_Billable_MYR": st.column_config.NumberColumn(
            "Net Billable (MYR)", format="MYR %.2f"
        ),
        "Power_Rate_kW": st.column_config.NumberColumn("Power Rate (kW)", format="%.2f"),
        "Duration_Mins": st.column_config.NumberColumn("Duration (mins)", format="%.1f"),
    }
    st.dataframe(table, hide_index=True, column_config=session_config)

    try:
        draw_temperature_scatter(filtered_sessions, threshold)
    except Exception as exc:
        st.error(f"Could not draw the alert scatter: {exc}")


def page_data_viewer():
    """Show the filtered charging and sales tables with CSV downloads."""
    sessions, sales, filtered_sessions, filtered_sales, selection = load_filtered_view()
    render_page_header("Data Viewer", sessions, filtered_sessions, selection)
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)

    charging_tab, sales_tab = st.tabs(["Charging sessions", "Sales transactions"])

    session_config = {
        "Battery_Temp_C": st.column_config.NumberColumn("Battery Temp (°C)", format="%.1f"),
        "Overstay_Mins": st.column_config.NumberColumn("Overstay (mins)", format="%.1f"),
        "Energy_kWh": st.column_config.NumberColumn("Energy (kWh)", format="%.2f"),
        "Carbon_Savings_kgCO2": st.column_config.NumberColumn(
            "Carbon Savings (kg CO₂)", format="%.2f"
        ),
        "Net_Billable_MYR": st.column_config.NumberColumn(
            "Net Billable (MYR)", format="MYR %.2f"
        ),
        "Power_Rate_kW": st.column_config.NumberColumn("Power Rate (kW)", format="%.2f"),
        "Duration_Mins": st.column_config.NumberColumn("Duration (mins)", format="%.1f"),
    }
    sales_config = {
        "Total_Revenue_MYR": st.column_config.NumberColumn(
            "Total Revenue (MYR)", format="MYR %.2f"
        ),
        "Unit_Price_MYR": st.column_config.NumberColumn(
            "Unit Price (MYR)", format="MYR %.2f"
        ),
        "Units_Sold": st.column_config.NumberColumn("Units Sold", format="%.0f"),
        "Transaction_Date": st.column_config.DateColumn(
            "Transaction Date", format="YYYY-MM-DD"
        ),
    }

    with charging_tab:
        st.caption(f"{len(filtered_sessions):,} of {len(sessions):,} charging sessions")
        st.dataframe(
            filtered_sessions,
            hide_index=True,
            height=420,
            column_config=session_config,
        )
        st.download_button(
            "Download filtered charging sessions",
            data=filtered_sessions.to_csv(index=False).encode("utf-8"),
            file_name="filtered_charging_sessions.csv",
            mime="text/csv",
            key="download_charging_sessions",
            type="primary",
            icon=":material/download:",
        )

    with sales_tab:
        st.caption(f"{len(filtered_sales):,} of {len(sales):,} sales transactions")
        st.dataframe(
            filtered_sales,
            hide_index=True,
            height=420,
            column_config=sales_config,
        )
        st.download_button(
            "Download filtered sales transactions",
            data=filtered_sales.to_csv(index=False).encode("utf-8"),
            file_name="filtered_sales_transactions.csv",
            mime="text/csv",
            key="download_sales_transactions",
            type="primary",
            icon=":material/download:",
        )


def main():
    """Configure the app and run the selected page."""
    st.set_page_config(page_title="Smart Campus Energy & EV Dashboard", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    selected = st.navigation([
        st.Page(
            page_executive_overview,
            title="Executive Overview",
            url_path="",
            default=True,
        ),
        st.Page(page_alert_center, title="Alert Center", url_path="alerts"),
        st.Page(page_data_viewer, title="Data Viewer", url_path="data"),
    ])
    selected.run()


if __name__ == "__main__":
    main()
