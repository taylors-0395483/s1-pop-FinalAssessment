"""Multipage dashboard for campus charging sessions and retail energy sales."""

from __future__ import annotations

import base64
import math
import os
import sys
from datetime import date, datetime
from html import escape
from pathlib import Path

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

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "taylors-logo.png"
CAMPUS_PATH = ASSETS_DIR / "taylors-campus.webp"

TEMP_KEY = "alert_temp_threshold"
EV_COLUMNS = (
    "Session_ID", "Charger_Type", "User_Type", "Start_Hour", "Energy_kWh",
    "Overstay_Mins", "Battery_Temp_C", "Carbon_Savings_kgCO2", "Net_Billable_MYR",
)
SALES_COLUMNS = ("Transaction_Date", "Total_Revenue_MYR")

COLOR_PRIMARY_BLUE = "#155EEF"
COLOR_AXIS_DOMAIN = "#c9d6e8"
COLOR_AXIS_GRID = "#e5edf7"
COLOR_MUTED_TEXT = "#687b96"
COLOR_CHART_BG = "#ffffff"

USER_TYPE_COLORS = {
    "domain": ["Student", "Staff", "Visitor"],
    "range": ["#155EEF", "#eb6834", "#1baf7a"],
}

STATUS_COLORS = {
    "domain": ["High risk", "High temp (no overstay)", "Normal"],
    "range": ["#d03b3b", "#fab219", "#c3c2b7"],
}

CHINESE = {
    # Brand and navigation
    "Smart Campus Energy Console": "智慧校园能源控制台",
    "Taylor's Lakeside Campus": "泰莱大学湖畔校区",
    "Executive Overview": "管理概览",
    "Alert Center": "告警中心",
    "Data Viewer": "数据查看",
    "English": "English",
    "中文": "中文",

    # Campus banner descriptions
    "Track campus energy demand, EV charging metrics, and retail sales revenue.":
        "监控校园能源需求、充电指标及零售销售收入。",
    "Identify over-temperature battery events and overstay risk sessions.":
        "监测电池过温事件与超时停放高风险会话。",
    "Inspect and export raw charging sessions and retail transactions.":
        "查看并导出充电会话与零售交易明细数据。",
    "{count:,} of {total:,} sessions · Sales {start} → {end} · Chargers: {chargers}":
        "{total:,} 条记录中已筛选 {count:,} 条 · 销售日期 {start} → {end} · 充电桩类型: {chargers}",
    "None": "无",
    "AC": "AC（交流）",
    "DC": "DC（直流）",

    # Sidebar controls
    "Filters": "筛选条件",
    "{count} active": "{count} 项已启用",
    "SALES": "销售",
    "CHARGING SESSIONS": "充电会话",
    "Reset filters": "重置筛选",
    "Sales filters": "销售筛选",
    "Date range": "日期范围",
    "Sales data has no valid transaction dates.": "销售数据中无有效的交易日期。",
    "Charging session filters": "充电会话筛选",
    "Minimum energy (kWh)": "最低充电量（kWh）",
    "Charger type": "充电桩类型",

    # KPI strip
    "Total energy (kWh)": "总用电量 (kWh)",
    "Delta: average energy per filtered session vs all sessions.":
        "Delta：筛选会话的平均充电量与全部会话对比。",
    "Carbon offset (kg CO₂)": "碳减排量 (kg CO₂)",
    "Delta: average carbon saving per filtered session vs all sessions.":
        "Delta：筛选会话的平均碳减排量与全部会话对比。",
    "Total revenue (MYR)": "总收入 (MYR)",
    "Filtered sales revenue plus filtered charging Net Billable. Delta: average sales "
    "transaction value vs all transactions.":
        "筛选销售收入加上筛选充电净应收账单。Delta：平均销售交易额与全部交易对比。",
    "Active alerts": "活跃告警",
    "Sessions above the alert temperature with more than 20 minutes overstay. Delta: "
    "change versus the default 40 °C threshold.":
        "温度超过告警阈值且超时停放超过 20 分钟的会话。Delta：较默认 40 °C 阈值的变化量。",

    # Executive Overview charts
    "Energy demand by start hour": "按会话开始小时统计能源需求",
    "Total energy delivered across campus by session start hour.":
        "按会话开始小时统计全校总充电交付量。",
    "No charging sessions match the current filters.": "当前筛选条件下无匹配的充电会话。",
    "Start hour": "开始小时",
    "Energy delivered (kWh)": "交付电量 (kWh)",
    "Average net bill by charger and user type": "按充电桩与用户类型统计平均净应收金额",
    "Mean billable revenue comparing AC and DC chargers across user groups.":
        "比较交流与直流充电桩在各用户群体中的平均应收收入。",
    "Average net billable (MYR)": "平均净应收金额 (MYR)",
    "Average net bill (MYR)": "平均净应收金额 (MYR)",
    "User type": "用户类型",
    "Student": "学生",
    "Staff": "教职工",
    "Visitor": "访客",
    "Daily sales revenue": "每日销售收入",
    "Aggregated daily revenue from campus retail and energy services.":
        "校园零售与能源服务的每日汇总收入。",
    "No sales transactions match the current filters.": "当前筛选条件下无匹配的销售交易。",
    "Transaction date": "交易日期",
    "Revenue (MYR)": "收入 (MYR)",

    # Alert Center
    "Temperature threshold (°C)": "温度阈值 (°C)",
    "High-risk sessions are hotter than this value and overstay by more than 20 minutes.":
        "高风险会话电池温度高于此值且超时停放超过 20 分钟。",
    "**High-risk sessions: {count}**\n\n{pct}% of filtered sessions · battery above "
    "{threshold} °C and overstay above 20 min":
        "**高风险会话：{count} 条**\n\n占筛选会话 {pct}% · 电池温度高于 "
        "{threshold} °C 且超时停放超过 20 分钟",
    "**High temperature, no overstay: {count}**\n\n{pct}% of filtered sessions · "
    "battery above {threshold} °C, overstay 20 min or less":
        "**高温未超时：{count} 条**\n\n占筛选会话 {pct}% · 电池温度高于 "
        "{threshold} °C，超时停放不超过 20 分钟",
    "{count:,} high-risk sessions ({share:.1f}% of filtered sessions). Battery temperature "
    "is above {threshold:.1f}°C and overstay is above {limit} minutes.":
        "{count:,} 条高风险充电会话（占筛选会话的 {share:.1f}%）。电池温度高于 "
        "{threshold:.1f}°C 且超时停放超过 {limit} 分钟。",
    "{count:,} sessions are above {threshold:.1f}°C without an overstay above "
    "{limit} minutes ({share:.1f}% of filtered sessions).":
        "{count:,} 条充电会话电池温度高于 {threshold:.1f}°C 且超时未超过 "
        "{limit} 分钟（占筛选会话的 {share:.1f}%）。",
    "No filtered sessions are above the {threshold:.1f}°C temperature threshold.":
        "当前筛选的充电会话中无电池温度高于 {threshold:.1f}°C 的记录。",
    "High-risk sessions": "高风险会话",
    "Sessions above the threshold with an overstay above 20 minutes, hottest first.":
        "高于阈值且超时停放超过 20 分钟的会话，按温度由高到低排序。",
    "Battery temperature vs overstay": "电池温度与超时停放关系",
    "Scatter plot of battery temperature versus overstay minutes with safety thresholds.":
        "电池温度与超时停放分钟数的散点图及安全阈值线。",
    "No valid temperature and overstay data to display.": "无有效的温度与超时数据可供展示。",
    "Battery temperature (°C)": "电池温度 (°C)",
    "Battery temp (°C)": "电池温度 (°C)",
    "Battery Temp (°C)": "电池温度 (°C)",
    "Overstay (minutes)": "超时停放（分钟）",
    "Overstay (mins)": "超时停放（分钟）",
    "Status": "状态",
    "High risk": "高风险",
    "High temp (no overstay)": "高温未超时",
    "Normal": "正常",
    "Session ID": "会话编号",

    # Column configs
    "Session ID": "会话编号",
    "Charger Type": "充电桩类型",
    "User Type": "用户类型",
    "Start Hour": "开始小时",
    "Duration (mins)": "充电时长（分钟）",
    "Duration (hours)": "充电时长（小时）",
    "Energy (kWh)": "充电量 (kWh)",
    "Overstay (mins)": "超时停放（分钟）",
    "Battery Temp (°C)": "电池温度 (°C)",
    "Grid Status": "电网状态",
    "Power Rate (kW)": "功率 (kW)",
    "Carbon Savings (kg CO₂)": "碳减排量 (kg CO₂)",
    "Peak Hour": "高峰时段",
    "Base Rate (MYR)": "基础费率 (MYR)",
    "Base Fee (MYR)": "基础费用 (MYR)",
    "Peak Surge (MYR)": "高峰附加费 (MYR)",
    "Overstay Fee (MYR)": "超时附加费 (MYR)",
    "Net Billable (MYR)": "净应收金额 (MYR)",
    "High Risk": "高风险",

    "Transaction ID": "交易编号",
    "Hub Location": "网点位置",
    "Category": "商品类别",
    "Units Sold": "售出数量",
    "Unit Price (MYR)": "单价 (MYR)",
    "Total Revenue (MYR)": "总收入 (MYR)",
    "Customer Type": "客户类型",
    "Transaction Date": "交易日期",
    "Payment Method": "支付方式",

    # Data Viewer tabs and downloads
    "Charging sessions": "充电会话",
    "Sales transactions": "销售交易",
    "{filtered:,} of {total:,} charging sessions":
        "{total:,} 条充电会话中已筛选 {filtered:,} 条",
    "{filtered:,} of {total:,} sales transactions":
        "{total:,} 条销售交易中已筛选 {filtered:,} 条",
    "Download filtered charging sessions": "下载筛选后的充电会话 CSV",
    "Download filtered sales transactions": "下载筛选后的销售交易 CSV",

    # Error handling messages
    "Could not load the dashboard data: {exc}": "无法加载控制台数据：{exc}",
    "Could not draw hourly energy demand: {exc}": "无法绘制小时能源需求图：{exc}",
    "Could not draw average net billable: {exc}": "无法绘制平均净应收金额图：{exc}",
    "Could not draw sales revenue: {exc}": "无法绘制销售收入图：{exc}",
    "Could not draw the alert scatter: {exc}": "无法绘制告警散点图：{exc}",
}


def t(english: str, **values) -> str:
    """Translate presentation text without altering data category codes."""
    text = (
        CHINESE.get(english, english)
        if st.session_state.get("language") == "中文"
        else english
    )
    return text.format(**values) if values else text


def image_uri(name: str) -> str:
    """Return a base64 data URI for an asset image."""
    image = ASSETS_DIR / name
    if not image.is_file():
        return ""
    mime = "image/png" if image.suffix == ".png" else "image/webp"
    return f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}"


def apply_custom_style():
    """Apply Taylor's campus formal theme styling across all components."""
    st.markdown("""<style>
    .stApp { background:#eef5ff; color:#18253c; }
    [data-testid="stMainBlockContainer"] { max-width:1480px; padding:2rem 2.5rem 3rem; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] {
        background-color: #eef5ff !important;
        border-right: none !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #eef5ff !important;
    }
    .st-key-top_navigation {
        background:white;
        padding:12px 18px;
        border-radius:12px;
        margin-bottom:12px;
        border:1px solid #e1e9f5;
        overflow:hidden;
    }
    .brand { display:flex; align-items:center; gap:10px; min-height:42px; margin:0; }
    .brand img {
        width:42px;
        height:42px;
        max-height:42px;
        object-fit:contain;
        flex-shrink:0;
        display:block;
        border-radius:4px;
    }
    .brand strong {
        display:block;
        font-size:18px;
        letter-spacing:-.6px;
        color:#122344;
        line-height:1.2;
    }
    .brand small { font-size:12px; color:#74829a; }
    .st-key-top_navigation [data-testid="stColumn"]:nth-child(2) [data-testid="stColumn"] {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
    }
    .st-key-top_navigation [data-testid="stColumn"]:nth-child(2) [data-testid="stVerticalBlock"] {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        gap:0 !important;
        width:100% !important;
    }
    .st-key-top_navigation [data-testid="stColumn"]:nth-child(2)
        [data-testid="stElementContainer"],
    .st-key-top_navigation [data-testid="stColumn"]:nth-child(2)
        [data-testid="stMarkdownContainer"],
    .st-key-top_navigation [data-testid="stColumn"]:nth-child(2)
        [data-testid="stPageLink"],
    .st-key-top_navigation [data-testid="stColumn"]:nth-child(2)
        [data-testid="stPageLink"] > div {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        margin:0 !important;
        padding:0 !important;
        height:36px !important;
        min-height:36px !important;
        max-height:36px !important;
        width:100% !important;
    }
    .st-key-top_navigation [data-testid="stPageLink-NavLink"],
    .nav-active {
        box-sizing:border-box !important;
        display:inline-flex !important;
        align-items:center !important;
        justify-content:center !important;
        height:36px !important;
        min-height:36px !important;
        max-height:36px !important;
        padding:6px 12px !important;
        font-size:14px !important;
        font-weight:500 !important;
        line-height:20px !important;
        text-decoration:none !important;
        border:none !important;
        border-bottom:2px solid transparent !important;
        border-radius:0 !important;
        margin:0 !important;
        vertical-align:middle !important;
    }
    .st-key-top_navigation [data-testid="stPageLink-NavLink"] {
        background:transparent !important;
        color:#52637a !important;
    }
    .st-key-top_navigation [data-testid="stPageLink-NavLink"]:hover {
        color:#155eef !important;
        background:transparent !important;
    }
    .st-key-top_navigation [data-testid="stPageLink-NavLink"] p,
    .st-key-top_navigation [data-testid="stPageLink-NavLink"] span {
        font-size:14px !important;
        font-weight:500 !important;
        line-height:20px !important;
        color:inherit !important;
        margin:0 !important;
        padding:0 !important;
    }
    .nav-active-wrap {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        width:100% !important;
        height:36px !important;
        margin:0 !important;
        padding:0 !important;
    }
    .nav-active {
        color:#155eef !important;
        border-bottom:2px solid #155eef !important;
    }
    .campus-banner {
        display:flex;
        align-items:center;
        justify-content:space-between;
        height:150px;
        overflow:hidden;
        margin:4px 0 14px;
    }
    .campus-banner .banner-copy { width:49%; padding:8px 4px; }
    .campus-banner h1 {
        font-size:30px;
        line-height:1.15;
        margin:0 0 8px;
        padding:0;
        letter-spacing:-0.8px;
        color:#122344;
    }
    .campus-banner p { color:#687b96; margin:0 0 6px; font-size:13.5px; }
    .campus-banner img {
        width:49%;
        height:150px;
        object-fit:cover;
        object-position:center 65%;
        border-radius:12px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color:#e1e9f5;
        border-radius:12px;
    }
    .st-key-kpi_strip, .st-key-records, div[class*="st-key-records_"],
    div[class*="st-key-chart_"] {
        background:white;
        border-radius:12px;
    }
    .st-key-records, div[class*="st-key-records_"] { overflow:hidden; }
    .st-key-kpi_strip [data-testid="stMetric"] { padding:3px 8px; }
    .st-key-kpi_strip [data-testid="stMetricLabel"] p { color:#687b96; font-size:13px; }
    .st-key-kpi_strip [data-testid="stMetricValue"] {
        font-size:clamp(23px,2.3vw,34px) !important;
        font-weight:650 !important;
    }
    .st-key-kpi_strip [data-testid="stColumn"] + [data-testid="stColumn"] {
        border-left:1px solid #e5edf7;
    }
    [data-testid="stMetricDelta"] { font-size:12px; }
    h3 { font-size:17px !important; letter-spacing:-.25px; }
    [data-testid="stCaptionContainer"] { color:#74829a; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-filter_panel) > div,
    .st-key-filter_panel {
        background: #ffffff !important;
        border: 1px solid #e1e9f5 !important;
        border-radius: 12px !important;
    }
    .filter-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .filter-header .filter-title {
        font-size: 15px;
        font-weight: 600;
        color: #18253c;
    }
    .filter-header .filter-active {
        font-size: 12px;
        color: #687b96;
    }
    .filter-group-label {
        font-size: 11px;
        letter-spacing: .06em;
        color: #687b96;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    .st-key-filter_panel [data-testid="stWidgetLabel"] p,
    .st-key-filter_panel label p {
        font-size: 13px !important;
    }
    .st-key-alert_cards {
        margin-top: 8px !important;
        margin-bottom: 16px !important;
    }
    .st-key-alert_cards [data-testid="stAlert"] [data-testid="stAlertContainer"] {
        background: #ffffff !important;
        border: 1px solid #e1e9f5 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        color: #18253c !important;
    }
    .st-key-alert_cards [data-testid="stAlert"]:has([data-testid="stAlertContentError"])
        [data-testid="stAlertContainer"] {
        border-left: 4px solid #d03b3b !important;
    }
    .st-key-alert_cards [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"])
        [data-testid="stAlertContainer"] {
        border-left: 4px solid #f79009 !important;
    }
    .st-key-alert_cards [data-testid="stAlert"] [data-testid="stAlertContainer"] p,
    .st-key-alert_cards [data-testid="stAlert"] [data-testid="stAlertContainer"] span,
    .st-key-alert_cards [data-testid="stAlert"] [data-testid="stAlertContainer"] div {
        color: #18253c !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    .st-key-alert_cards [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {
        margin: 0 0 2px !important;
    }
    .st-key-alert_cards [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"]
        p:last-child {
        color: #687b96 !important;
        font-size: 13px !important;
        margin: 0 !important;
    }
    .st-key-alert_cards [data-testid="stAlert"] [data-testid="stAlertContainer"] strong {
        font-size: 14px !important;
        color: #18253c !important;
    }
    .st-key-alert_cards [data-testid="stAlert"]:has([data-testid="stAlertContentError"])
        [data-testid="stAlertDynamicIcon"],
    .st-key-alert_cards [data-testid="stAlert"]:has([data-testid="stAlertContentError"])
        [data-testid="stAlertDynamicIcon"] * {
        color: #d03b3b !important;
    }
    .st-key-alert_cards [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"])
        [data-testid="stAlertDynamicIcon"],
    .st-key-alert_cards [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"])
        [data-testid="stAlertDynamicIcon"] * {
        color: #f79009 !important;
    }
    @media(max-width:760px) {
        [data-testid="stMainBlockContainer"] { padding:2rem 1rem; }
        .campus-banner { height:auto; gap:12px; align-items:stretch; }
        .campus-banner .banner-copy { width:57%; }
        .campus-banner h1 { font-size:24px; }
        .campus-banner img { width:40%; min-height:150px; }
        .brand strong { font-size:16px; }
        .st-key-kpi_strip [data-testid="stColumn"] + [data-testid="stColumn"] {
            border-left:0;
        }
    }
    </style>""", unsafe_allow_html=True)


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
        st.error(t("Could not load the dashboard data: {exc}", exc=exc))
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


def alert_mask(sessions, threshold):
    """Mark sessions above the temperature threshold that also overstay."""
    temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
    overstay = pd.to_numeric(sessions["Overstay_Mins"], errors="coerce")
    return temperature.gt(threshold) & overstay.gt(OVERSTAY_LIMIT_MINS)


def render_sidebar(sessions, sales):
    """Draw sidebar filter panel inside a card container, returning current selection."""
    days = transaction_days(sales).dropna()
    if days.empty:
        st.error(t("Sales data has no valid transaction dates."))
        st.stop()
    first_day, last_day = days.min().date(), days.max().date()
    default_range = (first_day, last_day)
    default_chargers = ordered_labels(sessions["Charger_Type"], ("AC", "DC"))

    observed_max = pd.to_numeric(sessions["Energy_kWh"], errors="coerce").max()
    max_energy = (
        1.0 if pd.isna(observed_max) or float(observed_max) <= 0 else float(observed_max)
    )

    def reset_filters():
        st.session_state["sales_date_range"] = default_range
        st.session_state["minimum_energy_kwh"] = 0.0
        st.session_state["charger_types"] = list(default_chargers)

    cur_range = st.session_state.get("sales_date_range", default_range)
    cur_start, cur_end = date_bounds(cur_range, first_day, last_day)
    is_date_active = (cur_start, cur_end) != default_range

    cur_energy = float(st.session_state.get("minimum_energy_kwh", 0.0))
    is_energy_active = cur_energy > 0.0

    cur_chargers = st.session_state.get("charger_types", default_chargers)
    is_chargers_active = set(cur_chargers) != set(default_chargers)

    active_count = int(is_date_active) + int(is_energy_active) + int(is_chargers_active)
    active_text = t("{count} active", count=active_count)

    # Seed defaults once; passing value= as well would trigger a Session State warning.
    st.session_state.setdefault("sales_date_range", default_range)
    st.session_state.setdefault("minimum_energy_kwh", 0.0)
    st.session_state.setdefault("charger_types", list(default_chargers))

    with st.sidebar.container(key="filter_panel", border=True):
        st.markdown(
            f'<div class="filter-header">'
            f'<span class="filter-title">{escape(t("Filters"))}</span>'
            f'<span class="filter-active">{escape(active_text)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="filter-group-label">{escape(t("SALES"))}</div>',
            unsafe_allow_html=True,
        )
        raw_range = st.date_input(
            t("Date range"),
            min_value=first_day,
            max_value=last_day,
            key="sales_date_range",
        )

        st.markdown(
            f'<div class="filter-group-label">{escape(t("CHARGING SESSIONS"))}</div>',
            unsafe_allow_html=True,
        )
        minimum_energy = st.slider(
            t("Minimum energy (kWh)"),
            min_value=0.0,
            max_value=max_energy,
            step=0.1,
            key="minimum_energy_kwh",
            format="%.1f",
        )
        selected = st.multiselect(
            t("Charger type"),
            options=default_chargers,
            format_func=t,
            key="charger_types",
        )

        st.button(
            t("Reset filters"),
            type="secondary",
            width="stretch",
            icon=":material/restart_alt:",
            on_click=reset_filters,
            key="reset_filters_btn",
        )

    start, end = date_bounds(raw_range, first_day, last_day)
    return {
        "start": start,
        "end": end,
        "minimum_energy": float(minimum_energy),
        "charger_types": tuple(selected),
    }


def load_filtered_view():
    """Load both datasets, draw the sidebar, and apply the current filters."""
    sessions, sales = load_dashboard_data()
    selection = render_sidebar(sessions, sales)
    filtered_sessions, filtered_sales = apply_filters(sessions, sales, selection)
    return sessions, sales, filtered_sessions, filtered_sales, selection


def format_filter_summary(sessions, filtered_sessions, selection):
    """Format one-line filter summary for the campus banner."""
    chargers_list = selection["charger_types"]
    if not chargers_list:
        chargers_text = t("None")
    else:
        chargers_text = ", ".join(t(c) for c in chargers_list)
    return t(
        "{count:,} of {total:,} sessions · Sales {start} → {end} · Chargers: {chargers}",
        count=len(filtered_sessions),
        total=len(sessions),
        start=selection["start"],
        end=selection["end"],
        chargers=chargers_text,
    )


def render_campus_banner(title: str, subtitle: str, summary: str):
    """Render top campus banner with page heading, description, filter summary, and photo."""
    banner_html = (
        f'<div class="campus-banner">'
        f'<div class="banner-copy">'
        f'<h1>{escape(t(title))}</h1>'
        f'<p>{escape(t(subtitle))}</p>'
        f'<p>{escape(summary)}</p>'
        f'</div>'
        f'<img src="{image_uri("taylors-campus.webp")}" '
        f'alt="Taylor’s Lakeside Campus buildings and lake">'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


def render_kpi_header(sessions, filtered_sessions, sales, filtered_sales):
    """Show four executive KPI metric cards inside a bordered strip container."""
    energy_sum = numeric_sum(filtered_sessions, "Energy_kWh")
    carbon_sum = numeric_sum(filtered_sessions, "Carbon_Savings_kgCO2")
    revenue_total = numeric_sum(filtered_sales, "Total_Revenue_MYR")
    revenue_total += numeric_sum(filtered_sessions, "Net_Billable_MYR")
    threshold = float(st.session_state.get(TEMP_KEY, TEMP_LIMIT_C))
    alerts = int(alert_mask(filtered_sessions, threshold).fillna(False).sum())
    baseline = int(alert_mask(filtered_sessions, TEMP_LIMIT_C).fillna(False).sum())

    with st.container(key="kpi_strip", border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                t("Total energy (kWh)"),
                f"{energy_sum:,.2f}",
                delta=mean_delta(filtered_sessions["Energy_kWh"], sessions["Energy_kWh"]),
                help=t("Delta: average energy per filtered session vs all sessions."),
            )
        with col2:
            st.metric(
                t("Carbon offset (kg CO₂)"),
                f"{carbon_sum:,.2f}",
                delta=mean_delta(
                    filtered_sessions["Carbon_Savings_kgCO2"],
                    sessions["Carbon_Savings_kgCO2"],
                ),
                help=t("Delta: average carbon saving per filtered session vs all sessions."),
            )
        with col3:
            st.metric(
                t("Total revenue (MYR)"),
                f"{revenue_total:,.2f}",
                delta=mean_delta(
                    filtered_sales["Total_Revenue_MYR"],
                    sales["Total_Revenue_MYR"],
                ),
                help=t(
                    "Filtered sales revenue plus filtered charging Net Billable. "
                    "Delta: average sales transaction value vs all transactions."
                ),
            )
        with col4:
            st.metric(
                t("Active alerts"),
                f"{alerts:,}",
                delta=alerts - baseline,
                delta_color="inverse",
                help=t(
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
    with st.container(key="chart_hourly_demand", border=True):
        st.markdown(f"### {t('Energy demand by start hour')}")
        st.caption(t("Total energy delivered across campus by session start hour."))
        usable = pd.DataFrame({
            "hour": pd.to_numeric(sessions["Start_Hour"], errors="coerce"),
            "energy": pd.to_numeric(sessions["Energy_kWh"], errors="coerce"),
        }).dropna()
        if usable.empty:
            st.info(t("No charging sessions match the current filters."))
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
                title=t("Start hour"),
                scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(tickMinStep=2, format="d", labelAngle=0),
            ),
            y=alt.Y(
                "Energy_kWh:Q",
                title=t("Energy delivered (kWh)"),
                scale=alt.Scale(zero=True),
            ),
            tooltip=[
                alt.Tooltip("Start_Hour:Q", title=t("Start hour"), format="d"),
                alt.Tooltip("Energy_kWh:Q", title=t("Energy delivered (kWh)"), format=",.1f"),
            ],
        )
        line = base.mark_line(color=COLOR_PRIMARY_BLUE, strokeWidth=2)
        points = base.mark_point(color=COLOR_PRIMARY_BLUE, size=40, filled=True)
        chart = base_config(line + points)
        st.altair_chart(chart, width="stretch")


def draw_average_billable(sessions):
    """Draw average net billable amount by charger and user type."""
    with st.container(key="chart_average_billable", border=True):
        st.markdown(f"### {t('Average net bill by charger and user type')}")
        st.caption(
            t("Mean billable revenue comparing AC and DC chargers across user groups.")
        )
        plot_data = sessions.copy()
        plot_data["Net_Billable_MYR"] = pd.to_numeric(
            plot_data["Net_Billable_MYR"], errors="coerce"
        )
        plot_data = plot_data.dropna(
            subset=["Charger_Type", "User_Type", "Net_Billable_MYR"]
        )
        if plot_data.empty:
            st.info(t("No charging sessions match the current filters."))
            return
        grouped = (
            plot_data.groupby(["Charger_Type", "User_Type"], as_index=False)[
                "Net_Billable_MYR"
            ]
            .mean()
        )
        chart = alt.Chart(grouped).mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        ).encode(
            x=alt.X(
                "Charger_Type:N",
                title=t("Charger type"),
                sort=["AC", "DC"],
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("User_Type:N", sort=USER_TYPE_COLORS["domain"]),
            y=alt.Y("Net_Billable_MYR:Q", title=t("Average net billable (MYR)")),
            color=alt.Color(
                "User_Type:N",
                title=t("User type"),
                scale=alt.Scale(
                    domain=USER_TYPE_COLORS["domain"],
                    range=USER_TYPE_COLORS["range"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Charger_Type:N", title=t("Charger type")),
                alt.Tooltip("User_Type:N", title=t("User type")),
                alt.Tooltip(
                    "Net_Billable_MYR:Q",
                    title=t("Average net bill (MYR)"),
                    format=",.2f",
                ),
            ],
        )
        st.altair_chart(base_config(chart), width="stretch")


def draw_sales_revenue(sales):
    """Draw filtered sales revenue totaled by transaction date."""
    with st.container(key="chart_sales_revenue", border=True):
        st.markdown(f"### {t('Daily sales revenue')}")
        st.caption(t("Aggregated daily revenue from campus retail and energy services."))
        usable = pd.DataFrame({
            "day": transaction_days(sales),
            "revenue": pd.to_numeric(sales["Total_Revenue_MYR"], errors="coerce"),
        }).dropna()
        if usable.empty:
            st.info(t("No sales transactions match the current filters."))
            return
        daily = usable.groupby("day", as_index=False)["revenue"].sum().sort_values("day")
        base = alt.Chart(daily).encode(
            x=alt.X(
                "day:T",
                title=t("Transaction date"),
                axis=alt.Axis(format="%Y-%m-%d", labelAngle=0),
            ),
            y=alt.Y("revenue:Q", title=t("Revenue (MYR)"), scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("day:T", title=t("Transaction date"), format="%Y-%m-%d"),
                alt.Tooltip("revenue:Q", title=t("Revenue (MYR)"), format=",.2f"),
            ],
        )
        line = base.mark_line(color=COLOR_PRIMARY_BLUE, strokeWidth=2)
        points = base.mark_point(color=COLOR_PRIMARY_BLUE, size=40, filled=True)
        chart = base_config(line + points)
        st.altair_chart(chart, width="stretch")


def draw_temperature_scatter(sessions, threshold):
    """Plot battery temperature against overstay with safety reference rules."""
    with st.container(key="chart_temperature_scatter", border=True):
        st.markdown(f"### {t('Battery temperature vs overstay')}")
        st.caption(
            t("Scatter plot of battery temperature versus overstay minutes with safety thresholds.")
        )
        temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
        overstay = pd.to_numeric(sessions["Overstay_Mins"], errors="coerce")
        valid = temperature.notna() & overstay.notna()
        plot_data = sessions.loc[valid].copy()
        if plot_data.empty:
            st.info(t("No valid temperature and overstay data to display."))
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
            x=alt.X(
                "Battery_Temp_C:Q",
                title=t("Battery temperature (°C)"),
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("Overstay_Mins:Q", title=t("Overstay (minutes)")),
            color=alt.Color(
                "Status:N",
                title=t("Status"),
                scale=alt.Scale(
                    domain=STATUS_COLORS["domain"],
                    range=STATUS_COLORS["range"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Session_ID:N", title=t("Session ID")),
                alt.Tooltip("Charger_Type:N", title=t("Charger type")),
                alt.Tooltip(
                    "Battery_Temp_C:Q",
                    title=t("Battery temp (°C)"),
                    format=".1f",
                ),
                alt.Tooltip(
                    "Overstay_Mins:Q",
                    title=t("Overstay (mins)"),
                    format=".1f",
                ),
                alt.Tooltip("Status:N", title=t("Status")),
            ],
        )
        rule_x = alt.Chart(pd.DataFrame({"threshold": [float(threshold)]})).mark_rule(
            color=COLOR_MUTED_TEXT, strokeDash=[4, 4], strokeWidth=1.2
        ).encode(x="threshold:Q")
        rule_y = alt.Chart(
            pd.DataFrame({"limit": [float(OVERSTAY_LIMIT_MINS)]})
        ).mark_rule(
            color=COLOR_MUTED_TEXT, strokeDash=[4, 4], strokeWidth=1.2
        ).encode(y="limit:Q")

        chart = base_config(scatter + rule_x + rule_y)
        st.altair_chart(chart, width="stretch")


def render_alert_banners(sessions, threshold):
    """Show paired error and warning status cards for the current threshold."""
    temperature = pd.to_numeric(sessions["Battery_Temp_C"], errors="coerce")
    hot = temperature.gt(threshold).fillna(False)
    risky = alert_mask(sessions, threshold).fillna(False)
    risky_count = int(risky.sum())
    warm_count = int((hot & ~risky).sum())
    total = len(sessions)
    risky_share = 0.0 if total == 0 else risky_count / total * 100
    warm_share = 0.0 if total == 0 else warm_count / total * 100
    thresh_val = f"{threshold:g}"

    with st.container(key="alert_cards"):
        left_col, right_col = st.columns(2)
        with left_col:
            st.error(
                t(
                    "**High-risk sessions: {count}**\n\n{pct}% of filtered sessions · "
                    "battery above {threshold} °C and overstay above 20 min",
                    count=risky_count,
                    pct=f"{risky_share:.1f}",
                    threshold=thresh_val,
                ),
                icon=":material/error:",
            )
        with right_col:
            st.warning(
                t(
                    "**High temperature, no overstay: {count}**\n\n{pct}% of filtered sessions · "
                    "battery above {threshold} °C, overstay 20 min or less",
                    count=warm_count,
                    pct=f"{warm_share:.1f}",
                    threshold=thresh_val,
                ),
                icon=":material/warning:",
            )


def get_session_column_config():
    """Return localized column configuration for the charging sessions table."""
    return {
        "Session_ID": st.column_config.TextColumn(t("Session ID")),
        "Charger_Type": st.column_config.TextColumn(t("Charger Type")),
        "User_Type": st.column_config.TextColumn(t("User Type")),
        "Start_Hour": st.column_config.NumberColumn(
            t("Start Hour"), format="%d"
        ),
        "Duration_Mins": st.column_config.NumberColumn(
            t("Duration (mins)"), format="%.0f"
        ),
        "Energy_kWh": st.column_config.NumberColumn(
            t("Energy (kWh)"), format="%.2f"
        ),
        "Overstay_Mins": st.column_config.NumberColumn(
            t("Overstay (mins)"), format="%.0f"
        ),
        "Battery_Temp_C": st.column_config.NumberColumn(
            t("Battery Temp (°C)"), format="%.1f"
        ),
        "Grid_Status_Label": st.column_config.TextColumn(t("Grid Status")),
        "Duration_Hours": st.column_config.NumberColumn(
            t("Duration (hours)"), format="%.2f"
        ),
        "Power_Rate_kW": st.column_config.NumberColumn(
            t("Power Rate (kW)"), format="%.2f"
        ),
        "Carbon_Savings_kgCO2": st.column_config.NumberColumn(
            t("Carbon Savings (kg CO₂)"), format="%.2f"
        ),
        "Peak_Hour_Flag": st.column_config.CheckboxColumn(t("Peak Hour")),
        "Base_Rate_MYR": st.column_config.NumberColumn(
            t("Base Rate (MYR)"), format="%.2f"
        ),
        "Base_Fee_MYR": st.column_config.NumberColumn(
            t("Base Fee (MYR)"), format="%.2f"
        ),
        "Peak_Surge_MYR": st.column_config.NumberColumn(
            t("Peak Surge (MYR)"), format="%.2f"
        ),
        "Overstay_Fee_MYR": st.column_config.NumberColumn(
            t("Overstay Fee (MYR)"), format="%.2f"
        ),
        "Net_Billable_MYR": st.column_config.NumberColumn(
            t("Net Billable (MYR)"), format="%.2f"
        ),
        "High_Risk_Flag": st.column_config.CheckboxColumn(t("High Risk")),
    }


def get_sales_column_config():
    """Return localized column configuration for the sales transactions table."""
    return {
        "Transaction_ID": st.column_config.TextColumn(t("Transaction ID")),
        "Hub_Location": st.column_config.TextColumn(t("Hub Location")),
        "Category": st.column_config.TextColumn(t("Category")),
        "Units_Sold": st.column_config.NumberColumn(
            t("Units Sold"), format="%.0f"
        ),
        "Unit_Price_MYR": st.column_config.NumberColumn(
            t("Unit Price (MYR)"), format="%.2f"
        ),
        "Total_Revenue_MYR": st.column_config.NumberColumn(
            t("Total Revenue (MYR)"), format="%.2f"
        ),
        "Customer_Type": st.column_config.TextColumn(t("Customer Type")),
        "Transaction_Date": st.column_config.DateColumn(
            t("Transaction Date"), format="YYYY-MM-DD"
        ),
        "Payment_Method": st.column_config.TextColumn(t("Payment Method")),
    }


def page_executive_overview():
    """Executive overview with shared KPIs and three filtered charts."""
    sessions, sales, filtered_sessions, filtered_sales, selection = load_filtered_view()
    summary = format_filter_summary(sessions, filtered_sessions, selection)
    render_campus_banner(
        "Executive Overview",
        "Track campus energy demand, EV charging metrics, and retail sales revenue.",
        summary,
    )
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)

    left, right = st.columns(2)
    with left:
        try:
            draw_hourly_demand(filtered_sessions)
        except Exception as exc:
            st.error(t("Could not draw hourly energy demand: {exc}", exc=exc))
    with right:
        try:
            draw_average_billable(filtered_sessions)
        except Exception as exc:
            st.error(t("Could not draw average net billable: {exc}", exc=exc))

    try:
        draw_sales_revenue(filtered_sales)
    except Exception as exc:
        st.error(t("Could not draw sales revenue: {exc}", exc=exc))


def page_alert_center():
    """Alert center for over-temperature and high-risk overstay sessions."""
    sessions, sales, filtered_sessions, filtered_sales, selection = load_filtered_view()
    summary = format_filter_summary(sessions, filtered_sessions, selection)
    render_campus_banner(
        "Alert Center",
        "Identify over-temperature battery events and overstay risk sessions.",
        summary,
    )
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)

    threshold = float(st.slider(
        t("Temperature threshold (°C)"),
        min_value=30.0,
        max_value=60.0,
        value=float(st.session_state.get(TEMP_KEY, TEMP_LIMIT_C)),
        step=0.5,
        format="%.1f",
        key=TEMP_KEY,
        help=t(
            "High-risk sessions are hotter than this value and overstay by more than 20 minutes."
        ),
    ))
    render_alert_banners(filtered_sessions, threshold)

    with st.container(key="records_high_risk", border=True):
        st.markdown(f"### {t('High-risk sessions')}")
        st.caption(
            t("Sessions above the threshold with an overstay above 20 minutes, hottest first.")
        )
        risky = alert_mask(filtered_sessions, threshold).fillna(False)
        table = filtered_sessions.loc[risky].sort_values(
            "Battery_Temp_C", ascending=False
        )
        st.dataframe(
            table,
            hide_index=True,
            column_config=get_session_column_config(),
        )

    try:
        draw_temperature_scatter(filtered_sessions, threshold)
    except Exception as exc:
        st.error(t("Could not draw the alert scatter: {exc}", exc=exc))


def page_data_viewer():
    """Show the filtered charging and sales tables with CSV downloads."""
    sessions, sales, filtered_sessions, filtered_sales, selection = load_filtered_view()
    summary = format_filter_summary(sessions, filtered_sessions, selection)
    render_campus_banner(
        "Data Viewer",
        "Inspect and export raw charging sessions and retail transactions.",
        summary,
    )
    render_kpi_header(sessions, filtered_sessions, sales, filtered_sales)

    charging_tab, sales_tab = st.tabs([t("Charging sessions"), t("Sales transactions")])

    with charging_tab:
        with st.container(key="records_charging", border=True):
            st.caption(
                t(
                    "{filtered:,} of {total:,} charging sessions",
                    filtered=len(filtered_sessions),
                    total=len(sessions),
                )
            )
            st.dataframe(
                filtered_sessions,
                hide_index=True,
                height=420,
                column_config=get_session_column_config(),
            )
            st.download_button(
                t("Download filtered charging sessions"),
                data=filtered_sessions.to_csv(index=False).encode("utf-8"),
                file_name="filtered_charging_sessions.csv",
                mime="text/csv",
                key="download_charging_sessions",
                type="primary",
                icon=":material/download:",
            )

    with sales_tab:
        with st.container(key="records_sales", border=True):
            st.caption(
                t(
                    "{filtered:,} of {total:,} sales transactions",
                    filtered=len(filtered_sales),
                    total=len(sales),
                )
            )
            st.dataframe(
                filtered_sales,
                hide_index=True,
                height=420,
                column_config=get_sales_column_config(),
            )
            st.download_button(
                t("Download filtered sales transactions"),
                data=filtered_sales.to_csv(index=False).encode("utf-8"),
                file_name="filtered_sales_transactions.csv",
                mime="text/csv",
                key="download_sales_transactions",
                type="primary",
                icon=":material/download:",
            )


def render_top_navigation(selected_page, pages):
    """Render top brand card with logo, navigation links, and language selector."""
    with st.container(key="top_navigation"):
        brand_col, nav_col, lang_col = st.columns(
            [3.4, 4.8, 1.4], vertical_alignment="center"
        )
        with brand_col:
            logo_data = image_uri("taylors-logo.png")
            st.markdown(
                f'<div class="brand">'
                f'<img src="{logo_data}" alt="Taylor’s University logo">'
                f'<div>'
                f'<strong>{escape(t("Smart Campus Energy Console"))}</strong>'
                f'<small>{escape(t("Taylor\'s Lakeside Campus"))}</small>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with nav_col:
            page_items = [
                ("Executive Overview", pages[0]),
                ("Alert Center", pages[1]),
                ("Data Viewer", pages[2]),
            ]
            c1, c2, c3 = st.columns(3, vertical_alignment="center")
            for col, (label_key, page_obj) in zip([c1, c2, c3], page_items):
                with col:
                    is_current = (
                        page_obj == selected_page
                        or page_obj.url_path == getattr(selected_page, "url_path", None)
                    )
                    if is_current:
                        st.markdown(
                            f'<div class="nav-active-wrap">'
                            f'<span class="nav-active">{escape(t(label_key))}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.page_link(page_obj, label=t(label_key))
        with lang_col:
            st.selectbox(
                "Language / 语言",
                ["English", "中文"],
                key="language",
                label_visibility="collapsed",
            )


def main():
    """Configure the app and run the selected page."""
    if "language" not in st.session_state:
        st.session_state["language"] = "English"

    st.set_page_config(
        page_title=t("Smart Campus Energy Console"),
        page_icon=str(LOGO_PATH),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_style()

    pages = [
        st.Page(
            page_executive_overview,
            title=t("Executive Overview"),
            url_path="",
            default=True,
        ),
        st.Page(
            page_alert_center,
            title=t("Alert Center"),
            url_path="alerts",
        ),
        st.Page(
            page_data_viewer,
            title=t("Data Viewer"),
            url_path="data",
        ),
    ]
    selected_page = st.navigation(pages, position="hidden")
    render_top_navigation(selected_page, pages)
    selected_page.run()


if __name__ == "__main__":
    main()
