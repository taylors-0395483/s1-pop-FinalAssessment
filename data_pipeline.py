"""Data pipeline for the Smart Campus Energy & EV Fleet platform (Section B1).

The script reads the raw charging-session and sales files, cleans them,
adds the derived billing and carbon metrics, saves the cleaned files and
prints summary statistics. The other modules import the same functions so
that every part of the submission uses one set of business rules.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
EV_FILE = RAW_DIR / "ev_campus_dataset.csv"
SALES_FILE = RAW_DIR / "sales_data.csv"
CLEANED_EV_FILE = CLEANED_DIR / "cleaned_energy_sales_data.csv"
CLEANED_SALES_FILE = CLEANED_DIR / "cleaned_sales_data.csv"

# Business rules shared by the pipeline, the charts and the web app.
CARBON_FACTOR = 0.475  # kg CO2 saved per kWh delivered
BASE_RATE_MYR_PER_HOUR = {"AC": 6.0, "DC": 15.0}
PEAK_START, PEAK_END = 12, 17  # campus peak window, 12:00-17:00
PEAK_SURCHARGE_RATE = 0.25  # extra 25% of the base fee in the peak window
OVERSTAY_GRACE_MINS = 15
OVERSTAY_RATE_MYR_PER_MIN = 0.5
OVERSTAY_LIMIT_MINS = 20  # high-risk rule: overstay above 20 minutes ...
TEMP_LIMIT_C = 40.0  # ... and battery temperature above 40 °C

EV_CATEGORIES = {
    "Charger_Type": {"AC", "DC"},
    "User_Type": {"Student", "Staff", "Visitor"},
    "Grid_Status_Label": {"Normal", "Grid_Surge", "Overheated"},
}
EV_NUMERIC = ["Start_Hour", "Duration_Mins", "Energy_kWh", "Overstay_Mins", "Battery_Temp_C"]
SALES_CATEGORIES = {
    "Hub_Location": {"Lakeside_Hub", "Subang_Hub", "KL_Campus_Hub", "Taylor_Tower_Hub"},
    "Category": {"Charging_Service", "EV_Accessory", "Energy_Credit", "Fleet_Pass"},
    "Customer_Type": {"Student", "Staff", "Public_Fleet"},
    "Payment_Method": {"TouchNGo", "Credit_Card", "Campus_Wallet"},
}
SALES_NUMERIC = ["Units_Sold", "Unit_Price_MYR", "Total_Revenue_MYR"]


def report(verbose, message):
    """Print a cleaning step only when the caller asks for console output."""
    if verbose:
        print(message)


def load_csv(path):
    """Read a CSV file and stop with a clear message if it is missing or empty."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Data file not found: {path}") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Data file is empty: {path}") from error
    return df


def validate_categories(df, allowed, verbose=True):
    """Trim text columns and replace values outside the allowed set with the column mode."""
    for column, valid_values in allowed.items():
        df[column] = df[column].astype("string").str.strip()
        invalid = ~df[column].isin(valid_values)
        if invalid.any():
            mode_value = df.loc[~invalid, column].mode()[0]
            report(verbose, f"  {column}: {invalid.sum()} invalid value(s) "
                            f"replaced with '{mode_value}'")
            df.loc[invalid, column] = mode_value
        else:
            report(verbose, f"  {column}: all values valid {sorted(valid_values)}")
    return df


def impute_median(df, columns, verbose=True):
    """Fill missing numeric values with the median of each column."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing = int(df[column].isna().sum())
        if missing:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            report(verbose, f"  {column}: {missing} missing value(s) "
                            f"filled with median {median_value:g}")
    return df


def clean_ev_data(df, verbose=False):
    """Clean the charging-session table: duplicates, numeric gaps and categories."""
    df = df.copy()
    duplicates = int(df.duplicated("Session_ID").sum())
    df = df.drop_duplicates("Session_ID")
    report(verbose, f"  Duplicate Session_ID rows removed: {duplicates}")
    df = impute_median(df, EV_NUMERIC, verbose)
    df = validate_categories(df, EV_CATEGORIES, verbose)
    if (df["Duration_Mins"] <= 0).any() or (df["Energy_kWh"] < 0).any():
        raise ValueError("Duration must be positive and energy cannot be negative.")
    df["Start_Hour"] = df["Start_Hour"].astype(int)
    return df


def flag_peak_overlap(start_hour, duration_hours):
    """Return 1 when a session overlaps 12:00-17:00, including sessions past midnight."""
    end_hour = start_hour + duration_hours
    same_day = (start_hour < PEAK_END) & (end_hour > PEAK_START)
    next_day = end_hour > 24 + PEAK_START
    return (same_day | next_day).astype(int)


def engineer_ev_features(df):
    """Add power, carbon, billing and risk columns to the cleaned session table."""
    df = df.copy()
    df["Duration_Hours"] = df["Duration_Mins"] / 60
    df["Power_Rate_kW"] = df["Energy_kWh"] / df["Duration_Hours"]
    df["Carbon_Savings_kgCO2"] = (df["Energy_kWh"] * CARBON_FACTOR).round(2)
    df["Peak_Hour_Flag"] = flag_peak_overlap(df["Start_Hour"], df["Duration_Hours"])
    df["Base_Rate_MYR"] = df["Charger_Type"].map(BASE_RATE_MYR_PER_HOUR)
    df["Base_Fee_MYR"] = (df["Base_Rate_MYR"] * df["Duration_Hours"]).round(2)
    peak_surge = df["Base_Fee_MYR"] * PEAK_SURCHARGE_RATE * df["Peak_Hour_Flag"]
    df["Peak_Surge_MYR"] = peak_surge.round(2)
    chargeable_overstay = np.clip(df["Overstay_Mins"] - OVERSTAY_GRACE_MINS, 0, None)
    df["Overstay_Fee_MYR"] = (chargeable_overstay * OVERSTAY_RATE_MYR_PER_MIN).round(2)
    df["Net_Billable_MYR"] = df[["Base_Fee_MYR", "Peak_Surge_MYR", "Overstay_Fee_MYR"]].sum(axis=1)
    high_risk = (df["Overstay_Mins"] > OVERSTAY_LIMIT_MINS) & (df["Battery_Temp_C"] > TEMP_LIMIT_C)
    df["High_Risk_Flag"] = high_risk.astype(int)
    return df


def clean_sales_data(df, verbose=False):
    """Clean the sales table and rebuild units or revenue from the price relation."""
    df = df.copy()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    if df["Transaction_Date"].isna().any():
        raise ValueError("Some Transaction_Date values could not be parsed.")
    for column in SALES_NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # Units can be recovered exactly from revenue / price; the median is the fallback.
    units_gap = df["Units_Sold"].isna() & df["Total_Revenue_MYR"].notna()
    df.loc[units_gap, "Units_Sold"] = df["Total_Revenue_MYR"] / df["Unit_Price_MYR"]
    report(verbose, f"  Units_Sold: {int(units_gap.sum())} value(s) recovered from revenue / price")
    df = impute_median(df, ["Units_Sold"], verbose)
    revenue_gap = df["Total_Revenue_MYR"].isna()
    df.loc[revenue_gap, "Total_Revenue_MYR"] = df["Units_Sold"] * df["Unit_Price_MYR"]
    report(verbose, f"  Total_Revenue_MYR: {int(revenue_gap.sum())} value(s) "
                    "rebuilt as units x price")
    df = validate_categories(df, SALES_CATEGORIES, verbose)
    return df


def prepare_ev_data(path=EV_FILE, verbose=False):
    """Load, clean and enrich the charging-session data in one call."""
    return engineer_ev_features(clean_ev_data(load_csv(path), verbose))


def prepare_sales_data(path=SALES_FILE, verbose=False):
    """Load and clean the sales data in one call."""
    return clean_sales_data(load_csv(path), verbose)


def print_summary(ev, sales):
    """Print the summary statistics required for the console evidence."""
    pd.set_option("display.width", 120)
    print("\n=== Charging sessions: descriptive statistics ===")
    columns = ["Duration_Mins", "Energy_kWh", "Battery_Temp_C", "Power_Rate_kW", "Net_Billable_MYR"]
    print(ev[columns].describe().round(2).to_string())
    print("\n=== Totals by charger type ===")
    totals = ev.groupby("Charger_Type").agg(
        Sessions=("Session_ID", "count"),
        Energy_kWh=("Energy_kWh", "sum"),
        Carbon_kgCO2=("Carbon_Savings_kgCO2", "sum"),
        Net_Billable_MYR=("Net_Billable_MYR", "sum"),
    )
    print(totals.round(2).to_string())
    print(f"\nPeak-window sessions: {int(ev['Peak_Hour_Flag'].sum())}"
          f" | High-risk sessions: {int(ev['High_Risk_Flag'].sum())}")
    print("\n=== Sales revenue by hub (MYR) ===")
    print(sales.groupby("Hub_Location")["Total_Revenue_MYR"].sum().round(2).to_string())
    print(f"Missing values left -> sessions: {int(ev.isna().sum().sum())},"
          f" sales: {int(sales.isna().sum().sum())}")


def main():
    """Run the full pipeline and save both cleaned files."""
    try:
        print("Cleaning charging sessions:")
        ev = prepare_ev_data(verbose=True)
        print("Cleaning sales transactions:")
        sales = prepare_sales_data(verbose=True)
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        ev.to_csv(CLEANED_EV_FILE, index=False)
        sales.to_csv(CLEANED_SALES_FILE, index=False, date_format="%Y-%m-%d")
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Pipeline stopped: {error}")
        raise SystemExit(1) from error
    print(f"\nSaved {len(ev)} sessions to {CLEANED_EV_FILE.name}")
    print(f"Saved {len(sales)} transactions to {CLEANED_SALES_FILE.name}")
    print_summary(ev, sales)


if __name__ == "__main__":
    main()
