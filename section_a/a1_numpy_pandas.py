"""Section A1: Demonstration of NumPy and Pandas telemetry operations.

Showcases 2 NumPy array operations (2D slicing, boolean masking),
2 Pandas DataFrame operations (median imputation, groupby aggregation),
and benchmarks vectorization vs pure Python loop performance.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Configure project root to enable data_pipeline imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import EV_FILE, load_csv, prepare_ev_data  # noqa: E402


def run_pandas_operations(raw_path: Path):
    """Execute Pandas missing value imputation and group-wise billing summary."""
    print("=" * 60 + "\nPandas Operation 1: Missing Value Imputation\n" + "=" * 60)
    try:
        raw_df = load_csv(raw_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading raw EV data: {err}", file=sys.stderr)
        raise

    cols = ["Duration_Mins", "Energy_kWh"]
    print("Missing values before imputation:\n" + raw_df[cols].isna().sum().to_string())
    medians = raw_df[cols].median()
    print(f"\nCalculated medians:\n{medians.to_string()}")
    raw_df[cols] = raw_df[cols].fillna(medians)
    print("\nMissing values after median imputation:\n" + raw_df[cols].isna().sum().to_string())

    print("\n" + "=" * 60 + "\nPandas Operation 2: Group-wise Net Billable Summary\n" + "=" * 60)
    ev_df = prepare_ev_data()
    summary = (
        ev_df.groupby(["Charger_Type", "User_Type"])["Net_Billable_MYR"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )
    print(summary)
    return raw_df, ev_df


def run_numpy_operations(df: pd.DataFrame):
    """Execute 2D slicing for peak hours and boolean masking for thermal alerts."""
    print("\n" + "=" * 60 + "\nNumPy Operation 1: 2D Grid Load Slicing\n" + "=" * 60)
    charger_map = {"AC": 0, "DC": 1}
    charger_idx = df["Charger_Type"].map(charger_map).to_numpy(dtype=int)
    hours = df["Start_Hour"].to_numpy(dtype=int)
    energy = df["Energy_kWh"].to_numpy(dtype=float)

    # Accumulate hourly load into 2D grid (2, 24) and slice peak hours (12:00-16:59)
    grid_matrix = np.zeros((2, 24), dtype=float)
    np.add.at(grid_matrix, (charger_idx, hours), energy)
    peak_totals = grid_matrix[:, 12:17].sum(axis=1)
    day_totals = grid_matrix.sum(axis=1)
    peak_pct = (peak_totals / day_totals) * 100.0
    print(
        f"AC Peak Energy (12:00-16:59): {peak_totals[0]:.2f} kWh "
        f"({peak_pct[0]:.2f}% of daily AC)"
    )
    print(
        f"DC Peak Energy (12:00-16:59): {peak_totals[1]:.2f} kWh "
        f"({peak_pct[1]:.2f}% of daily DC)"
    )

    print("\n" + "=" * 60 + "\nNumPy Operation 2: Boolean Masking Alerts\n" + "=" * 60)
    temps = df["Battery_Temp_C"].to_numpy(dtype=float)
    alert_mask = temps > 45.0
    alert_count = int(np.sum(alert_mask))
    alert_pct = (alert_count / len(temps)) * 100.0
    mean_temp = float(np.mean(temps[alert_mask])) if alert_count > 0 else 0.0
    print(f"Sessions exceeding 45°C: {alert_count} / {len(temps)} ({alert_pct:.2f}%)")
    print(f"Mean temperature of alert sessions: {mean_temp:.2f}°C")


def run_benchmark(ev_df: pd.DataFrame):
    """Benchmark vectorized NumPy operations against pure Python loops."""
    print("\n" + "=" * 60)
    print("Performance Benchmark: NumPy Vectorization vs Python Loop")
    print("=" * 60)
    duration = ev_df["Duration_Hours"].to_numpy(dtype=float)
    energy = ev_df["Energy_kWh"].to_numpy(dtype=float)

    # Tile array up to ~1.2 million rows (1,200 * 1,000)
    large_energy = np.tile(energy, 1000)
    large_duration = np.tile(duration, 1000)
    print(f"Scaled dataset size: {len(large_energy):,} records")

    # Benchmark: Pure Python loop vs NumPy vectorized
    e_list, d_list = large_energy.tolist(), large_duration.tolist()
    t0 = time.perf_counter()
    _ = [e / d for e, d in zip(e_list, d_list)]
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = large_energy / large_duration
    t_np = time.perf_counter() - t0

    speedup = t_py / t_np if t_np > 0 else 0.0
    print(f"Pure Python Loop Execution Time: {t_py:.4f} seconds")
    print(f"NumPy Vectorized Execution Time: {t_np:.4f} seconds")
    print(f"Vectorization Speedup Factor   : {speedup:.2f}x faster")


def main():
    """Main execution orchestrating data operations and benchmarks."""
    imputed_raw_df, prepared_df = run_pandas_operations(EV_FILE)
    run_numpy_operations(imputed_raw_df)
    run_benchmark(prepared_df)


if __name__ == "__main__":
    main()
