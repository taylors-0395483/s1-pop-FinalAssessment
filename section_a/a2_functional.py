"""Section A2: Functional programming pipeline for high-risk EV sessions.

Demonstrates lazy data streams via generator, filter, map, and reduce,
contrasting with an equivalent list comprehension and asserting parity.
"""

import csv
import functools
import sys
from pathlib import Path

# Project path resolution (parent directory containing the dataset)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "cleaned" / "cleaned_energy_sales_data.csv"

# Domain constants defining alert thresholds and fee multipliers
OVERSTAY_LIMIT = 20  # Overstay threshold in minutes
TEMP_LIMIT = 40.0  # Battery temperature threshold in degrees Celsius
SURCHARGE_RATE = 0.15  # Surcharge multiplier (15%) for high-risk sessions


def load_records(path: Path):
    """Yield records lazily as dictionaries with numeric strings cast to float.

    Memory Efficiency:
    Yielding records row by row as a generator achieves O(1) space complexity.
    It streams records on demand from disk rather than materializing the full CSV.
    """
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = {}
                for key, val in row.items():
                    try:
                        record[key] = float(val)
                    except ValueError:
                        record[key] = val
                yield record
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading CSV file: {err}", file=sys.stderr)
        raise


def main():
    """Execute functional pipeline and list comprehension comparison."""
    print("=" * 65)
    print("Section A2: Functional Programming Alert & Surcharge Pipeline")
    print("=" * 65)

    # 1. Functional lazy pipeline (filter -> map -> reduce)
    # Memory Efficiency: filter() and map() return lazy iterators in Python 3.
    # No intermediate collections or lists are allocated during streaming.
    # Pure lambda predicate: single responsibility for evaluating risk conditions.
    # Evaluates lazily per record without side effects or mutable state.
    risk_stream = filter(
        lambda r: (
            r["Overstay_Mins"] > OVERSTAY_LIMIT and r["Battery_Temp_C"] > TEMP_LIMIT
        ),
        load_records(DATA_FILE),
    )

    # Pure lambda transformation: copies the record and appends Surcharge_MYR.
    # Immutability principle: creates a new dictionary to avoid mutating input data.
    surcharge_stream = map(
        lambda r: {
            **r,
            "Surcharge_MYR": round(r["Base_Fee_MYR"] * SURCHARGE_RATE, 2),
        },
        risk_stream,
    )

    # functools.reduce folds the stream into a single scalar sum.
    # The initial value 0.0 serves two vital purposes:
    # 1) Prevents TypeError when the stream contains zero items (empty iterator).
    # 2) Enforces the accumulator type as a float from the start.
    total_reduce = round(
        functools.reduce(
            lambda acc, r: acc + r["Surcharge_MYR"],
            surcharge_stream,
            0.0,
        ),
        2,
    )

    # 2. Equivalent list comprehension implementation
    # Materializes matching records eagerly for direct inspection and validation.
    # Uses inline condition and dictionary expression without calling named lambdas.
    list_comp_records = [
        {**r, "Surcharge_MYR": round(r["Base_Fee_MYR"] * SURCHARGE_RATE, 2)}
        for r in load_records(DATA_FILE)
        if r["Overstay_Mins"] > OVERSTAY_LIMIT and r["Battery_Temp_C"] > TEMP_LIMIT
    ]
    total_list_comp = round(
        sum(r["Surcharge_MYR"] for r in list_comp_records),
        2,
    )

    # Verify identical financial outcomes between paradigms
    assert abs(total_reduce - total_list_comp) < 1e-6, (
        f"Validation failed: reduce={total_reduce} vs list_comp={total_list_comp}"
    )

    # Output executive findings
    print(f"Total High-Risk Alert Sessions Detected : {len(list_comp_records)}")
    print(f"Total Surcharge Revenue (functools.reduce): MYR {total_reduce:.2f}")
    print(f"Total Surcharge Revenue (list comp sum)  : MYR {total_list_comp:.2f}")
    print("\nTop 5 High-Risk Flagged Charging Sessions:")
    header = (
        f"{'Session_ID':<12} {'Temp (°C)':<12} {'Overstay (m)':<14} "
        f"{'Base Fee':<12} {'Surcharge':<10}"
    )
    print(header)
    print("-" * len(header))
    for rec in list_comp_records[:5]:
        print(
            f"{rec['Session_ID']:<12} {rec['Battery_Temp_C']:<12.1f} "
            f"{rec['Overstay_Mins']:<14.0f} {rec['Base_Fee_MYR']:<12.2f} "
            f"{rec['Surcharge_MYR']:<10.2f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
