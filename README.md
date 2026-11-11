# Smart Campus Energy & EV Fleet Platform

ITS72604 Principles of Programming, Final Examination (Sept 2026).

## Folders

| Folder | Contents |
| --- | --- |
| `section_a/` | Section A scripts: `a1_numpy_pandas.py`, `a2_functional.py`, `a3_charts.py`, with their `figures/` and console `outputs/`. |
| `section_b/` | Section B code: `data_pipeline.py` (B1), `visualizations.py` (B2), `app.py` (B3 Streamlit dashboard), with `figures/` and console `outputs/`. |
| `data/raw/` | Provided datasets: `ev_campus_dataset.csv` and `sales_data.csv`. |
| `data/cleaned/` | Pipeline outputs: `cleaned_energy_sales_data.csv` and `cleaned_sales_data.csv`. |

## Billing rules used

- Carbon savings = energy (kWh) x 0.475 kg CO2.
- Base fee = hourly rate x charging hours (AC 6 MYR/h, DC 15 MYR/h).
- Sessions overlapping 12:00-17:00 pay a 25% peak surcharge on the base fee.
- Overstay after a 15-minute grace period costs 0.5 MYR per minute.
- High-risk session = overstay above 20 minutes and battery temperature above 40 °C.

## How to run (from the project root)

```bash
pip install -r requirements.txt
python section_b/data_pipeline.py
python section_b/visualizations.py
streamlit run section_b/app.py
```
