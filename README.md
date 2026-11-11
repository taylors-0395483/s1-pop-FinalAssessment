# Smart Campus Energy & EV Fleet Platform

ITS72604 Principles of Programming, Final Examination (Sept 2026).

## Files

| File | Purpose |
| --- | --- |
| `data/raw/` | Provided datasets: `ev_campus_dataset.csv` (charging sessions) and `sales_data.csv` (sales transactions). |
| `data/cleaned/` | Pipeline outputs: `cleaned_energy_sales_data.csv` and `cleaned_sales_data.csv`. |
| `data_pipeline.py` | Section B1: cleans the raw files, adds power, carbon and billing columns, saves the cleaned files, prints summary statistics. |
| `visualizations.py` | Section B2: three Matplotlib/Seaborn panels saved in `figures/`. |
| `app.py` | Section B3: multi-page Streamlit dashboard (overview, alert center, data viewer). |
| `section_a/` | Example scripts for Section A questions 1 to 3 and their outputs. |
| `outputs/` | Console output captured from each script. |
| `requirements.txt` | Python libraries needed to run everything. |

## Billing rules used

- Carbon savings = energy (kWh) x 0.475 kg CO2.
- Base fee = hourly rate x charging hours (AC 6 MYR/h, DC 15 MYR/h).
- Sessions overlapping 12:00-17:00 pay a 25% peak surcharge on the base fee.
- Overstay after a 15-minute grace period costs 0.5 MYR per minute.
- High-risk session = overstay above 20 minutes and battery temperature above 40 °C.

## How to run

```bash
pip install -r requirements.txt
python data_pipeline.py
python visualizations.py
streamlit run app.py
```
