# Chicago Crime Forecasting & Hotspot Prediction

This project forecasts **daily crime counts for each of Chicago's 22 police districts, 7 days ahead**, and predicts **next week's top-20 crime beats**. Everything runs in one Colab notebook ([`chicago_crime_forecasting.ipynb`](chicago_crime_forecasting.ipynb)) on the free CPU tier, with no API keys. Every number below comes from running that notebook on the **2026-11-04 API snapshot**. The query window is pinned, but Socrata revises past records, so later reruns can drift slightly. All outputs are saved in the notebook and in `outputs/results.json`.

## Data
- **Source:** City of Chicago "Crimes - 2001 to Present" (Socrata `ijzp-q8t2`), 2015-01-01 to 2026-10-12.
- **Aggregated on the server:** SoQL `GROUP BY` queries cover **2,952,223 raw records** but download only **96,718** daily-district rows and **200,629** weekly-beat rows. Cleaning is done in DuckDB.
- **Removed:**
  - **1** record with no district.
  - **2,632** records with non-city district codes (031, 061).
  - **8,210** records from the last 14 days before the API's latest date, which are incomplete because of reporting lag.
- **Kept:** **2,941,380** records, giving 94,682 district-days with no date gaps.
- **Beat table (separate):** 272 beats × 614 weeks; **4,740** records in sparse or invalid beats dropped. Beats were chosen from pre-2023 weeks only, so the backtest period does not influence which beats are included.
- **Check:** weekly totals rebuilt from the beat query match the daily query exactly.

## Method (and why)
- **Horizon:** 7 days, one row per district-day. *Why:* a weekly staffing and deployment cycle needs about a week of lead time.
- **Features:**
  - Lags of 7, 14, 28 and 364 days.
  - 28-day rolling mean and standard deviation, shifted by 7 days.
  - Day of week, month, US-holiday flag, and district as a category.

  *Why lags ≥ 7:* every feature for all 7 target days is already known at the forecast origin, so nothing from the forecast window leaks into training.
- **Models, from simple to complex:**
  - Seasonal naive (same weekday last week).
  - Mean of the last 4 same weekdays.
  - Holt-Winters per district (damped additive trend + weekly seasonality, fit on the last 2 years).
  - One global LightGBM model (Poisson objective, 300 trees, seed 42).

  *Why a global LightGBM:* a single model learns shared holiday and seasonal effects from all 22 districts, where per-district models see only their own data. *Why Poisson:* the targets are non-negative counts. *Why Holt-Winters uses only 2 years:* the smoothing already discounts older data, so a longer history adds fitting time without adding information.
- **Backtest:** walk-forward from 2023 to 2025, with a new origin every 28 days (39 origins). Each origin retrains on all data before it and predicts the next 7 days. *Why:* a random split would leak future information; 28 days gives 39 near-independent test weeks across every season.
- **Metrics:** MAE, RMSE and WAPE. *Why:* MAE is the error in crimes per day, which is what a staffing decision uses; RMSE penalises big misses; WAPE is scale-free across districts and, unlike MAPE, does not blow up on low-count days.
- **Confidence interval:** paired bootstrap over the 39 origin blocks, 1,000 resamples. *Why resample whole origins rather than single rows:* errors within the same week are correlated, so row-level resampling would make the interval too narrow.

## Results: 2023-2025 backtest (6,006 district-day forecasts per model)
| Model | MAE | RMSE | WAPE | Time, laptop CPU | Time, Colab free CPU |
|---|---|---|---|---|---|
| Seasonal naive | 7.314 | 9.448 | 23.2% | 0.013 s | 0.015 s |
| Mean of last 4 weekdays | 5.682 | 7.368 | 18.0% | 0.012 s | 0.011 s |
| Holt-Winters (858 fits) | 5.308 | 6.860 | 16.8% | 211.0 s | 298.1 s |
| **LightGBM (global)** | **5.197** | **6.677** | **16.5%** | 32.7 s | 92.6 s |

- **Overall:** LightGBM cuts MAE by **28.9% versus seasonal naive (95% CI 26.6-31.5%; 2.12 crimes/day, CI 1.90-2.38)**. It is better in **all 22 districts** (22.6-33.8%) and on every day of the week.
- **Cost:** Holt-Winters is nearly as accurate but **3.2× (Colab) to 6.5× (laptop) slower**. Times are train + predict over all 39 origins. The baseline times cover only selecting the precomputed lag columns.
- **Holidays:** on the 198 holiday rows, seasonal naive's MAE is 10.33 against LightGBM's 5.96.

**EDA:**
- The yearly cycle explains 48.8% of the detrended variance (MSTL) and the weekly cycle 8.1%.
- The COVID break cut crime by **32.2%** (21 Mar - 31 May 2020 against the same weeks of 2019).
- Christmas is 31% below neighbouring same weekdays. New Year's Day (+66%) and the 1st of each month (+25%) are inflated, most likely a recording artifact for offences with an unknown date.

## Drift & failure modes
- **Frozen vs retrained:** a LightGBM model frozen on 2016-2019 is 2.9-6.8% worse per year than the walk-forward model in 2020-2025. It over-predicts by **+2.00 / +2.27 crimes per district-day in 2020 / 2021**, against **+0.56 / +0.54** for the retrained model, so retraining every 28 days cuts the bias by 53-76% (2020, 2021, 2025). The gap stays small, probably because the lag features carry the new level.
- **PSI (2016-2019 vs 2020-2025):** the rolling-mean feature shifted moderately (**0.195**); the target and lag-7 are around 0.11.
- **Worst window:** both models fail in the window containing **31 May 2020**, the civil unrest after George Floyd's murder: 1,901 crimes city-wide against a mean of 542/day for the rest of May-June 2020 (window MAE 16.0).
- **3 biggest backtest failures** (MAE per district-week; the causes are likely explanations, not proven):
  1. **District 8, week of 2024-09-09 (MAE 15.2):** a week-long level jump with no holiday (mean 58.7 against a forecast of 46.7). A 7-day-lagged model cannot react inside the week.
  2. **District 19, week of 2024-06-17 (MAE 15.0):** lag-364 probably "remembered" the 2023 spike, but the Pride Parade was a week later in 2024.
  3. **District 19, week of 2023-06-19 (MAE 13.8):** 124 crimes on Sunday 25 June against a forecast of 63, likely the Pride Parade (route through district 19). A one-off event spike.

## Hotspot prediction (beats, weekly)
Each week, the model predicts every beat's count for the next week and takes the top 20. LightGBM uses lags of 1, 2, 4 and 52 weeks plus 4- and 13-week means, retrained every 4 weeks. *Why retrain every 4 weeks rather than weekly:* it gives the same walk-forward guarantee at a quarter of the cost. Over **156 weeks × 272 beats** (2023-2025):
- **Precision@20:** **0.575** for LightGBM against **0.467** for "last week's top-20".
- **Week by week:** LightGBM is better in 131 weeks and tied in 18.

## Limitations
- The target is **recorded** crime, not actual crime.
- Event spikes (parades, unrest) are not in the features.
- Holt-Winters is fit on 2 years, not on all history.
- The hotspot model is retrained every 4 weeks, not weekly.
- **The backtest is somewhat optimistic:** it uses fully revised counts, whereas a live forecast would see the most recent days under-counted.
- Prediction intervals were not built.
- Nothing from the brief was cut.
- Runtime: **566 s (about 9.5 min)** end to end on a laptop CPU (`runtime_s` in `results.json`). On Colab's free CPU a fresh run took **1,183 s (about 20 min)**, of which **560 s** was the city API download, so it exceeds the 10-minute target. Reruns in the same session use the cached CSVs in `data/` and skip the download. Verified on Colab: every accuracy metric matched the laptop run exactly; only the timings differ.

## Ethics note
Crime records measure **where police look and where people report**, not only where crime happens. Sending more patrols to predicted hotspots produces more recorded crime there, which the model then predicts again: a **feedback loop** that can over-police already heavily policed neighbourhoods. Reporting rates also differ by community and crime type. Outputs like these should inform resource planning (for example, staffing or emergency response) together with community input and audits of error by area. They should not be used as a basis for targeting individuals.

## Resume bullets
- Built a 7-day-ahead crime-count forecaster for Chicago's 22 police districts from 2.95M records aggregated on the server with SoQL + DuckDB. In a 39-origin walk-forward backtest (2023-2025), a global LightGBM model cut MAE by 28.9% versus seasonal naive (bootstrap 95% CI 26.6-31.5%), beating it in all 22 districts.
- Benchmarked 4 models by accuracy and cost: LightGBM (MAE 5.20, WAPE 16.5%) beat Holt-Winters (MAE 5.31) while training about 3× faster (93 s vs 298 s on Colab's free CPU). Quantified COVID-era drift with PSI and a frozen-vs-retrained comparison, which showed that retraining cut the frozen 2016-2019 model's COVID-era bias from +2.0 to +0.6 crimes per district-day.
- Predicted weekly top-20 crime hotspots across 272 beats. Precision@20 reached 0.575 against 0.467 for the last-week baseline over 156 backtest weeks, and the write-up documents reporting-bias and feedback-loop risks.
