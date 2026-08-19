# Forecasting Bitcoin's Hourly Realized Variance

Data, code and model outputs for the master's thesis *Forecasting Bitcoin's
Hourly Realized Variance: A Comparative Evaluation of Heterogeneous
Autoregressive, GARCH, and Long Short-Term Memory Models Across Market Regimes
(2016–2025)*.

Yılmaz Can Ekmekçi · Faculty of Economic Sciences, University of Warsaw ·
Data Science and Business Analytics · supervised by dr Paweł Sakowski and
dr Jakub Michańków (Quantitative Finance Research Group, Department of
Quantitative Finance and Machine Learning).

---

## What the thesis does

Hourly realized variance of bitcoin is constructed from five-minute Bitstamp
data covering December 2016 to December 2025 — a window containing the
LUNA/Terra collapse, the FTX bankruptcy, the spot-ETF approval and the 2024
halving. Four models are estimated and compared on a held-out 2024–2025
sample: HAR-Classic, HAR-Extended (with jump and downside-semivariance
components), a multiplicative-component GARCH, and a two-layer LSTM, against
random-walk and historical-mean benchmarks.

Headline results: the parsimonious linear benchmark forecasts significantly
more accurately than the neural network under squared log-error; the jump and
semivariance components reduce rather than improve accuracy; and the ranking
of the models reverses when the loss function changes from squared log-error
to QLIKE.

---

## Repository layout

```
data/
  raw/
    BTCUSD_5min_2015_2025.csv      raw 5-minute OHLCV, collected from Bitstamp
  mc_garch_output/
    mcgarch_forecasts.csv          log_RV_hat and sigma2_t, test period
    mcgarch_wfv.csv                MC-GARCH walk-forward fold metrics
    mcgarch_metadata.csv           fitted ARMA-GARCH parameters
  lstm_test_predictions.csv        canonical LSTM ensemble forecasts

final_scripts/
  mcgarch_btc_final.R              MC-GARCH estimation (rugarch)

notebooks/
  01_retrieve_bitstamp_data.ipynb        data collection via ccxt
  02_btc_thesis_main.ipynb               full pipeline, all reported results
  03_common_window_and_appendix.ipynb    robustness checks and appendix figures
```

All figures are produced by the notebooks and are visible in their saved
outputs; they are not stored separately.

---

## How to reproduce

1. `final_scripts/mcgarch_btc_final.R` — requires `data.table`, `rugarch`,
   `lubridate`. Writes the three files in `data/mc_garch_output/`.
2. `notebooks/02_btc_thesis_main.ipynb` — the full pipeline. Produces every
   result reported in the thesis, including LSTM training.
3. `notebooks/03_common_window_and_appendix.ipynb` — common-window robustness
   checks and the appendix figures.

`notebooks/01_retrieve_bitstamp_data.ipynb` regenerates `data/raw/` from the
exchange API and does not need to be run to reproduce the results.

The notebooks were run on Kaggle (GPU enabled for the LSTM) and read their
inputs from `/kaggle/input/`; adjust the paths at the top of each notebook to
point at `data/` when running them elsewhere.

### Reproducibility

Everything except the LSTM is deterministic and reproduces exactly. LSTM
training on GPU is not bitwise reproducible: cuDNN's parallelised reductions
sum floating-point values in an order that varies between runs, and
`tf.config.experimental.enable_op_determinism()` is not enabled because of its
cost in training time. Across independent re-runs the direction of every
result is stable, but the exact metrics move slightly.

All LSTM figures reported in the thesis therefore come from a **single
canonical run, fixed before hypothesis testing**, preserved with its outputs
in `02_btc_thesis_main.ipynb` and exported to `data/lstm_test_predictions.csv`.

Notebook `03` deliberately does not retrain the network. It re-executes the
deterministic part of the pipeline from scratch, loads the canonical LSTM
forecasts from that CSV, and verifies by assertion that they belong to the
same data build. That re-execution reproduces the reported figures exactly,
including the full Diebold–Mariano table to four decimal places. Re-running
`02` will produce slightly different LSTM numbers from those printed in the
thesis; this is expected and documented in the thesis.

---

## Environment

Python: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`,
`tensorflow`, `shap`, `matplotlib`, `ccxt`.
R: `rugarch`, `data.table`, `lubridate`.
