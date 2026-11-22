# Store Sales — Time Series Forecasting

Forecast daily sales for **54 stores × 33 product families (1,782 series)** of Corporación Favorita, **16 days ahead**.
The metric is **RMSLE** (lower is better). Competition: [store-sales-time-series-forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) on Kaggle.
The original competition text is kept in [COMPETITION.md](COMPETITION.md).

This project compares LightGBM, XGBoost, CatBoost, Random Forest and an LSTM on one validation window, works out what
actually reduces the error, and presents the results as a static website ([site/](site/)).

## Results

Measured on the **last 16 days of the training data** (2017-07-31 to 2017-08-15), the same length as the real forecast horizon.

| Model | RMSLE | Hardware |
|---|---|---|
| Average of tuned LightGBM + per-horizon LightGBM | **0.38206** | CPU |
| LightGBM (tuned) | 0.38500 | CPU |
| LightGBM per-horizon (`d3`) | 0.38721 | CPU |
| LightGBM baseline (starting point) | 0.38817 | CPU |
| XGBoost (tuned) | 0.39142 | GPU |
| CatBoost (tuned) | 0.39998 | GPU |
| Random Forest (tuned) | 0.40697 | CPU |
| LSTM (2 seeds) | 0.42304 | GPU |

**Caveats**
- All figures come from a single validation window. The real Kaggle score may differ somewhat.
- The best blend was picked after looking at several combinations (11 in total), so it may look slightly better than it really is. The gap to the next-best blend is about 0.0001.
- The LSTM was not tuned and uses the older feature set without the leading-zero trim, so it is not a fair comparison against the tuned tree models.
- CatBoost on the GPU gives slightly different results between runs (0.39941 during the search versus 0.39998 in the final fit). The table uses the final fit.

### What actually reduced the error

| Step | RMSLE |
|---|---|
| Trim the leading zeros before a series really starts selling (`s0` → `s1`) | 0.39001 → 0.38571 |
| Richer holiday features (`s1` → `s2`) | 0.38571 → 0.38595 (no help, not kept) |
| Tune LightGBM hyper-parameters | 0.38571 → 0.38500 |
| Average with the per-horizon models | 0.38500 → 0.38206 |

Tried and did not help: weekday-mean features, store-level features aggregated over all families, Ridge and a Ridge + LightGBM hybrid,
and averaging with XGBoost, CatBoost, Random Forest or the LSTM (their predictions are very close to LightGBM's, with correlation 0.997–0.999).

Every model gets worse further out: LightGBM averages about 0.380 RMSLE on days 1–4 but about 0.399 on days 13–16.

## Project layout

```
Dataset/                    Kaggle data (train, test, stores, oil, holidays_events, transactions); not in git, see below
EDA.ipynb                   Exploratory analysis with charts and insights (executed; outputs are saved in the file)

lightgbm_model.py           Baseline features and the first LightGBM (other scripts reuse its feature code)
lightgbm_v2.py              Experiments s0 / s1 / s2 / d3 (leading-zero trim, richer holidays, per-horizon models)
tune_all.py                 Hyper-parameter search for LightGBM, XGBoost, CatBoost and Random Forest
tune_lightgbm.py            First LightGBM search (old feature set)
ensemble_model.py           LightGBM + XGBoost + CatBoost on the old feature set
rf_model.py                 Random Forest (old feature set)
lstm_model.py               LSTM (PyTorch, GPU)

best_params.json            Best hyper-parameters per algorithm
preds_<name>.npz            Predictions of each model (format below)
ensemble_preds.npz          Predictions from ensemble_model.py plus the validation targets

site/                       Static comparison website, with a Dockerfile and docker-compose.yml for Dokploy
```

## How the models are built

### Features
- **Core:** store, product family, `onpromotion` and its 7-day mean, store metadata (city, state, type, cluster)
- **Calendar:** weekday, day of month, month, year, payday (15th and month end), days to Christmas, the April 2016 earthquake period
- **External:** oil price (missing days interpolated), national / regional / local holidays
- **History:** lags and rolling means of `log1p(sales)` shifted by **at least 16 days**, so nothing unavailable on the real forecast day is used
  (lags 16, 17, 21, 28, 35 and 364 days; rolling means over 7, 14, 28 and 56 days)
- `transactions` is not used: it ends on the last training day and does not exist for the test period

Models are trained on `log1p(sales)` to match RMSLE, and predictions are converted back with `expm1`.

### Experiments in `lightgbm_v2.py`

| Name | What it does |
|---|---|
| `s0` | Reference: lags computed on a complete day grid (25 Dec is missing every year, so row-based shifts drift by one day) |
| `s1` | `s0` + trim the leading zeros before a series really starts selling (both training rows and lag inputs) |
| `s2` | `s1` + richer holiday features (additional days, bridge days, work days, events, days to/since a holiday) |
| `d3` | `s2` + 16 separate models, one per forecast day (1–16), each using lags measured from the last known day |

### LSTM
Reads the last 56 days of each series (ending 16 days before the target day), combines the final state with embeddings of store and
product family and with information known in advance for the target day. Prediction = mean of the last 14 days + a learned correction.

## How to run

Uses a Python environment in `.venv` created with [uv](https://docs.astral.sh/uv/) (commands below are for Windows).

```
uv venv .venv
uv pip install --python .venv/Scripts/python.exe lightgbm xgboost catboost pandas numpy scikit-learn matplotlib seaborn nbformat nbconvert ipykernel
uv pip install --python .venv/Scripts/python.exe "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

Versions used: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, torch 2.11.0+cu128

**Data:** the competition files are not stored in git (`train.csv` is about 122 MB, above GitHub's 100 MB file limit). Download them from the
[competition's Data tab](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and put them in a `Dataset/` folder next to the scripts.

Run from the project folder:

```
.venv/Scripts/python.exe lightgbm_v2.py s0 s1 s2 d3     # experiments 1-3 (about 15 minutes in total)
.venv/Scripts/python.exe tune_all.py                    # search all 4 algorithms (about 40 minutes)
.venv/Scripts/python.exe lstm_model.py                  # LSTM (needs a GPU)
.venv/Scripts/python.exe ensemble_model.py              # XGBoost + CatBoost + LightGBM on the old features
.venv/Scripts/python.exe rf_model.py                    # Random Forest on the old features
```

**GPU:** XGBoost (`device="cuda"`), CatBoost (`task_type="GPU"`) and the LSTM can use a GPU (tested on an RTX 5070 with CUDA 12.8).
The pip build of LightGBM has no GPU support, so it runs on the CPU, and so does scikit-learn's Random Forest.

### EDA
Open [EDA.ipynb](EDA.ipynb) in VS Code or Jupyter. Outputs are already saved in the file. To re-run it:

```
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute EDA.ipynb --output EDA.ipynb
```

### `preds_*.npz` format
- Every file has an array `val` (the last 16 days of the training data) and `test` (the 16 days to forecast)
- Values are on the **`log1p`** scale; use `np.expm1` to get sales back
- Rows are ordered by `(store_nbr, family, date)` in every file, so files can be averaged directly.
  `ensemble_preds.npz` also holds the validation targets as `y_val`, on the same scale
- Example: average two models and score them

```python
import numpy as np
y = np.load("ensemble_preds.npz")["y_val"]
a, b = (np.load(f"preds_{k}.npz")["val"] for k in ("lightgbm_tuned", "d3"))
print(np.sqrt(np.mean(((a + b) / 2 - y) ** 2)))   # 0.38206
```

Note: `preds_lightgbm.npz`, `preds_xgboost.npz`, `preds_catboost.npz` and `preds_rf.npz` come from the old feature set;
the `*_tuned` files and `s0`–`d3` use the feature set with the leading-zero trim.

The scripts **no longer write submission files**. If you need one, convert the `test` predictions with `expm1` and pair them with the
`id` column of `Dataset/test.csv`, sorted by `(store_nbr, family, date)`.

## Comparison website

[site/](site/) is a static website (HTML + CSS + JS, English and Thai) that walks through the method step by step and shows the ranking, charts,
a plain-language explanation of each algorithm, the reasoning behind each choice, an FAQ, a glossary and the technical details.
Run it locally with `python -m http.server 3011 --directory site`, and deploy it on Dokploy as a Compose app following [site/README.md](site/README.md).
The numbers shown on the site live in `site/assets/data.js`.

## Limitations

- Results come from one 16-day validation window; there is no rolling-origin cross-validation yet
- There is no requirements or lock file; use the package list and versions above
- Not tested on Linux or macOS (the run commands are written for Windows)
- `tune_lightgbm.py` is the first search, on the old feature set; it is kept for reference. The results actually used come from `tune_all.py`
