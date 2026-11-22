"""LightGBM model for Store Sales - Time Series Forecasting (RMSLE metric).

Strategy: train on log1p(sales); every lag/rolling feature uses a shift >= 16 days
(the test horizon), so the test set can be predicted in one pass with no recursion.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

D = Path(__file__).parent / "Dataset"
HORIZON = 16

train = pd.read_csv(D / "train.csv", parse_dates=["date"])
test = pd.read_csv(D / "test.csv", parse_dates=["date"])
stores = pd.read_csv(D / "stores.csv")
oil = pd.read_csv(D / "oil.csv", parse_dates=["date"])
hol = pd.read_csv(D / "holidays_events.csv", parse_dates=["date"])
trans = pd.read_csv(D / "transactions.csv", parse_dates=["date"])

train["is_test"] = 0
test["is_test"] = 1
df = pd.concat([train, test], ignore_index=True)
df["y"] = np.log1p(df["sales"])

# ---- external features ----
full_dates = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())})
oil = full_dates.merge(oil, on="date", how="left")
oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().bfill().ffill()
oil["oil_ma7"] = oil["dcoilwtico"].rolling(7, min_periods=1).mean()
df = df.merge(oil, on="date", how="left").merge(stores, on="store_nbr", how="left")
df = df.rename(columns={"type": "store_type"})

# national holidays (not transferred; transferred ones appear as "Transfer" type)
h = hol[~hol.transferred]
h = h[h.type != "Work Day"]
nat = h[h.locale == "National"].drop_duplicates("date")[["date"]].assign(nat_hol=1)
reg = h[h.locale == "Regional"][["date", "locale_name"]].assign(reg_hol=1).drop_duplicates()
loc = h[h.locale == "Local"][["date", "locale_name"]].assign(loc_hol=1).drop_duplicates()
df = df.merge(nat, on="date", how="left")
df = df.merge(reg, left_on=["date", "state"], right_on=["date", "locale_name"], how="left").drop(columns="locale_name")
df = df.merge(loc, left_on=["date", "city"], right_on=["date", "locale_name"], how="left").drop(columns="locale_name")
for c in ["nat_hol", "reg_hol", "loc_hol"]:
    df[c] = df[c].fillna(0).astype(int)

# ---- calendar ----
d = df["date"]
df["dow"] = d.dt.dayofweek
df["dom"] = d.dt.day
df["month"] = d.dt.month
df["year"] = d.dt.year
df["doy"] = d.dt.dayofyear
df["payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype(int)
df["days_to_xmas"] = (pd.to_datetime(d.dt.year.astype(str) + "-12-25") - d).dt.days
df["earthquake"] = ((d >= "2016-04-16") & (d <= "2016-05-16")).astype(int)

# ---- lag / rolling features (all >= HORIZON) ----
df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
g = df.groupby(["store_nbr", "family"])["y"]
for lag in [16, 17, 21, 28, 35, 364]:
    df[f"lag_{lag}"] = g.shift(lag)
shifted = g.shift(HORIZON)
grp = shifted.groupby([df.store_nbr, df.family])
for w in [7, 14, 28, 56]:
    df[f"rmean_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
df["rstd_14"] = grp.transform(lambda s: s.rolling(14, min_periods=2).std())
df["promo_ma7"] = df.groupby(["store_nbr", "family"])["onpromotion"].transform(
    lambda s: s.rolling(7, min_periods=1).mean())

for c in ["family", "city", "state", "store_type"]:
    df[c] = df[c].astype("category")

FEATURES = [
    "store_nbr", "family", "onpromotion", "promo_ma7", "city", "state", "store_type", "cluster",
    "dcoilwtico", "oil_ma7", "nat_hol", "reg_hol", "loc_hol",
    "dow", "dom", "month", "year", "doy", "payday", "days_to_xmas", "earthquake",
] + [c for c in df.columns if c.startswith(("lag_", "rmean_", "rstd_"))]

# ---- split: validate on the last 16 days of train ----
last_train = train.date.max()
val_start = last_train - pd.Timedelta(days=HORIZON - 1)
tr = df[(df.is_test == 0) & (df.date < val_start) & (df.date >= "2015-01-01")]
va = df[(df.is_test == 0) & (df.date >= val_start)]

params = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=42)
dtr = lgb.Dataset(tr[FEATURES], tr["y"])
dva = lgb.Dataset(va[FEATURES], va["y"], reference=dtr)
model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
pred_va = np.clip(model.predict(va[FEATURES], num_iteration=model.best_iteration), 0, None)
print(f"Validation RMSLE: {np.sqrt(np.mean((pred_va - va['y']) ** 2)):.5f}  (best iter {model.best_iteration})")

imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print("\nTop features (gain):\n", imp.head(10).round(0))
