"""LightGBM v2 - experiments on top of lightgbm_model.py.

  (1) trim   : drop each series' leading zero-sales period (before the store/family really starts selling),
               both as training rows and as lag/rolling inputs.
  (2) direct : one model per forecast horizon h=1..16 using lags measured from the last known day
               (instead of one model whose lags are all >= 16 days old).
  (3) hol    : richer holiday features (Additional / Bridge / Work Day / Event flags, days to/since a national holiday).
All lag features are computed on a complete [series x day] grid, so a shift of k rows is exactly k days
(the baseline's row shifts were off around the missing Dec-25 rows).

Usage: python lightgbm_v2.py s0 s1 s2 d3   (any subset of the configs below)
Outputs: preds_<name>.npz (val/test arrays in the same row order as the other preds_*.npz)
"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).parent
D = ROOT / "Dataset"
HZ = 16
CONFIGS = {  # name: (trim, newhol, direct)
    "s0": (False, False, False),   # grid-aligned baseline
    "s1": (True, False, False),    # + (1) trim
    "s2": (True, True, False),     # + (1) + (3)
    "d3": (True, True, True),      # + (1) + (3) + (2)
}
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)

# ------------------------------------------------------------------ base (static) features
def build_base(newhol):
    train = pd.read_csv(D / "train.csv", parse_dates=["date"]); test = pd.read_csv(D / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(D / "stores.csv"); oil = pd.read_csv(D / "oil.csv", parse_dates=["date"])
    hol = pd.read_csv(D / "holidays_events.csv", parse_dates=["date"])
    train["is_test"] = 0; test["is_test"] = 1
    df = pd.concat([train, test], ignore_index=True)
    df["y"] = np.log1p(df["sales"])
    df["fam_str"] = df["family"].astype(str)

    fd = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max())})
    oil = fd.merge(oil, on="date", how="left")
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().bfill().ffill()
    oil["oil_ma7"] = oil["dcoilwtico"].rolling(7, min_periods=1).mean()
    df = df.merge(oil, on="date", how="left").merge(stores, on="store_nbr", how="left").rename(columns={"type": "store_type"})

    h = hol[~hol.transferred]
    if not newhol:
        h = h[h.type != "Work Day"]
        nat = h[h.locale == "National"].drop_duplicates("date")[["date"]].assign(nat_hol=1)
        reg = h[h.locale == "Regional"][["date", "locale_name"]].assign(reg_hol=1).drop_duplicates()
        loc = h[h.locale == "Local"][["date", "locale_name"]].assign(loc_hol=1).drop_duplicates()
    else:
        hh = h[h.type.isin(["Holiday", "Transfer", "Additional", "Bridge"])]
        nat = hh[hh.locale == "National"].drop_duplicates("date")[["date"]].assign(nat_hol=1)
        reg = hh[hh.locale == "Regional"][["date", "locale_name"]].assign(reg_hol=1).drop_duplicates()
        loc = hh[hh.locale == "Local"][["date", "locale_name"]].assign(loc_hol=1).drop_duplicates()
    df = df.merge(nat, on="date", how="left")
    df = df.merge(reg, left_on=["date", "state"], right_on=["date", "locale_name"], how="left").drop(columns="locale_name")
    df = df.merge(loc, left_on=["date", "city"], right_on=["date", "locale_name"], how="left").drop(columns="locale_name")
    extra = []
    if newhol:
        for t, nm in [("Additional", "nat_add"), ("Bridge", "nat_bridge"), ("Event", "nat_event"), ("Work Day", "nat_work")]:
            s = h[(h.locale == "National") & (h.type == t)].drop_duplicates("date")[["date"]].assign(**{nm: 1})
            df = df.merge(s, on="date", how="left"); extra.append(nm)
    for c in ["nat_hol", "reg_hol", "loc_hol"] + extra:
        df[c] = df[c].fillna(0).astype(int)

    D0 = df.date.min()
    df["day"] = (df.date - D0).dt.days
    if newhol:
        alld = pd.date_range(D0, df.date.max()); fl = alld.isin(nat.date)
        idx = np.arange(len(alld))
        nxt = np.minimum.accumulate(np.where(fl, idx, np.inf)[::-1])[::-1] - idx
        prv = idx - np.maximum.accumulate(np.where(fl, idx, -np.inf))
        df["days_to_nat"] = np.clip(nxt[df.day.values], 0, 14); df["days_since_nat"] = np.clip(prv[df.day.values], 0, 14)
        df["hol_any"] = ((df.nat_hol + df.reg_hol + df.loc_hol) > 0).astype(int)
        extra += ["days_to_nat", "days_since_nat", "hol_any"]

    d = df["date"]
    df["dow"] = d.dt.dayofweek; df["dom"] = d.dt.day; df["month"] = d.dt.month; df["year"] = d.dt.year
    df["doy"] = d.dt.dayofyear
    df["payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype(int)
    df["days_to_xmas"] = (pd.to_datetime(d.dt.year.astype(str) + "-12-25") - d).dt.days
    df["earthquake"] = ((d >= "2016-04-16") & (d <= "2016-05-16")).astype(int)

    df["family"] = df["family"].astype("category")
    ser = ((df.store_nbr.astype(int) - 1) * 33 + df.family.cat.codes.astype(int)).values
    for c in ["city", "state", "store_type"]:
        df[c] = df[c].astype("category")
    ND = int(df.day.max()) + 1
    PRM = np.zeros((54 * 33, ND), "float32"); PRM[ser, df.day.values] = df.onpromotion.values
    cs = np.cumsum(PRM, 1); dy = df.day.values
    df["promo_ma7"] = (cs[ser, dy] - np.where(dy - 7 >= 0, cs[ser, np.maximum(dy - 7, 0)], 0)) / np.minimum(dy + 1, 7)

    static = ["store_nbr", "family", "onpromotion", "promo_ma7", "city", "state", "store_type", "cluster", "dcoilwtico",
              "oil_ma7", "nat_hol", "reg_hol", "loc_hol", "dow", "dom", "month", "year", "doy", "payday",
              "days_to_xmas", "earthquake"] + extra
    return df, ser, static, ND

# ------------------------------------------------------------------ lag features on the [series x day] grid
class Grid:
    def __init__(self, Yu):
        self.Y = Yu
        Z = np.nan_to_num(Yu); self.cs = np.cumsum(Z, 1); self.cs2 = np.cumsum(Z * Z, 1)
        self.cc = np.cumsum(~np.isnan(Yu), 1).astype("float32")

    def feats(self, s, d, h):
        n, out = len(d), {}
        def lag(k):
            idx = d - k; ok = idx >= 0
            v = np.full(n, np.nan, "float32"); v[ok] = self.Y[s[ok], idx[ok]]; return v
        m = 7 * ((h + 6) // 7)                       # smallest multiple of 7 >= h (same weekday as target)
        for j, k in enumerate([h, h + 1, h + 2]): out[f"lag_r{j}"] = lag(k)
        for j in range(4): out[f"lag_w{j+1}"] = lag(m + 7 * j)
        out["lag_y1"] = lag(364)
        e = d - h; okE = e >= 0; ec = np.maximum(e, 0)
        def win(arr, w):
            b = np.where(e - w >= 0, arr[s, np.maximum(e - w, 0)], 0)
            return arr[s, ec] - b
        for w in [7, 14, 28, 56]:
            cnt = win(self.cc, w); sm = win(self.cs, w)
            out[f"rmean_{w}"] = np.where(okE & (cnt > 0), sm / np.maximum(cnt, 1), np.nan).astype("float32")
            if w == 14:
                s2 = win(self.cs2, w); var = s2 / np.maximum(cnt, 1) - (sm / np.maximum(cnt, 1)) ** 2
                out["rstd_14"] = np.where(okE & (cnt >= 2), np.sqrt(np.maximum(var, 0)), np.nan).astype("float32")
        return pd.DataFrame(out)

# ------------------------------------------------------------------ experiment
def rmsle(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))

def run(name):
    trim, newhol, direct = CONFIGS[name]
    t0 = time.time()
    df, ser, static, ND = build_base(newhol)
    day, is_test = df.day.values, df.is_test.values
    real = is_test == 0
    Y = np.full((54 * 33, ND), np.nan, "float32"); Y[ser[real], day[real]] = df.y.values[real]
    pos = np.where(np.nan_to_num(Y) > 0, np.arange(ND)[None, :], ND + 1).min(1)
    pos[pos == ND + 1] = 0                                  # series that never sell: keep as is
    Yu = Y.copy()
    if trim: Yu[np.arange(ND)[None, :] < pos[:, None]] = np.nan
    grid = Grid(Yu)

    last_train = int(day[real].max()); val_day0 = last_train - HZ + 1
    in_train = real & (df.date.values >= np.datetime64("2015-01-01"))
    if trim: in_train &= day >= pos[ser]
    tr_m = in_train & (day < val_day0); full_m = in_train
    va_m = real & (day >= val_day0); te_m = ~real

    def order(mask):  # row positions in the same order as the other preds_*.npz (store, family(str), date)
        p = np.where(mask)[0]
        return p[np.lexsort((day[p], df.fam_str.values[p], df.store_nbr.values[p]))]
    va_p, te_p = order(va_m), order(te_m)
    y_va = df.y.values[va_p]
    statics = df[static]

    def X_for(pos_, h):
        return pd.concat([statics.iloc[pos_].reset_index(drop=True), grid.feats(ser[pos_], day[pos_], h)], axis=1)

    if not direct:
        tr_p, full_p = np.where(tr_m)[0], np.where(full_m)[0]
        Xtr, Xva = X_for(tr_p, HZ), X_for(va_p, HZ)
        dtr = lgb.Dataset(Xtr, df.y.values[tr_p]); dva = lgb.Dataset(Xva, y_va, reference=dtr)
        m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        val = np.clip(m.predict(Xva, num_iteration=m.best_iteration), 0, None)
        f = lgb.train(PARAMS, lgb.Dataset(X_for(full_p, HZ), df.y.values[full_p]), int(m.best_iteration * 1.1))
        test = np.clip(f.predict(X_for(te_p, HZ)), 0, None)
        info = f"iter {m.best_iteration}"
    else:
        p = {**PARAMS, "learning_rate": 0.08}; R = 600
        tr_p, full_p = np.where(tr_m)[0], np.where(full_m)[0]
        models, curves, va_h = {}, [], {}
        for h in range(1, HZ + 1):
            sel = va_p[day[va_p] == val_day0 - 1 + h]
            Xtr, Xv = X_for(tr_p, h), X_for(sel, h)
            dtr = lgb.Dataset(Xtr, df.y.values[tr_p]); dv = lgb.Dataset(Xv, df.y.values[sel], reference=dtr)
            res = {}
            models[h] = lgb.train(p, dtr, R, valid_sets=[dv], callbacks=[lgb.record_evaluation(res)])
            curves.append(np.array(res["valid_0"]["rmse"]) ** 2); va_h[h] = (sel, Xv)
            print(f"  [{name}] horizon {h}/{HZ} trained, val RMSLE@{R} {np.sqrt(curves[-1][-1]):.5f} ({time.time()-t0:.0f}s)", flush=True)
        best = int(np.argmin(np.mean(curves, 0))) + 1
        pred = {}
        for h, (sel, Xv) in va_h.items():
            for pos_, v in zip(sel, models[h].predict(Xv, num_iteration=best)): pred[pos_] = v
        val = np.clip(np.array([pred[q] for q in va_p]), 0, None)
        tpred = {}
        for h in range(1, HZ + 1):
            sel = te_p[day[te_p] == last_train + h]
            f = lgb.train(p, lgb.Dataset(X_for(full_p, h), df.y.values[full_p]), int(best * 1.1))
            for pos_, v in zip(sel, f.predict(X_for(sel, h))): tpred[pos_] = v
            print(f"  [{name}] refit horizon {h}/{HZ} ({time.time()-t0:.0f}s)", flush=True)
        test = np.clip(np.array([tpred[q] for q in te_p]), 0, None)
        info = f"pooled best round {best}"
    np.savez(ROOT / f"preds_{name}.npz", val=val, test=test)
    print(f"{name} (trim={trim}, newhol={newhol}, direct={direct}): val RMSLE {rmsle(val, y_va):.5f}  [{info}, {time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(CONFIGS)): run(n)
