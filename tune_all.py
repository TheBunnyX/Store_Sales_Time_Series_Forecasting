"""Pick the best hyper-parameters for each algorithm (LightGBM, XGBoost-GPU, CatBoost-GPU, RandomForest).

All models use the best feature set found so far (lightgbm_v2 config "s1": complete day grid + leading-zero trim)
and the same train / validation split (validation = last 16 days). For each algorithm a small explicit search is
run, the best config is refit on all data, and val/test predictions are saved to preds_<algo>_tuned.npz.
Results -> best_params.json
"""
import json, sys, time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
import lightgbm_v2 as v2

ROOT = Path(__file__).parent
HZ = v2.HZ
t0 = time.time()

# ---------------------------------------------------------------- data (config s1)
df, ser, static, ND = v2.build_base(newhol=False)
day, is_test = df.day.values, df.is_test.values
real = is_test == 0
Y = np.full((54 * 33, ND), np.nan, "float32"); Y[ser[real], day[real]] = df.y.values[real]
pos = np.where(np.nan_to_num(Y) > 0, np.arange(ND)[None, :], ND + 1).min(1); pos[pos == ND + 1] = 0
Y[np.arange(ND)[None, :] < pos[:, None]] = np.nan
grid = v2.Grid(Y)
last_train = int(day[real].max()); val_day0 = last_train - HZ + 1
in_train = real & (df.date.values >= np.datetime64("2015-01-01")) & (day >= pos[ser])
tr_p, full_p = np.where(in_train & (day < val_day0))[0], np.where(in_train)[0]
def order(mask):
    p = np.where(mask)[0]
    return p[np.lexsort((day[p], df.fam_str.values[p], df.store_nbr.values[p]))]
va_p, te_p = order(real & (day >= val_day0)), order(~real)
yv, ytr, yfu = df.y.values[va_p], df.y.values[tr_p], df.y.values[full_p]
statics = df[static]
X = lambda p: pd.concat([statics.iloc[p].reset_index(drop=True), grid.feats(ser[p], day[p], HZ)], axis=1)
Xtr, Xva, Xfu, Xte = X(tr_p), X(va_p), X(full_p), X(te_p)
CATS = ["family", "city", "state", "store_type"]
rm = lambda p: float(np.sqrt(np.mean((np.clip(p, 0, None) - yv) ** 2)))
print(f"data ready: train {len(tr_p)}, val {len(va_p)}, test {len(te_p)} ({time.time()-t0:.0f}s)", flush=True)

best = {}
def save(name, val, test, params, score, extra=None):
    np.savez(ROOT / f"preds_{name}_tuned.npz", val=np.clip(val, 0, None), test=np.clip(test, 0, None))
    best[name] = {"val_rmsle": round(score, 5), "params": params, **(extra or {})}
    (ROOT / "best_params.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    print(f"==> {name} BEST val RMSLE {score:.5f} {params} ({time.time()-t0:.0f}s)", flush=True)

def search(name, base, variants, trial):
    res = []
    for v in variants:
        p = {**base, **v}; s, it, _ = trial(p)
        res.append((s, p, it)); print(f"  [{name}] {s:.5f} iter={it} {v}", flush=True)
    return min(res, key=lambda r: r[0])

# ---------------------------------------------------------------- LightGBM (CPU)
LB = dict(objective="regression", metric="rmse", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42,
          feature_pre_filter=False)
def lgb_trial(p):
    m = lgb.train(p, lgb.Dataset(Xtr, ytr, params=p), 2000, valid_sets=[lgb.Dataset(Xva, yv, params=p)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return rm(m.predict(Xva, num_iteration=m.best_iteration)), m.best_iteration, m
lgb_grid = [{}, {"num_leaves": 63}, {"num_leaves": 255}, {"min_data_in_leaf": 100}, {"min_data_in_leaf": 200},
            {"feature_fraction": 0.6}, {"feature_fraction": 0.95}, {"lambda_l2": 10}, {"lambda_l2": 50},
            {"num_leaves": 255, "min_data_in_leaf": 100}, {"num_leaves": 255, "min_data_in_leaf": 200, "lambda_l2": 10},
            {"num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.6}]
s, p, _ = search("lightgbm", LB, lgb_grid, lgb_trial)
pf = {**p, "learning_rate": 0.05}                       # final: slower learning rate
m = lgb.train(pf, lgb.Dataset(Xtr, ytr, params=pf), 3000, valid_sets=[lgb.Dataset(Xva, yv, params=pf)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
val = m.predict(Xva, num_iteration=m.best_iteration)
f = lgb.train(pf, lgb.Dataset(Xfu, yfu, params=pf), int(m.best_iteration * 1.1))
save("lightgbm", val, f.predict(Xte), {k: v for k, v in pf.items() if k not in ("verbose", "feature_pre_filter")},
     rm(val), {"rounds": int(m.best_iteration * 1.1)})

# ---------------------------------------------------------------- XGBoost (GPU)
XB = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.1, max_depth=8, min_child_weight=50,
          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", device="cuda", max_cat_to_onehot=1, seed=42)
dxtr, dxva = xgb.DMatrix(Xtr, ytr, enable_categorical=True), xgb.DMatrix(Xva, yv, enable_categorical=True)
def xgb_trial(p):
    m = xgb.train(p, dxtr, 2000, evals=[(dxva, "v")], early_stopping_rounds=50, verbose_eval=False)
    n = m.best_iteration + 1
    return rm(m.predict(dxva, iteration_range=(0, n))), n, m
xgb_grid = [{}, {"max_depth": 6}, {"max_depth": 10}, {"max_depth": 12}, {"min_child_weight": 10}, {"min_child_weight": 200},
            {"colsample_bytree": 0.5}, {"colsample_bytree": 1.0}, {"subsample": 0.6}, {"reg_lambda": 10}, {"reg_lambda": 50},
            {"max_depth": 10, "min_child_weight": 100, "reg_lambda": 10}, {"max_depth": 10, "colsample_bytree": 0.6, "reg_lambda": 10}]
s, p, _ = search("xgboost", XB, xgb_grid, xgb_trial)
pf = {**p, "learning_rate": 0.05}
m = xgb.train(pf, dxtr, 4000, evals=[(dxva, "v")], early_stopping_rounds=100, verbose_eval=False)
n = m.best_iteration + 1; val = m.predict(dxva, iteration_range=(0, n))
f = xgb.train(pf, xgb.DMatrix(Xfu, yfu, enable_categorical=True), int(n * 1.1))
save("xgboost", val, f.predict(xgb.DMatrix(Xte, enable_categorical=True)),
     {k: v for k, v in pf.items() if k not in ("eval_metric",)}, rm(val), {"rounds": int(n * 1.1)})

# ---------------------------------------------------------------- CatBoost (GPU)
def cat_prep(d):
    d = d.copy()
    for c in CATS: d[c] = d[c].astype(str)
    return d
Ctr, Cva, Cfu, Cte = map(cat_prep, (Xtr, Xva, Xfu, Xte))
CB = dict(loss_function="RMSE", learning_rate=0.1, depth=8, l2_leaf_reg=3, iterations=3000, random_seed=42, verbose=0,
          task_type="GPU", od_type="Iter", od_wait=100)
ptr, pva = Pool(Ctr, ytr, cat_features=CATS), Pool(Cva, yv, cat_features=CATS)
def cat_trial(p):
    m = CatBoostRegressor(**p).fit(ptr, eval_set=pva, use_best_model=True)
    return rm(m.predict(pva)), m.get_best_iteration() + 1, m
cat_grid = [{}, {"depth": 6}, {"depth": 10}, {"depth": 10, "l2_leaf_reg": 10}, {"l2_leaf_reg": 10}, {"l2_leaf_reg": 30},
            {"learning_rate": 0.05}, {"border_count": 254}, {"depth": 10, "learning_rate": 0.05, "l2_leaf_reg": 10}]
s, p, it = search("catboost", CB, cat_grid, cat_trial)
pc = {k: v for k, v in p.items() if k not in ("od_type", "od_wait")}
f = CatBoostRegressor(**{**pc, "iterations": int(it * 1.1)}).fit(Pool(Cfu, yfu, cat_features=CATS))
m = CatBoostRegressor(**p).fit(ptr, eval_set=pva, use_best_model=True)
save("catboost", m.predict(pva), f.predict(Pool(Cte, cat_features=CATS)),
     {k: v for k, v in p.items() if k not in ("verbose",)}, s, {"rounds": int(it * 1.1)})

# ---------------------------------------------------------------- Random Forest (CPU)
def rf_X(d):
    d = d.copy()
    for c in CATS: d[c] = d[c].cat.codes
    return d.astype("float32")
Rtr, Rva, Rfu, Rte = map(rf_X, (Xtr, Xva, Xfu, Xte))
RB = dict(n_estimators=100, min_samples_leaf=20, max_features=0.5, max_samples=0.4, n_jobs=-1, random_state=42)
def rf_trial(p):
    m = RandomForestRegressor(**p).fit(Rtr, ytr); return rm(m.predict(Rva)), p["n_estimators"], m
rf_grid = [{}, {"min_samples_leaf": 5}, {"min_samples_leaf": 50}, {"max_features": 0.33}, {"max_features": 0.8},
           {"max_samples": 0.7}, {"min_samples_leaf": 5, "max_features": 0.8, "max_samples": 0.7}]
s, p, _ = search("rf", RB, rf_grid, rf_trial)
p = {**p, "n_estimators": 250}
m = RandomForestRegressor(**p).fit(Rtr, ytr); val = m.predict(Rva)
f = RandomForestRegressor(**p).fit(Rfu, yfu)
save("rf", val, f.predict(Rte), p, rm(val))
print("ALL DONE", f"{time.time()-t0:.0f}s", flush=True)
