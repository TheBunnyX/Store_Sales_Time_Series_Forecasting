"""LightGBM + XGBoost + CatBoost ensemble (reuses the feature pipeline in lightgbm_model.py).
Each model: early-stop on the last-16-day validation, refit on all data, predict test; blend = mean in log space."""
import time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
P = Path(__file__).parent / "lightgbm_model.py"
exec(compile(P.read_text(encoding="utf-8").split("params = dict(")[0].replace("__file__", repr(str(P))), "feat", "exec"))

full = df[(df.is_test == 0) & (df.date >= "2015-01-01")]
te = df[df.is_test == 1]
CATS = ["family", "city", "state", "store_type"]
rmsle = lambda p: np.sqrt(np.mean((p - va["y"].values) ** 2))
vals, tests = {}, {}

def run_lgb():
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
    dtr = lgb.Dataset(tr[FEATURES], tr["y"]); dva = lgb.Dataset(va[FEATURES], va["y"], reference=dtr)
    m = lgb.train(p, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    v = m.predict(va[FEATURES], num_iteration=m.best_iteration)
    f = lgb.train(p, lgb.Dataset(full[FEATURES], full["y"]), int(m.best_iteration * 1.1))
    return v, f.predict(te[FEATURES]), m.best_iteration

def run_xgb():
    p = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.05, max_depth=8, min_child_weight=50,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", device="cuda", enable_categorical=True,
             max_cat_to_onehot=1, seed=42)
    dtr = xgb.DMatrix(tr[FEATURES], tr["y"], enable_categorical=True)
    dva = xgb.DMatrix(va[FEATURES], va["y"], enable_categorical=True)
    m = xgb.train(p, dtr, 2000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    n = m.best_iteration + 1
    v = m.predict(dva, iteration_range=(0, n))
    f = xgb.train(p, xgb.DMatrix(full[FEATURES], full["y"], enable_categorical=True), int(n * 1.1))
    return v, f.predict(xgb.DMatrix(te[FEATURES], enable_categorical=True)), n

def run_linear():
    """Linear model: one-hot categoricals (+ family x weekday, family x store) + scaled numeric lags; Ridge(alpha=1)
    is ordinary least squares with a tiny penalty so the collinear one-hot columns stay solvable."""
    ohe_cols = CATS + ["store_nbr", "dow", "month", "cluster", "fam_dow", "fam_store"]
    num_cols = [c for c in FEATURES if c not in CATS + ["store_nbr", "cluster", "dow", "month"]]
    def prep(d):
        d = d[FEATURES].copy()
        for c in CATS: d[c] = d[c].astype(str)
        d["fam_dow"] = d["family"] + "_" + d["dow"].astype(str)
        d["fam_store"] = d["family"] + "_" + d["store_nbr"].astype(str)
        for c in ["store_nbr", "dow", "month", "cluster"]: d[c] = d[c].astype(str)
        return d
    def make():
        ct = ColumnTransformer([
            ("oh", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols)])
        return make_pipeline(ct, Ridge(alpha=1.0, solver="sparse_cg", max_iter=500, random_state=0))
    m = make().fit(prep(tr), tr["y"])
    v = m.predict(prep(va))
    f = make().fit(prep(full), full["y"])
    return v, f.predict(prep(te)), 0

def run_cat():
    def prep(d):
        d = d[FEATURES].copy()
        for c in CATS: d[c] = d[c].astype(str)
        return d
    p = dict(loss_function="RMSE", learning_rate=0.1, depth=8, l2_leaf_reg=3, iterations=2000,
             random_seed=42, verbose=0, task_type="GPU", od_type="Iter", od_wait=100)
    m = CatBoostRegressor(**p)
    m.fit(Pool(prep(tr), tr["y"], cat_features=CATS), eval_set=Pool(prep(va), va["y"], cat_features=CATS),
          use_best_model=True)
    n = m.get_best_iteration() + 1
    v = m.predict(prep(va))
    p2 = {**p, "iterations": int(n * 1.1)}; p2.pop("od_type"); p2.pop("od_wait")
    f = CatBoostRegressor(**p2).fit(Pool(prep(full), full["y"], cat_features=CATS))
    return v, f.predict(prep(te)), n

for name, fn in [("lightgbm", run_lgb), ("xgboost", run_xgb), ("catboost", run_cat)]:
    t = time.time(); v, f, n = fn()
    vals[name], tests[name] = np.clip(v, 0, None), np.clip(f, 0, None)
    np.savez(Path(__file__).parent / f"preds_{name}.npz", val=vals[name], test=tests[name])
    print(f"{name}: val RMSLE {rmsle(vals[name]):.5f} (iter {n}, {time.time()-t:.0f}s)", flush=True)

np.savez(Path(__file__).parent / "ensemble_preds.npz", **{f"val_{k}": v for k, v in vals.items()}, **{f"test_{k}": v for k, v in tests.items()}, y_val=va["y"].values)
blend_v = np.mean(list(vals.values()), axis=0)
print(f"BLEND (mean of all): val RMSLE {rmsle(blend_v):.5f}")
for a in vals:
    others = [k for k in vals if k != a]
    print(f"  blend w/o {a}: {rmsle(np.mean([vals[k] for k in others], axis=0)):.5f}")
