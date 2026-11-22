"""Param search for lightgbm_model.py (reuses its feature pipeline up to the split)."""
import time, numpy as np, lightgbm as lgb
src = open(__file__.replace("tune_lightgbm", "lightgbm_model"), encoding="utf-8").read()
exec(compile(src.split("params = dict(")[0].replace("__file__", repr(__file__.replace("tune_lightgbm", "lightgbm_model"))), "feat", "exec"))

base = dict(feature_pre_filter=False, objective="regression", metric="rmse", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=0)
grid = [
    {"num_leaves": 95}, {"num_leaves": 159},
    {"min_data_in_leaf": 20}, {"min_data_in_leaf": 100}, {"min_data_in_leaf": 400},
    {"feature_fraction": 0.5}, {"feature_fraction": 0.65}, {"feature_fraction": 0.95},
    {"lambda_l2": 10}, {"lambda_l2": 50},
    {"objective": "huber", "alpha": 1.0},
    {"max_bin": 511},
    {"num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.65, "lambda_l2": 10},
]
res = []
for g in grid:
    p = {**base, **g}; t = time.time()
    dtr = lgb.Dataset(tr[FEATURES], tr["y"], params=p); dva = lgb.Dataset(va[FEATURES], va["y"], reference=dtr, params=p)
    m = lgb.train(p, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    pr = np.clip(m.predict(va[FEATURES], num_iteration=m.best_iteration), 0, None)
    s = np.sqrt(np.mean((pr - va["y"]) ** 2)); res.append((s, g, m.best_iteration))
    print(f"{s:.5f} iter={m.best_iteration} {g} ({time.time()-t:.0f}s)", flush=True)
print("BEST", min(res, key=lambda r: r[0]))
