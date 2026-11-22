"""Random Forest (scikit-learn, CPU). Reuses lightgbm_model.py's features and train/validation split."""
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
P = Path(__file__).parent / "lightgbm_model.py"
exec(compile(P.read_text(encoding="utf-8").split("params = dict(")[0].replace("__file__", repr(str(P))), "feat", "exec"))

CATS = ["family", "city", "state", "store_type"]
def X(d):
    d = d[FEATURES].copy()
    for c in CATS: d[c] = d[c].cat.codes
    return d.astype("float32")          # sklearn's RF handles NaN natively
full = df[(df.is_test == 0) & (df.date >= "2015-01-01")]
te = df[df.is_test == 1]
mk = lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.5, max_samples=0.4,
                                   n_jobs=-1, random_state=42)
t = time.time()
rf = mk().fit(X(tr), tr["y"])
val = np.clip(rf.predict(X(va)), 0, None)
print(f"RandomForest: val RMSLE {np.sqrt(np.mean((val - va['y'].values) ** 2)):.5f} ({time.time()-t:.0f}s)", flush=True)
test = np.clip(mk().fit(X(full), full["y"]).predict(X(te)), 0, None)
np.savez(Path(__file__).parent / "preds_rf.npz", val=val, test=test)
print("Saved preds_rf.npz")
