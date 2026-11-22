"""LSTM (PyTorch, GPU). For a target (store, family, day t) the LSTM reads the previous L days of that series
(log sales, missing-mask, promo, calendar) ending at t-16 (the forecast horizon, so no leakage); its last state is
concatenated with embeddings + known features of day t. Output = recent-mean baseline + learned correction."""
import time, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
P = Path(__file__).parent / "lightgbm_model.py"
exec(compile(P.read_text(encoding="utf-8").split("params = dict(")[0].replace("__file__", repr(str(P))), "feat", "exec"))

dev = "cuda"
full = df[(df.is_test == 0) & (df.date >= "2015-01-01")]
te = df[df.is_test == 1]
L, HZ, EPOCHS, BS, LR, SEEDS = 56, 16, 12, 2048, 2e-3, [0, 1]

# ---- dense series grid [series, day] ----
D0 = df.date.min()
day = (df.date - D0).dt.days.values
ser = ((df.store_nbr.astype(int) - 1) * 33 + df.family.cat.codes.astype(int)).values
S, ND = 54 * 33, int(day.max()) + 1
known = (df.is_test.values == 0)
Y = np.zeros((S, ND), "float32"); M = np.zeros((S, ND), "float32"); PR = np.zeros((S, ND), "float32")
Y[ser[known], day[known]] = df.y.values[known]; M[ser[known], day[known]] = 1
PR[ser, day] = np.log1p(df.onpromotion.values)

dates = pd.date_range(D0, periods=ND)
dd = df.groupby("date")[["nat_hol", "payday", "dcoilwtico"]].first().reindex(dates).ffill().bfill()
oil = ((dd.dcoilwtico - dd.dcoilwtico.mean()) / dd.dcoilwtico.std()).values
C = np.stack([np.sin(2 * np.pi * dates.dayofweek / 7), np.cos(2 * np.pi * dates.dayofweek / 7),
              dates.day / 31, dd.payday.values, dd.nat_hol.values, oil], 1).astype("float32")
Yt, Mt, Pt, Ct = [torch.tensor(a, device=dev) for a in (Y, M, PR, C)]

# ---- per-row target-day features ----
CAT = [df.store_nbr.astype(int) - 1, df.family.cat.codes, df.dow, df.month - 1, df.dom - 1,
       df.city.cat.codes, df.state.cat.codes, df.store_type.cat.codes, df.cluster.astype(int) - 1]
CARD = [54, 33, 7, 12, 31, 22, 16, 5, 17]
NUMF = np.stack([np.log1p(df.onpromotion.values), np.log1p(df.promo_ma7.values), df.dcoilwtico.values, df.nat_hol.values,
                 df.reg_hol.values, df.loc_hol.values, df.payday.values, df.days_to_xmas.values / 365.0,
                 df.earthquake.values], 1).astype("float32")
mu, sd = NUMF[tr.index.values].mean(0), NUMF[tr.index.values].std(0) + 1e-6
NUMF = (NUMF - mu) / sd
CATt = torch.tensor(np.stack([np.asarray(c).astype("int64") for c in CAT], 1), device=dev)
NUMt = torch.tensor(NUMF, device=dev)
SER = torch.tensor(ser, device=dev); START = torch.tensor(day - HZ - L + 1, device=dev)
YT = torch.tensor(np.nan_to_num(df.y.values).astype("float32"), device=dev)
AR = torch.arange(L, device=dev)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, min(16, n // 2 + 1)) for n in CARD])
        self.lstm = nn.LSTM(3 + C.shape[1], 128, num_layers=2, batch_first=True, dropout=0.1)
        d = 128 + sum(e.embedding_dim for e in self.embs) + NUMF.shape[1]
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, idx):
        di = START[idx][:, None] + AR[None, :]; s = SER[idx][:, None]
        y, m, p = Yt[s, di], Mt[s, di], Pt[s, di]
        x = torch.cat([(y / 5).unsqueeze(-1), m.unsqueeze(-1), p.unsqueeze(-1), Ct[di]], -1)
        h = self.lstm(x)[0][:, -1]
        cat = CATt[idx]
        z = torch.cat([h] + [e(cat[:, i]) for i, e in enumerate(self.embs)] + [NUMt[idx]], 1)
        base = (y[:, -14:] * m[:, -14:]).sum(1) / m[:, -14:].sum(1).clamp(min=1)
        return base + self.head(z).squeeze(1)

def predict(net, idx):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 16384): out.append(net(idx[i:i + 16384]).float().cpu().numpy())
    return np.clip(np.concatenate(out), 0, None)

def fit(train_df, seed, evaluate):
    torch.manual_seed(seed)
    tri = torch.tensor(train_df.index.values, device=dev); vai = torch.tensor(va.index.values, device=dev)
    tei = torch.tensor(te.index.values, device=dev)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * ((len(tri) + BS - 1) // BS))
    for ep in range(EPOCHS):
        net.train(); perm = tri[torch.randperm(len(tri), device=dev)]; t = time.time()
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            loss = nn.functional.mse_loss(net(b), YT[b]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if evaluate:
            pv = predict(net, vai)
            print(f"  seed {seed} epoch {ep+1}/{EPOCHS}: val RMSLE {np.sqrt(np.mean((pv - va['y'].values) ** 2)):.5f} ({time.time()-t:.0f}s)", flush=True)
    return predict(net, vai), predict(net, tei)

vs, ts = [], []
for s in SEEDS:
    vs.append(fit(tr, s, True)[0]); ts.append(fit(full, s, False)[1])
    print(f"seed {s} done; running-mean val RMSLE {np.sqrt(np.mean((np.mean(vs, 0) - va['y'].values) ** 2)):.5f}", flush=True)
val, test = np.mean(vs, 0), np.mean(ts, 0)
print(f"LSTM ({len(SEEDS)}-seed mean): val RMSLE {np.sqrt(np.mean((val - va['y'].values) ** 2)):.5f}")
np.savez(Path(__file__).parent / "preds_lstm.npz", val=val, test=test)
print("Saved preds_lstm.npz")
