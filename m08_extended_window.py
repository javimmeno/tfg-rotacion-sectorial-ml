"""
Módulo 8 — Validación sobre ventana extendida (2012-2026)
==========================================================
Prueba de robustez complementaria al resultado principal (2020-2026).
Para que toda la ventana de evaluación sea estrictamente fuera de muestra,
se RESELECCIONAN los hiperparámetros sobre 2000-2011 mediante purged k-fold
CV con embargo, y a continuación se ejecuta un walk-forward con
reentrenamiento anual sobre 2012-2026 (14 años, múltiples regímenes de
mercado). El objetivo es comprobar si la conclusión del trabajo (la señal no
supera de forma significativa a referencias simples) se mantiene en una
muestra mucho más larga y diversa que la ventana principal.

Entrada:  data/dataset.parquet, data/prices.parquet
Salida:   data/extended_metrics.csv      — métricas de las 5 estrategias
          data/extended_inference.json   — Sharpe CI, Ledoit-Wolf, DSR
          figures/fig_extended_equity.png

Reutiliza exactamente los mismos procedimientos que m03, m04 y m05.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Parámetros (mismos que el protocolo principal)
# ──────────────────────────────────────────────
SEED = 26122003
N_FOLDS = 5
EMBARGO_STEPS = 1
N_LGBM_RANDOM = 40
TUNE_START = "2000-01-01"
TUNE_END = "2011-12-31"
WF_START = "2012-01-01"
WF_END = "2026-12-31"
K = 3
COST_BPS = 10

SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FEATURE_COLS = [
    "dist_ma_20", "dist_ma_60", "dist_ma_200", "slope_60", "slope_120",
    "ret_21", "ret_21_vs", "ret_63", "ret_63_vs", "ret_126", "ret_126_vs",
    "ret_252", "ret_252_vs", "vol_21", "vol_63", "vol_ratio",
    "rs_21", "rs_63", "rs_126", "rs_252", "zscore_mom_252",
]
LGBM_SPACE = {
    "learning_rate":    [0.01, 0.05, 0.10],
    "num_leaves":       [15, 31, 63],
    "max_depth":        [-1, 4, 6],
    "min_data_in_leaf": [50, 100, 200],
    "feature_fraction": [0.70, 0.85, 1.00],
    "bagging_fraction": [0.70, 0.85, 1.00],
}
LOGREG_C = [0.01, 0.1, 1.0, 10.0, 100.0]
B = 5000
BLOCK_MEAN = 21
ALPHA = 0.05
OUT_DIR = "data"
FIG_DIR = "figures"

rng = np.random.RandomState(SEED)

print("=" * 60)
print("MÓDULO 8 — Validación sobre ventana extendida 2012-2026")
print("=" * 60)

dataset = pd.read_parquet(os.path.join(OUT_DIR, "dataset.parquet"))
dataset.index = pd.to_datetime(dataset.index)
prices = pd.read_parquet(os.path.join(OUT_DIR, "prices.parquet"))


# ──────────────────────────────────────────────
# 1. Tuning sobre 2000-2011 (purged k-fold + embargo)
# ──────────────────────────────────────────────
tune_data = dataset[(dataset.index >= TUNE_START) & (dataset.index <= TUNE_END)].copy()
dates_tune = sorted(tune_data.index.unique())
n_dates_tune = len(dates_tune)
print(f"Tuning: {dates_tune[0].date()} -> {dates_tune[-1].date()} ({n_dates_tune} fechas)")

fold_size = n_dates_tune // N_FOLDS
fold_assignments = {}
for i, d in enumerate(dates_tune):
    fold_assignments[d] = min(i // fold_size, N_FOLDS - 1)
tune_data["fold"] = tune_data.index.map(fold_assignments)


def get_purged_train_test(data, dates_list, test_fold, embargo=EMBARGO_STEPS):
    test_mask = data["fold"] == test_fold
    test_dates = sorted(data[test_mask].index.unique())
    test_start, test_end = test_dates[0], test_dates[-1]
    train_candidates = data[~test_mask].copy()
    test_start_idx = dates_list.index(test_start)
    purge_start_date = dates_list[max(0, test_start_idx - embargo)]
    test_end_idx = dates_list.index(test_end)
    embargo_end_date = dates_list[min(len(dates_list) - 1, test_end_idx + embargo)]
    train_clean = train_candidates[
        (train_candidates.index < purge_start_date) |
        (train_candidates.index > embargo_end_date)
    ]
    return train_clean, data[test_mask]


def rank_ic_by_date(y_true, y_pred, dates):
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "date": dates})
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 3:
            continue
        corr, _ = spearmanr(g["true"], g["pred"])
        if not np.isnan(corr):
            ics.append(corr)
    return np.mean(ics) if ics else 0.0


def evaluate_lgbm(params, data, dates_list):
    fold_ics = []
    for fold in range(N_FOLDS):
        train, test = get_purged_train_test(data, dates_list, fold)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values)
        X_test = scaler.transform(test[FEATURE_COLS].values)
        y_train = train["target_rank"].values
        train_groups = [9] * len(sorted(train.index.unique()))
        test_groups = [9] * len(sorted(test.index.unique()))
        lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
        lgb_val = lgb.Dataset(X_test, label=test["target_rank"].values,
                              group=test_groups, reference=lgb_train)
        lgb_params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [3],
                      "verbosity": -1, "seed": SEED, **params}
        if params.get("bagging_fraction", 1.0) < 1.0:
            lgb_params["bagging_freq"] = 1
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        model = lgb.train(lgb_params, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_val], callbacks=callbacks)
        preds = model.predict(X_test)
        fold_ics.append(rank_ic_by_date(test["target_rank"].values, preds, test.index))
    return np.mean(fold_ics), np.std(fold_ics)


def evaluate_logreg(C, data, dates_list):
    fold_ics = []
    for fold in range(N_FOLDS):
        train, test = get_purged_train_test(data, dates_list, fold)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values)
        X_test = scaler.transform(test[FEATURE_COLS].values)
        y_train_bin = (train["target_rank"].values >= 7).astype(int)
        model = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                   max_iter=2000, random_state=SEED)
        model.fit(X_train, y_train_bin)
        preds = model.predict_proba(X_test)[:, 1]
        fold_ics.append(rank_ic_by_date(test["target_rank"].values, preds, test.index))
    return np.mean(fold_ics), np.std(fold_ics)


print("Random search LightGBM (N=40) sobre 2000-2011...")
all_keys = list(LGBM_SPACE.keys())
lgbm_results = []
for i in range(N_LGBM_RANDOM):
    params = {k: rng.choice(LGBM_SPACE[k]) for k in all_keys}
    params = {k: (int(v) if isinstance(v, np.integer) else
                  float(v) if isinstance(v, np.floating) else v)
              for k, v in params.items()}
    mean_ic, std_ic = evaluate_lgbm(params, tune_data, dates_tune)
    lgbm_results.append({"mean_ic": mean_ic, "std_ic": std_ic, **params})

logreg_results = []
for C in LOGREG_C:
    mean_ic, std_ic = evaluate_logreg(C, tune_data, dates_tune)
    logreg_results.append({"C": C, "mean_ic": mean_ic, "std_ic": std_ic})

lgbm_df = pd.DataFrame(lgbm_results).sort_values(
    ["mean_ic", "std_ic"], ascending=[False, True])
best_lgbm = lgbm_df.iloc[0]
# Desempate por menor std si las dos mejores difieren < 5 %
if len(lgbm_df) > 1:
    top2 = lgbm_df.head(2)
    if top2.iloc[0]["mean_ic"] != 0:
        diff_pct = abs(top2.iloc[0]["mean_ic"] - top2.iloc[1]["mean_ic"]) / \
                   abs(top2.iloc[0]["mean_ic"]) * 100
        if diff_pct < 5:
            best_lgbm = top2.sort_values("std_ic").iloc[0]
best_logreg = pd.DataFrame(logreg_results).sort_values(
    "mean_ic", ascending=False).iloc[0]

lgbm_params = {k: (int(best_lgbm[k]) if k in ["num_leaves", "max_depth", "min_data_in_leaf"]
                   else float(best_lgbm[k]))
               for k in LGBM_SPACE.keys()}
logreg_C = float(best_logreg["C"])
n_trials = N_LGBM_RANDOM + len(LOGREG_C)

print(f"  Mejor LightGBM: rank IC {best_lgbm['mean_ic']:.4f}  params={lgbm_params}")
print(f"  Mejor LogReg:   rank IC {best_logreg['mean_ic']:.4f}  C={logreg_C}")
print()


# ──────────────────────────────────────────────
# 2. Walk-forward 2012-2026 (reentrenamiento anual)
# ──────────────────────────────────────────────
print("Walk-forward 2012-2026 (reentrenamiento anual)...")
wf_dates = sorted(dataset[(dataset.index >= WF_START) &
                          (dataset.index <= WF_END)].index.unique())
print(f"  Fechas de decisión: {len(wf_dates)} "
      f"({wf_dates[0].date()} -> {wf_dates[-1].date()})")

predictions = []
for decision_date in wf_dates:
    train_end = f"{decision_date.year - 1}-12-31"
    train = dataset[dataset.index <= train_end].copy()
    test = dataset[dataset.index == decision_date].copy()
    if len(test) == 0:
        continue
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].values)
    X_test = scaler.transform(test[FEATURE_COLS].values)
    y_train = train["target_rank"].values

    train_groups = [9] * len(sorted(train.index.unique()))
    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
    lgb_p = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [3],
             "verbosity": -1, "seed": SEED, **lgbm_params}
    if lgb_p.get("bagging_fraction", 1.0) < 1.0:
        lgb_p["bagging_freq"] = 1
    model_lgbm = lgb.train(lgb_p, lgb_train, num_boost_round=500)
    pred_lgbm = model_lgbm.predict(X_test)

    y_train_bin = (y_train >= 7).astype(int)
    model_lr = LogisticRegression(C=logreg_C, penalty="l2", solver="lbfgs",
                                  max_iter=2000, random_state=SEED)
    model_lr.fit(X_train, y_train_bin)
    pred_lr = model_lr.predict_proba(X_test)[:, 1]

    for i, ticker in enumerate(test["ticker"].values):
        predictions.append({"date": decision_date, "ticker": ticker,
                             "pred_lgbm": pred_lgbm[i], "pred_logreg": pred_lr[i]})

pred_df = pd.DataFrame(predictions)
pred_df["date"] = pd.to_datetime(pred_df["date"])

daily_ret = prices[SECTOR_TICKERS].pct_change().dropna()
spy_daily_ret = prices["SPY"].pct_change().dropna()


def build_monthly_weights(pred_df, score_col, k=K):
    weights = {}
    for date, group in pred_df.groupby("date"):
        top_k = group.sort_values(score_col, ascending=False).head(k)["ticker"].tolist()
        weights[date] = {t: 1.0 / k if t in top_k else 0.0 for t in SECTOR_TICKERS}
    return pd.DataFrame(weights).T


def build_tsmom_weights(prices_df, decision_dates, lookback=252):
    weights = {}
    for date in decision_dates:
        pos = []
        loc = prices_df.index.get_loc(date)
        for ticker in SECTOR_TICKERS:
            if loc >= lookback:
                r = prices_df[ticker].iloc[loc] / prices_df[ticker].iloc[loc - lookback] - 1
                if r > 0:
                    pos.append(ticker)
        if pos:
            wt = 1.0 / len(pos)
            weights[date] = {t: wt if t in pos else 0.0 for t in SECTOR_TICKERS}
        else:
            weights[date] = {t: 0.0 for t in SECTOR_TICKERS}
    return pd.DataFrame(weights).T


def portfolio_daily_returns(weights_df, daily_ret_df, cost_bps=COST_BPS):
    cost = cost_bps / 10_000
    rebal_dates = sorted(weights_df.index)
    start = rebal_dates[0]
    port_ret = []
    prev_w = pd.Series(0.0, index=SECTOR_TICKERS)
    for date in [d for d in sorted(daily_ret_df.index) if d >= start]:
        if date in rebal_dates:
            new_w = weights_df.loc[date]
            tx = (new_w - prev_w).abs().sum() * cost
            prev_w = new_w.copy()
        else:
            tx = 0.0
        if date in daily_ret_df.index:
            port_ret.append({"date": date,
                             "return": (prev_w * daily_ret_df.loc[date]).sum() - tx})
    return pd.DataFrame(port_ret).set_index("date")["return"]


w_lgbm = build_monthly_weights(pred_df, "pred_lgbm")
w_logreg = build_monthly_weights(pred_df, "pred_logreg")
w_eq = pd.DataFrame({t: 1.0 / 9 for t in SECTOR_TICKERS}, index=w_lgbm.index)
w_tsmom = build_tsmom_weights(prices, wf_dates)

ret_lgbm = portfolio_daily_returns(w_lgbm, daily_ret)
ret_logreg = portfolio_daily_returns(w_logreg, daily_ret)
ret_eq = portfolio_daily_returns(w_eq, daily_ret)
ret_tsmom = portfolio_daily_returns(w_tsmom, daily_ret)
ret_spy = spy_daily_ret[spy_daily_ret.index >= wf_dates[0]]

common_idx = ret_lgbm.index.intersection(ret_spy.index)
results = pd.DataFrame({
    "SPY_BH": ret_spy.reindex(common_idx),
    "EqWeight": ret_eq.reindex(common_idx),
    "TSMom12m": ret_tsmom.reindex(common_idx),
    "LogReg": ret_logreg.reindex(common_idx),
    "LightGBM": ret_lgbm.reindex(common_idx),
}).dropna()
print(f"  Retornos diarios: {len(results)} días")
print()


# ──────────────────────────────────────────────
# 3. Métricas (mismas definiciones que m04/m06)
# ──────────────────────────────────────────────
def compute_metrics(returns, spy_returns, name, weights_df=None):
    n_years = len(returns) / 252
    equity = (1 + returns).cumprod()
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
    running_max = equity.cummax()
    max_dd = ((equity - running_max) / running_max).min()
    monthly = returns.resample("ME").sum()
    monthly_spy = spy_returns.reindex(returns.index).resample("ME").sum()
    cm = monthly.index.intersection(monthly_spy.index)
    hit = (monthly[cm] > monthly_spy[cm]).mean() if len(cm) else 0.0
    if weights_df is not None and len(weights_df) > 1:
        turn = (weights_df.diff().abs().sum(axis=1).dropna().mean() / 2) * 12
    else:
        turn = 0.0
    return {"Estrategia": name, "CAGR": f"{cagr:.2%}", "Vol": f"{vol:.2%}",
            "Sharpe": f"{sharpe:.2f}", "MaxDD": f"{max_dd:.2%}",
            "HitRate": f"{hit:.1%}", "Turnover": f"{turn:.1%}"}


metrics = [
    compute_metrics(results["SPY_BH"], results["SPY_BH"], "SPY B&H"),
    compute_metrics(results["EqWeight"], results["SPY_BH"], "Equiponderado", w_eq),
    compute_metrics(results["TSMom12m"], results["SPY_BH"], "TS-Mom 12m", w_tsmom),
    compute_metrics(results["LogReg"], results["SPY_BH"], "LogReg L2", w_logreg),
    compute_metrics(results["LightGBM"], results["SPY_BH"], "LightGBM", w_lgbm),
]
metrics_df = pd.DataFrame(metrics)
print("MÉTRICAS — ventana extendida 2012-2026 (10 pb)")
print(metrics_df.to_string(index=False))
print()
metrics_df.to_csv(os.path.join(OUT_DIR, "extended_metrics.csv"),
                  index=False, encoding="utf-8")


# ──────────────────────────────────────────────
# 4. Inferencia (Sharpe CI, Ledoit-Wolf HAC, DSR)
# ──────────────────────────────────────────────
def sharpe_ratio(r):
    s = r.std()
    return (r.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def stationary_bootstrap(series, n_boot, block_mean, rng):
    n = len(series)
    p = 1.0 / block_mean
    reps = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.randint(n)
        for t in range(1, n):
            idx[t] = rng.randint(n) if rng.random() < p else (idx[t - 1] + 1) % n
        reps[b] = series[idx]
    return reps


def bootstrap_sharpe_ci(returns):
    arr = returns.values if hasattr(returns, "values") else returns
    reps = stationary_bootstrap(arr, B, BLOCK_MEAN, rng)
    sh = np.array([sharpe_ratio(pd.Series(r)) for r in reps])
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


def _hac_lrv(Y, bandwidth=None):
    T, _ = Y.shape
    if bandwidth is None:
        bandwidth = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(bandwidth, 0)
    Omega = (Y.T @ Y) / T
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1.0)
        Gj = (Y[j:].T @ Y[:-j]) / T
        Omega += w * (Gj + Gj.T)
    return Omega


def ledoit_wolf_test(ret_a, ret_b):
    a = ret_a.values.astype(float) if hasattr(ret_a, "values") else np.asarray(ret_a, float)
    b = ret_b.values.astype(float) if hasattr(ret_b, "values") else np.asarray(ret_b, float)
    T = len(a)
    mu_a, mu_b = a.mean(), b.mean()
    m_a, m_b = (a ** 2).mean(), (b ** 2).mean()
    sig_a, sig_b = np.sqrt(m_a - mu_a ** 2), np.sqrt(m_b - mu_b ** 2)
    if sig_a <= 0 or sig_b <= 0:
        return 0.0, 1.0, 0.0
    diff = mu_a / sig_a - mu_b / sig_b
    grad = np.array([1.0 / sig_a + mu_a ** 2 / sig_a ** 3,
                     -(1.0 / sig_b + mu_b ** 2 / sig_b ** 3),
                     -mu_a / (2 * sig_a ** 3), mu_b / (2 * sig_b ** 3)])
    Y = np.column_stack([a - mu_a, b - mu_b, a ** 2 - m_a, b ** 2 - m_b])
    var_diff = float(grad @ _hac_lrv(Y) @ grad) / T
    if var_diff <= 0:
        return float(diff), 1.0, 0.0
    t = diff / np.sqrt(var_diff)
    return float(diff), float(2.0 * (1.0 - stats.norm.cdf(abs(t)))), float(t)


def deflated_sharpe_ratio(sharpe_obs, n_obs, n_trials, skew=0.0, kurt=3.0):
    gamma = 0.5772156649
    var_sr = (1 - skew * sharpe_obs + (kurt - 1) / 4 * sharpe_obs ** 2) / (n_obs - 1)
    if var_sr <= 0:
        return 0.0
    std_sr = np.sqrt(var_sr)
    if n_trials <= 1:
        sr_star = 0.0
    else:
        sr_star = std_sr * ((1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
                            + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float(stats.norm.cdf((sharpe_obs - sr_star) / std_sr)) if std_sr > 0 else 0.0


print("Inferencia (ventana extendida)...")
n_obs = len(results)
inference = {"sharpe_ci": {}, "ledoit_wolf_vs_spy": {}, "deflated_sharpe": {},
             "meta": {"n_obs": n_obs, "n_trials": n_trials, "B": B,
                      "tuning": f"{TUNE_START[:4]}-{TUNE_END[:4]}",
                      "oos": f"{wf_dates[0].date()} -> {wf_dates[-1].date()}"}}

for strat in ["SPY_BH", "EqWeight", "TSMom12m", "LogReg", "LightGBM"]:
    sr = sharpe_ratio(results[strat])
    lo, hi = bootstrap_sharpe_ci(results[strat])
    inference["sharpe_ci"][strat] = {"sharpe": round(sr, 4),
                                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    print(f"  {strat:10s}: Sharpe={sr:.3f}  IC95=[{lo:.3f}, {hi:.3f}]")

print()
lw_pvals = []
for strat in ["EqWeight", "TSMom12m", "LogReg", "LightGBM"]:
    diff, pval, tstat = ledoit_wolf_test(results[strat], results["SPY_BH"])
    inference["ledoit_wolf_vs_spy"][strat] = {"diff": round(diff, 4),
                                              "t_stat": round(tstat, 4),
                                              "p_value": round(pval, 4)}
    lw_pvals.append((strat, pval))
    print(f"  Ledoit-Wolf {strat:10s} vs SPY: t={tstat:.3f}, p={pval:.4f}")

# Corrección de Holm-Bonferroni (mismo procedimiento que el módulo principal)
order = sorted(range(len(lw_pvals)), key=lambda i: lw_pvals[i][1])
m = len(lw_pvals)
adj = [0.0] * m
prev = 0.0
for rank, i in enumerate(order):
    val = min(1.0, (m - rank) * lw_pvals[i][1])
    prev = max(prev, val)
    adj[i] = prev
print("  P-values ajustados (Holm-Bonferroni):")
for i, (strat, _) in enumerate(lw_pvals):
    inference["ledoit_wolf_vs_spy"][strat]["p_adj"] = round(adj[i], 4)
    sig = " (significativo al 5%)" if adj[i] < 0.05 else ""
    print(f"    {strat:10s}: p_adj={adj[i]:.4f}{sig}")

print()
for strat in ["LogReg", "LightGBM"]:
    r = results[strat].values
    daily_sr = r.mean() / r.std() if r.std() > 0 else 0.0
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurtosis() + 3)
    dsr = deflated_sharpe_ratio(daily_sr, n_obs, n_trials, sk, ku)
    inference["deflated_sharpe"][strat] = {"sharpe_anual": round(daily_sr * np.sqrt(252), 4),
                                           "dsr": round(dsr, 4)}
    sig = "significativo" if dsr > 0.95 else "no significativo"
    print(f"  DSR {strat:10s}: SR_anual={daily_sr*np.sqrt(252):.3f}, DSR={dsr:.4f} ({sig})")

with open(os.path.join(OUT_DIR, "extended_inference.json"), "w", encoding="utf-8") as f:
    json.dump(inference, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# 5. Figura: curvas de equity 2012-2026
# ──────────────────────────────────────────────
os.makedirs(FIG_DIR, exist_ok=True)
equity = (1 + results).cumprod()
COLORS = {"SPY_BH": "#1f77b4", "EqWeight": "#aec7e8", "TSMom12m": "#ff7f0e",
          "LogReg": "#2ca02c", "LightGBM": "#d62728"}
LABELS = {"SPY_BH": "SPY B&H", "EqWeight": "Equiponderado", "TSMom12m": "TS-Mom 12m",
          "LogReg": "LogReg L2", "LightGBM": "LightGBM"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 5))
for s in ["SPY_BH", "EqWeight", "TSMom12m", "LogReg", "LightGBM"]:
    ax.plot(equity.index, equity[s], label=LABELS[s], color=COLORS[s],
            linewidth=1.5 if s in ["SPY_BH", "LightGBM"] else 1.0,
            alpha=1.0 if s in ["SPY_BH", "LightGBM"] else 0.7)
ax.set_ylabel("Valor de la cartera (base 1.0)")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_extended_equity.png"), bbox_inches="tight")
plt.close()

print()
print("=" * 60)
print("Archivos: data/extended_metrics.csv, data/extended_inference.json")
print("Figura  : figures/fig_extended_equity.png")
print("Módulo 8 completado.")
