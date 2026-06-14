"""
Módulo 4 — Walk-forward 2020 en adelante y backtest
==============================================
Ejecuta el walk-forward con reentrenamiento anual, genera predicciones
mensuales y evalúa las 5 estrategias con costes de transacción.

Estrategias (sec. 3.6):
  1. SPY buy & hold
  2. Equiponderado mensual (1/9 por sector)
  3. TS-Momentum 12m (Moskowitz et al., 2012)
  4. LogReg L2 (línea base ML)
  5. LightGBM lambdarank (modelo principal)

Protocolo (secs. 3.5, 3.7):
  - Reentrenamiento anual: modelo entrenado con datos hasta 31-dic del año anterior
  - Decisión mensual: último día hábil de cada mes
  - Asignación: top-3 equiponderado (k=3, fijado a priori)
  - Coste: 10 pb por operación (one-way)
  - Walk-forward: 2020-01 hasta el final de los datos (la última decisión la fija m02)

Entrada:  data/dataset.parquet, data/best_config.json, data/prices.parquet
Salida:   data/backtest_results.parquet  — retornos diarios por estrategia
          data/metrics.csv               — tabla de métricas
          data/predictions.parquet       — predicciones mensuales del walk-forward
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. Parámetros
# ──────────────────────────────────────────────
SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
K = 3                # top-k fijado a priori
COST_BPS = 10        # coste por operación en puntos básicos
SEED = 26122003
WF_START = "2020-01-01"
WF_END = "2026-12-31"   # cota superior amplia; la última decisión real la fija m02
                        # (última fila con target a 21 días disponible)

FEATURE_COLS = [
    "dist_ma_20", "dist_ma_60", "dist_ma_200", "slope_60", "slope_120",
    "ret_21", "ret_21_vs", "ret_63", "ret_63_vs", "ret_126", "ret_126_vs",
    "ret_252", "ret_252_vs", "vol_21", "vol_63", "vol_ratio",
    "rs_21", "rs_63", "rs_126", "rs_252", "zscore_mom_252",
]
OUT_DIR = "data"

# ──────────────────────────────────────────────
# 1. Cargar datos y configuración
# ──────────────────────────────────────────────
print("=" * 60)
print("MÓDULO 4 — Walk-forward y backtest")
print("=" * 60)

dataset = pd.read_parquet(os.path.join(OUT_DIR, "dataset.parquet"))
dataset.index = pd.to_datetime(dataset.index)
prices = pd.read_parquet(os.path.join(OUT_DIR, "prices.parquet"))

with open(os.path.join(OUT_DIR, "best_config.json"), "r") as f:
    config = json.load(f)

lgbm_params = config["lgbm"]["params"]
logreg_C = config["logreg"]["C"]

print(f"LightGBM config: {lgbm_params}")
print(f"LogReg C: {logreg_C}")
print()

# ──────────────────────────────────────────────
# 2. Walk-forward con reentrenamiento anual
# ──────────────────────────────────────────────
print("-" * 60)
print("WALK-FORWARD 2020 EN ADELANTE")
print("-" * 60)

wf_dates = sorted(dataset[
    (dataset.index >= WF_START) & (dataset.index <= WF_END)
].index.unique())

print(f"Fechas de decisión: {len(wf_dates)}")
print(f"Primera: {wf_dates[0].date()}, Última: {wf_dates[-1].date()}")
print()

predictions = []

for decision_date in wf_dates:
    year = decision_date.year
    train_end = f"{year - 1}-12-31"

    # Datos de entrenamiento: todo hasta 31-dic del año anterior
    train = dataset[dataset.index <= train_end].copy()
    # Datos de test: la fecha de decisión actual
    test = dataset[dataset.index == decision_date].copy()

    if len(test) == 0:
        continue

    X_train = train[FEATURE_COLS].values
    y_train = train["target_rank"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target_rank"].values

    # Estandarizar
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- LightGBM ---
    train_dates_u = sorted(train.index.unique())
    train_groups = [9] * len(train_dates_u)

    lgb_train = lgb.Dataset(X_train_s, label=y_train, group=train_groups)

    lgb_p = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [3],
        "verbosity": -1,
        "seed": SEED,
        "num_boost_round": 1000,
        **{k_: (int(v) if k_ in ["num_leaves", "max_depth", "min_data_in_leaf"]
                else float(v))
           for k_, v in lgbm_params.items()},
    }
    if lgb_p.get("bagging_fraction", 1.0) < 1.0:
        lgb_p["bagging_freq"] = 1

    # Sin early stopping en walk-forward (no hay val set separado)
    # Usamos un número fijo de rondas razonable
    n_rounds = 500
    model_lgbm = lgb.train(lgb_p, lgb_train, num_boost_round=n_rounds)
    pred_lgbm = model_lgbm.predict(X_test_s)

    # --- LogReg ---
    y_train_bin = (y_train >= 7).astype(int)
    # Misma especificación que en tuning (m03): L2 puro con solver lbfgs.
    model_lr = LogisticRegression(C=logreg_C, penalty="l2", solver="lbfgs",
                                  max_iter=2000, random_state=SEED)
    model_lr.fit(X_train_s, y_train_bin)
    pred_lr = model_lr.predict_proba(X_test_s)[:, 1]

    # Guardar predicciones
    for i, ticker in enumerate(test["ticker"].values):
        predictions.append({
            "date": decision_date,
            "ticker": ticker,
            "pred_lgbm": pred_lgbm[i],
            "pred_logreg": pred_lr[i],
            "target_rank": y_test[i],
            "fwd_ret_21": test["fwd_ret_21"].values[i],
        })

pred_df = pd.DataFrame(predictions)
pred_df["date"] = pd.to_datetime(pred_df["date"])

# Rank IC por fecha
print("Rank IC por año (LightGBM):")
for year in sorted(pred_df["date"].dt.year.unique()):
    year_data = pred_df[pred_df["date"].dt.year == year]
    ics = []
    for d, g in year_data.groupby("date"):
        corr, _ = spearmanr(g["target_rank"], g["pred_lgbm"])
        ics.append(corr)
    print(f"  {year}: IC={np.mean(ics):.4f} +/- {np.std(ics):.4f} ({len(ics)} meses)")

print()

# ──────────────────────────────────────────────
# 3. Construir carteras mensuales
# ──────────────────────────────────────────────
print("-" * 60)
print("CONSTRUCCIÓN DE CARTERAS")
print("-" * 60)

# Retornos diarios de los sectores
daily_ret = prices[SECTOR_TICKERS].pct_change().dropna()
spy_daily_ret = prices["SPY"].pct_change().dropna()

# Para cada estrategia, construimos los pesos mensuales
# y luego calculamos retornos diarios ponderados

def build_monthly_weights(pred_df, score_col, k=K):
    """Dado un DataFrame de predicciones, devuelve pesos mensuales.
    Top-k equiponderado: 1/k en los k mejores, 0 en el resto."""
    weights = {}
    for date, group in pred_df.groupby("date"):
        ranked = group.sort_values(score_col, ascending=False)
        top_k = ranked.head(k)["ticker"].tolist()
        w = {t: 1.0 / k if t in top_k else 0.0 for t in SECTOR_TICKERS}
        weights[date] = w
    return pd.DataFrame(weights).T


def build_tsmom_weights(prices_df, decision_dates, lookback=252):
    """TS-Momentum 12m: invierte equiponderado en sectores con retorno
    12 meses positivo, efectivo en el resto."""
    weights = {}
    for date in decision_dates:
        w = {}
        pos_sectors = []
        for ticker in SECTOR_TICKERS:
            loc = prices_df.index.get_loc(date)
            if loc >= lookback:
                ret_12m = prices_df[ticker].iloc[loc] / prices_df[ticker].iloc[loc - lookback] - 1
                if ret_12m > 0:
                    pos_sectors.append(ticker)
        if len(pos_sectors) > 0:
            wt = 1.0 / len(pos_sectors)
            w = {t: wt if t in pos_sectors else 0.0 for t in SECTOR_TICKERS}
        else:
            w = {t: 0.0 for t in SECTOR_TICKERS}  # todo en efectivo
        weights[date] = w
    return pd.DataFrame(weights).T


# Pesos de cada estrategia
weights_lgbm = build_monthly_weights(pred_df, "pred_lgbm", K)
weights_logreg = build_monthly_weights(pred_df, "pred_logreg", K)
weights_eq = pd.DataFrame(
    {t: 1.0 / len(SECTOR_TICKERS) for t in SECTOR_TICKERS},
    index=weights_lgbm.index,
)
weights_tsmom = build_tsmom_weights(prices, wf_dates)

print(f"Estrategias construidas para {len(wf_dates)} fechas de decisión")
print()

# ──────────────────────────────────────────────
# 4. Calcular retornos diarios de cada cartera
# ──────────────────────────────────────────────
def portfolio_daily_returns(weights_df, daily_ret_df, cost_bps=COST_BPS):
    """Calcula retornos diarios de una cartera con rebalanceo mensual.
    Aplica costes de transacción en cada fecha de rebalanceo."""
    cost = cost_bps / 10_000  # convertir a decimal

    rebal_dates = sorted(weights_df.index)
    all_dates = sorted(daily_ret_df.index)

    # Filtrar al periodo del walk-forward
    start = rebal_dates[0]
    wf_dates_daily = [d for d in all_dates if d >= start]

    port_ret = []
    prev_weights = pd.Series(0.0, index=SECTOR_TICKERS)

    for date in wf_dates_daily:
        # ¿Es fecha de rebalanceo?
        if date in rebal_dates:
            new_weights = weights_df.loc[date]
            # Coste de transacción = sum(|cambio de peso|) * cost
            turnover = (new_weights - prev_weights).abs().sum()
            tx_cost = turnover * cost
            prev_weights = new_weights.copy()
        else:
            tx_cost = 0.0

        if date in daily_ret_df.index:
            day_ret = (prev_weights * daily_ret_df.loc[date]).sum()
            port_ret.append({"date": date, "return": day_ret - tx_cost})

    return pd.DataFrame(port_ret).set_index("date")["return"]


print("Calculando retornos diarios con costes de transacción...")

ret_lgbm = portfolio_daily_returns(weights_lgbm, daily_ret)
ret_logreg = portfolio_daily_returns(weights_logreg, daily_ret)
ret_eq = portfolio_daily_returns(weights_eq, daily_ret)
ret_tsmom = portfolio_daily_returns(weights_tsmom, daily_ret)

# SPY buy & hold (sin costes)
ret_spy = spy_daily_ret[spy_daily_ret.index >= wf_dates[0]]

# Alinear todas las series al mismo índice
common_idx = ret_lgbm.index.intersection(ret_spy.index)
results = pd.DataFrame({
    "SPY_BH": ret_spy.reindex(common_idx),
    "EqWeight": ret_eq.reindex(common_idx),
    "TSMom12m": ret_tsmom.reindex(common_idx),
    "LogReg": ret_logreg.reindex(common_idx),
    "LightGBM": ret_lgbm.reindex(common_idx),
}).dropna()

print(f"Retornos diarios: {len(results)} días, {results.shape[1]} estrategias")
print()

# ──────────────────────────────────────────────
# 5. Métricas de evaluación (sec. 3.8)
# ──────────────────────────────────────────────
print("=" * 60)
print("MÉTRICAS DE EVALUACIÓN (2020 en adelante, 10 pb por operación)")
print("=" * 60)


def compute_metrics(returns: pd.Series, spy_returns: pd.Series, name: str,
                    weights_df=None) -> dict:
    """Calcula todas las métricas de la sec. 3.8."""
    n_days = len(returns)
    n_years = n_days / 252

    # Equity curve
    equity = (1 + returns).cumprod()

    # CAGR (capital inicial = 1.0; misma base que m06 para coherencia entre tablas)
    total_ret = equity.iloc[-1]
    cagr = total_ret ** (1 / n_years) - 1

    # Volatilidad anualizada
    vol = returns.std() * np.sqrt(252)

    # Sharpe anualizado (rf = 0): media aritmética anualizada / vol anualizada.
    # MISMA definición que m05 y m06 para que las Tablas 6, 7 y 8 sean coherentes.
    # (Antes se usaba CAGR/vol, que daba un Sharpe distinto y descuadraba tablas.)
    ann_mean = returns.mean() * 252
    sharpe = ann_mean / vol if vol > 0 else 0.0

    # Maximum drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Duración del drawdown máximo (en días hábiles)
    dd_duration = 0
    if max_dd < 0:
        # Encontrar el periodo del max drawdown
        trough_idx = drawdown.idxmin()
        peak_before = equity[:trough_idx].idxmax()
        # Recuperación: primera fecha después del trough donde equity >= peak
        recovery_mask = equity[trough_idx:] >= equity[peak_before]
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            dd_duration = len(equity[peak_before:recovery_date])
        else:
            dd_duration = len(equity[trough_idx:])  # no se recuperó

    # Hit rate mensual (% de meses batiendo a SPY)
    monthly_ret = returns.resample("ME").sum()
    monthly_spy = spy_returns.reindex(returns.index).resample("ME").sum()
    common_months = monthly_ret.index.intersection(monthly_spy.index)
    if len(common_months) > 0:
        hit_rate = (monthly_ret[common_months] > monthly_spy[common_months]).mean()
    else:
        hit_rate = 0.0

    # Turnover anual medio
    if weights_df is not None and len(weights_df) > 1:
        changes = weights_df.diff().abs().sum(axis=1).dropna()
        turnover_per_rebal = changes.mean() / 2  # one-way
        rebal_per_year = 12  # mensual
        annual_turnover = turnover_per_rebal * rebal_per_year
    else:
        annual_turnover = 0.0

    return {
        "Estrategia": name,
        "CAGR": f"{cagr:.2%}",
        "Vol": f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDD": f"{max_dd:.2%}",
        "DD_dias": int(dd_duration),
        "HitRate": f"{hit_rate:.1%}",
        "Turnover": f"{annual_turnover:.1%}",
        "CAGR_raw": cagr,
        "Sharpe_raw": sharpe,
    }


metrics = []
metrics.append(compute_metrics(results["SPY_BH"], results["SPY_BH"], "SPY B&H"))
metrics.append(compute_metrics(results["EqWeight"], results["SPY_BH"], "Equiponderado",
                                weights_eq))
metrics.append(compute_metrics(results["TSMom12m"], results["SPY_BH"], "TS-Mom 12m",
                                weights_tsmom))
metrics.append(compute_metrics(results["LogReg"], results["SPY_BH"], "LogReg L2",
                                weights_logreg))
metrics.append(compute_metrics(results["LightGBM"], results["SPY_BH"], "LightGBM",
                                weights_lgbm))

metrics_df = pd.DataFrame(metrics)
display_cols = ["Estrategia", "CAGR", "Vol", "Sharpe", "MaxDD", "DD_dias",
                "HitRate", "Turnover"]

print()
print(metrics_df[display_cols].to_string(index=False))
print()

# ──────────────────────────────────────────────
# 6. Guardar
# ──────────────────────────────────────────────
results.to_parquet(os.path.join(OUT_DIR, "backtest_results.parquet"))
pred_df.to_parquet(os.path.join(OUT_DIR, "predictions.parquet"), index=False)
metrics_df[display_cols].to_csv(os.path.join(OUT_DIR, "metrics.csv"),
                                 index=False, encoding="utf-8")

# Guardar equity curves para gráficos
equity_curves = (1 + results).cumprod()
equity_curves.to_parquet(os.path.join(OUT_DIR, "equity_curves.parquet"))

print(f"Archivos guardados en {OUT_DIR}/:")
print(f"  backtest_results.parquet  — retornos diarios")
print(f"  predictions.parquet       — predicciones mensuales")
print(f"  metrics.csv               — tabla de métricas")
print(f"  equity_curves.parquet     — curvas de equity")
print()
print("Módulo 4 completado. Siguiente: m05_inference.py")
