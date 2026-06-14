"""
Módulo 6 — Pruebas de robustez (sec. 3.9)

Cuatro pruebas:
  1. Sensibilidad al horizonte del target (10 y 42 días hábiles)
  2. Sensibilidad a k (k=2 y k=4)
  3. Sensibilidad al coste de transacción (2 pb y 20 pb)
  4. Subperiod analysis (desglose anual 2020-2026)

Entrada: data/*.parquet, data/best_config.json
Salida:  data/robustness.json
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 26122003
SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FEATURE_COLS = [
    "dist_ma_20", "dist_ma_60", "dist_ma_200", "slope_60", "slope_120",
    "ret_21", "ret_21_vs", "ret_63", "ret_63_vs", "ret_126", "ret_126_vs",
    "ret_252", "ret_252_vs", "vol_21", "vol_63", "vol_ratio",
    "rs_21", "rs_63", "rs_126", "rs_252", "zscore_mom_252",
]
OUT_DIR = "data"


def sharpe_annual(returns):
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def cagr(returns):
    equity = (1 + returns).cumprod()
    n_years = len(returns) / 252
    if n_years == 0 or equity.iloc[-1] <= 0:
        return 0.0
    return (equity.iloc[-1]) ** (1 / n_years) - 1


def max_drawdown(returns):
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return dd.min()


def portfolio_daily_returns(weights_df, daily_ret_df, cost_bps):
    cost = cost_bps / 10_000
    rebal_dates = sorted(weights_df.index)
    all_dates = sorted(daily_ret_df.index)
    start = rebal_dates[0]
    wf_dates_daily = [d for d in all_dates if d >= start]
    port_ret = []
    prev_weights = pd.Series(0.0, index=SECTOR_TICKERS)
    for date in wf_dates_daily:
        if date in rebal_dates:
            new_weights = weights_df.loc[date]
            turnover = (new_weights - prev_weights).abs().sum()
            tx_cost = turnover * cost
            prev_weights = new_weights.copy()
        else:
            tx_cost = 0.0
        if date in daily_ret_df.index:
            day_ret = (prev_weights * daily_ret_df.loc[date]).sum()
            port_ret.append({"date": date, "return": day_ret - tx_cost})
    return pd.DataFrame(port_ret).set_index("date")["return"]


def build_weights_from_predictions(pred_df, score_col, k):
    weights = {}
    for date, group in pred_df.groupby("date"):
        ranked = group.sort_values(score_col, ascending=False)
        top_k = ranked.head(k)["ticker"].tolist()
        w = {t: 1.0 / k if t in top_k else 0.0 for t in SECTOR_TICKERS}
        weights[date] = w
    return pd.DataFrame(weights).T


def compute_summary(returns):
    return {
        "CAGR": round(cagr(returns), 4),
        "Sharpe": round(sharpe_annual(returns), 4),
        "MaxDD": round(max_drawdown(returns), 4),
    }


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MODULO 6 — Pruebas de robustez")
    print("=" * 60)

    prices = pd.read_parquet(os.path.join(OUT_DIR, "prices.parquet"))
    pred_df = pd.read_parquet(os.path.join(OUT_DIR, "predictions.parquet"))
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    backtest = pd.read_parquet(os.path.join(OUT_DIR, "backtest_results.parquet"))

    with open(os.path.join(OUT_DIR, "best_config.json"), "r") as f:
        config = json.load(f)
    lgbm_params = config["lgbm"]["params"]

    daily_ret = prices[SECTOR_TICKERS].pct_change().dropna()
    spy_daily_ret = prices["SPY"].pct_change().dropna()

    robustness = {}

    # ───── PRUEBA 1: Sensibilidad a k (k=2 y k=4) ─────
    print()
    print("--- Prueba 1: Sensibilidad a k ---")
    k_results = {}
    for k in [2, 3, 4]:
        if k == 3:
            # Fila principal: usar la serie guardada por m04 para que la Tabla 8
            # coincida exactamente con la Tabla 6 (misma serie, misma definición).
            ret = backtest["LightGBM"]
        else:
            w = build_weights_from_predictions(pred_df, "pred_lgbm", k)
            ret = portfolio_daily_returns(w, daily_ret, cost_bps=10)
        summary = compute_summary(ret)
        k_results[f"k={k}"] = summary
        label = " (principal)" if k == 3 else ""
        print(f"  k={k}{label}: CAGR={summary['CAGR']:.2%}, "
              f"Sharpe={summary['Sharpe']:.3f}, MaxDD={summary['MaxDD']:.2%}")

    robustness["sensitivity_k"] = k_results

    # ───── PRUEBA 2: Sensibilidad al coste ─────
    print()
    print("--- Prueba 2: Sensibilidad al coste de transaccion ---")
    cost_results = {}
    w3 = build_weights_from_predictions(pred_df, "pred_lgbm", k=3)
    for c in [2, 10, 20]:
        if c == 10:
            ret = backtest["LightGBM"]   # fila principal: serie guardada por m04
        else:
            ret = portfolio_daily_returns(w3, daily_ret, cost_bps=c)
        summary = compute_summary(ret)
        cost_results[f"{c}bps"] = summary
        label = " (principal)" if c == 10 else ""
        print(f"  {c:2d} bps{label}: CAGR={summary['CAGR']:.2%}, "
              f"Sharpe={summary['Sharpe']:.3f}, MaxDD={summary['MaxDD']:.2%}")

    robustness["sensitivity_cost"] = cost_results

    # ───── PRUEBA 3: Subperiod analysis ─────
    print()
    print("--- Prueba 3: Subperiod analysis (anual) ---")
    subperiod = {}
    for year in range(2020, 2027):  # incluye 2025 y 2026
        mask = backtest.index.year == year
        if mask.sum() == 0:
            continue
        row = {}
        for strat in ["SPY_BH", "LightGBM", "TSMom12m"]:
            yr_ret = backtest.loc[mask, strat]
            row[strat] = {
                "return": round(yr_ret.sum(), 4),
                "sharpe": round(sharpe_annual(yr_ret), 3),
            }
        subperiod[str(year)] = row
        print(f"  {year}: SPY={row['SPY_BH']['return']:+.2%} (SR {row['SPY_BH']['sharpe']:.2f}), "
              f"LGBM={row['LightGBM']['return']:+.2%} (SR {row['LightGBM']['sharpe']:.2f}), "
              f"TSMom={row['TSMom12m']['return']:+.2%} (SR {row['TSMom12m']['sharpe']:.2f})")

    robustness["subperiod"] = subperiod

    # ───── PRUEBA 4: Sensibilidad al horizonte del target ─────
    print()
    print("--- Prueba 4: Sensibilidad al horizonte del target ---")
    print("  (requiere re-ejecutar walk-forward con targets de 10 y 42 dias)")

    dataset_base = pd.read_parquet(os.path.join(OUT_DIR, "dataset.parquet"))
    dataset_base.index = pd.to_datetime(dataset_base.index)

    horizon_results = {}

    for hz in [10, 21, 42]:
        if hz == 21:
            # Ya tenemos estos resultados: serie principal guardada por m04
            ret_hz = backtest["LightGBM"]
            summary = compute_summary(ret_hz)
            horizon_results[f"{hz}d"] = summary
            print(f"  {hz:2d}d (principal): CAGR={summary['CAGR']:.2%}, "
                  f"Sharpe={summary['Sharpe']:.3f}, MaxDD={summary['MaxDD']:.2%}")
            continue

        # Recalcular target con nuevo horizonte
        fwd_ret_wide = pd.DataFrame(index=prices.index)
        for ticker in SECTOR_TICKERS:
            fwd_ret_wide[ticker] = (prices[ticker].shift(-hz) / prices[ticker]) - 1
        fwd_rank_wide = fwd_ret_wide.rank(axis=1, method="average")

        # Construir dataset alternativo
        alt = dataset_base.copy()
        alt = alt.drop(columns=["target_rank", "fwd_ret_21"], errors="ignore")

        fwd_rank_long = fwd_rank_wide.stack().reset_index()
        fwd_rank_long.columns = ["date", "ticker", "target_rank"]
        fwd_rank_long["date"] = pd.to_datetime(fwd_rank_long["date"])

        alt = alt.reset_index()
        alt["date"] = pd.to_datetime(alt["date"])
        alt = alt.merge(fwd_rank_long, on=["date", "ticker"], how="left")
        alt = alt.set_index("date")
        alt = alt.dropna(subset=["target_rank"])
        alt = alt.dropna(subset=FEATURE_COLS)

        # Walk-forward con este target
        wf_dates = sorted(alt[(alt.index >= "2020-01-01") &
                               (alt.index <= "2026-12-31")].index.unique())
        preds_hz = []

        for dd in wf_dates:
            year = dd.year
            train_end = f"{year - 1}-12-31"
            train = alt[alt.index <= train_end]
            test = alt[alt.index == dd]
            if len(test) == 0 or len(train) < 100:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[FEATURE_COLS].values)
            X_test = scaler.transform(test[FEATURE_COLS].values)
            y_train = train["target_rank"].values

            train_groups = [9] * train.index.nunique()
            lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)

            lgb_p = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "eval_at": [3],
                "verbosity": -1,
                "seed": SEED,
                **{k_: (int(v) if k_ in ["num_leaves", "max_depth", "min_data_in_leaf"]
                        else float(v))
                   for k_, v in lgbm_params.items()},
            }
            if lgb_p.get("bagging_fraction", 1.0) < 1.0:
                lgb_p["bagging_freq"] = 1

            model = lgb.train(lgb_p, lgb_train, num_boost_round=500)
            pred = model.predict(X_test)

            for i, ticker in enumerate(test["ticker"].values):
                preds_hz.append({"date": dd, "ticker": ticker, "pred": pred[i]})

        if len(preds_hz) == 0:
            horizon_results[f"{hz}d"] = {"CAGR": 0, "Sharpe": 0, "MaxDD": 0}
            print(f"  {hz:2d}d: sin predicciones validas")
            continue

        pred_hz_df = pd.DataFrame(preds_hz)
        pred_hz_df["date"] = pd.to_datetime(pred_hz_df["date"])
        w_hz = build_weights_from_predictions(pred_hz_df, "pred", k=3)
        ret_hz = portfolio_daily_returns(w_hz, daily_ret, cost_bps=10)
        summary = compute_summary(ret_hz)
        horizon_results[f"{hz}d"] = summary
        print(f"  {hz:2d}d: CAGR={summary['CAGR']:.2%}, "
              f"Sharpe={summary['Sharpe']:.3f}, MaxDD={summary['MaxDD']:.2%}")

    robustness["sensitivity_horizon"] = horizon_results

    # ───── Resumen ─────
    print()
    print("=" * 60)
    print("RESUMEN DE ROBUSTEZ")
    print("=" * 60)

    # Criterio: Sharpe > 0 en al menos 3 de 4 pruebas
    checks = []
    # k: Sharpe > 0 para todos los k
    k_ok = all(v["Sharpe"] > 0 for v in k_results.values())
    checks.append(("Sensibilidad k", k_ok))
    # Costes: Sharpe > 0 para todos los costes
    cost_ok = all(v["Sharpe"] > 0 for v in cost_results.values())
    checks.append(("Sensibilidad coste", cost_ok))
    # Horizonte: Sharpe > 0 para al menos 2 de 3
    hz_ok = sum(1 for v in horizon_results.values() if v["Sharpe"] > 0) >= 2
    checks.append(("Sensibilidad horizonte", hz_ok))
    # Subperiod: Sharpe > 0 en la mayoría de los años evaluados
    n_years = len(subperiod)
    n_pos_years = sum(1 for v in subperiod.values()
                      if v["LightGBM"]["sharpe"] > 0)
    sp_ok = n_pos_years >= (n_years // 2 + 1)
    checks.append(("Subperiod analysis", sp_ok))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {name:25s}: {'PASA' if ok else 'NO PASA'}")
    print(f"\n  Resultado: {passed}/4 pruebas superadas", end="")
    if passed >= 3:
        print(" -> estrategia considerada robusta")
    else:
        print(" -> estrategia NO considerada robusta")

    robustness["summary"] = {
        "tests_passed": passed,
        "total_tests": 4,
        "robust": passed >= 3,
        "details": {name: ok for name, ok in checks},
    }

    out_path = os.path.join(OUT_DIR, "robustness.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(robustness, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {out_path}")
    print()
    print("Modulo 6 completado. Siguiente: m07_figures.py")
