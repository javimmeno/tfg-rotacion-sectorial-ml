"""
Módulo 5 — Inferencia estadística (sec. 3.8.1)

Calcula:
  - Intervalos de confianza del Sharpe Ratio por stationary bootstrap
    (Politis y Romano, 1994). B=5000, bloque medio=21 días.
  - Test de Ledoit-Wolf (2008) para igualdad de Sharpe entre pares.
  - Deflated Sharpe Ratio (Bailey y López de Prado, 2014) con N_trials=45.
  - Corrección Holm-Bonferroni sobre comparaciones múltiples frente a SPY.

Entrada: data/backtest_results.parquet, data/best_config.json
Salida:  data/inference.json
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

SEED = 26122003
B = 5000              # réplicas bootstrap
BLOCK_MEAN = 21       # longitud media de bloque (días hábiles)
ALPHA = 0.05
OUT_DIR = "data"

rng = np.random.RandomState(SEED)


# --- Stationary bootstrap (Politis y Romano, 1994) ---

def stationary_bootstrap(series, n_boot, block_mean, rng):
    """Genera réplicas por stationary bootstrap.
    En cada paso, con probabilidad 1/block_mean se salta a una posición
    aleatoria; de lo contrario, avanza una posición."""
    n = len(series)
    p = 1.0 / block_mean
    replicas = np.empty((n_boot, n))

    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.randint(n)
        for t in range(1, n):
            if rng.random() < p:
                idx[t] = rng.randint(n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        replicas[b] = series[idx]

    return replicas


def sharpe_ratio(returns):
    """Sharpe anualizado con rf=0."""
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return mu / sigma if sigma > 0 else 0.0


def bootstrap_sharpe_ci(returns, n_boot=B, block_mean=BLOCK_MEAN, alpha=ALPHA):
    """Intervalo de confianza del Sharpe por stationary bootstrap."""
    arr = returns.values if hasattr(returns, "values") else returns
    replicas = stationary_bootstrap(arr, n_boot, block_mean, rng)
    sharpes = np.array([sharpe_ratio(r) for r in replicas])
    lo = np.percentile(sharpes, 100 * alpha / 2)
    hi = np.percentile(sharpes, 100 * (1 - alpha / 2))
    return float(np.median(sharpes)), float(lo), float(hi)


# --- Test de Ledoit y Wolf (2008): versión estudentizada con varianza HAC ---

def _hac_lrv(Y, bandwidth=None):
    """Matriz de covarianza de largo plazo (HAC, kernel de Bartlett /
    Newey-West) de las filas de Y, ya centradas. Maneja la autocorrelación
    serial de las series financieras."""
    T, _ = Y.shape
    if bandwidth is None:
        # Selección automática de retardos (Newey y West, 1994)
        bandwidth = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(bandwidth, 0)
    Omega = (Y.T @ Y) / T
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1.0)          # peso de Bartlett
        Gj = (Y[j:].T @ Y[:-j]) / T
        Omega += w * (Gj + Gj.T)
    return Omega


def ledoit_wolf_test(ret_a, ret_b, bandwidth=None):
    """Test de Ledoit y Wolf (2008) para H0: Sharpe(a) = Sharpe(b).

    Estadístico estudentizado del diferencial de Sharpe, con error estándar
    estimado por método delta sobre los momentos (media y segundo momento de
    cada serie) y matriz de covarianza HAC (Newey-West), robusto a
    no-normalidad y a dependencia serial. La anualización del Sharpe se
    cancela en el diferencial, por lo que el test se hace sobre el Sharpe
    por observación. Devuelve (diferencial, p_value, estadístico t).
    """
    a = np.asarray(ret_a, dtype=float)
    b = np.asarray(ret_b, dtype=float)
    if hasattr(ret_a, "values"):
        a = ret_a.values.astype(float)
    if hasattr(ret_b, "values"):
        b = ret_b.values.astype(float)
    T = len(a)

    mu_a, mu_b = a.mean(), b.mean()
    m_a, m_b = (a ** 2).mean(), (b ** 2).mean()      # segundos momentos no centrados
    sig_a = np.sqrt(m_a - mu_a ** 2)
    sig_b = np.sqrt(m_b - mu_b ** 2)
    if sig_a <= 0 or sig_b <= 0:
        return 0.0, 1.0, 0.0

    sr_a, sr_b = mu_a / sig_a, mu_b / sig_b
    diff = sr_a - sr_b

    # Gradiente del diferencial respecto a (mu_a, mu_b, m_a, m_b)
    grad = np.array([
        1.0 / sig_a + mu_a ** 2 / sig_a ** 3,
        -(1.0 / sig_b + mu_b ** 2 / sig_b ** 3),
        -mu_a / (2 * sig_a ** 3),
        mu_b / (2 * sig_b ** 3),
    ])

    # Funciones de momento centradas y su covarianza de largo plazo
    Y = np.column_stack([a - mu_a, b - mu_b, a ** 2 - m_a, b ** 2 - m_b])
    Omega = _hac_lrv(Y, bandwidth)
    var_diff = float(grad @ Omega @ grad) / T
    if var_diff <= 0:
        return float(diff), 1.0, 0.0

    se = np.sqrt(var_diff)
    tstat = diff / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(tstat)))
    return float(diff), float(p_value), float(tstat)


# --- Deflated Sharpe Ratio (Bailey y López de Prado, 2014) ---

def deflated_sharpe_ratio(sharpe_obs, n_obs, n_trials, skew=0.0, kurt=3.0):
    """DSR: probabilidad de que el Sharpe observado sea genuino,
    descontando el efecto de haber probado n_trials configuraciones.

    Usa la aproximación de Bailey y López de Prado (2014) eq. 14:
    SR* = sqrt(V[SR]) * ((1-gamma)*Z^{-1}(1-1/N) + gamma*Z^{-1}(1-1/N*e^{-1}))
    donde gamma es la constante de Euler-Mascheroni.
    """
    # Sharpe esperado bajo la hipótesis nula con n_trials intentos
    gamma_em = 0.5772156649  # Euler-Mascheroni
    var_sr = (1 - skew * sharpe_obs + (kurt - 1) / 4 * sharpe_obs ** 2) / (n_obs - 1)
    if var_sr <= 0:
        return 0.0
    std_sr = np.sqrt(var_sr)

    # Máximo esperado bajo H0 (Sharpe=0, n_trials intentos)
    if n_trials <= 1:
        sr_star = 0.0
    else:
        z_val = stats.norm.ppf(1 - 1.0 / n_trials)
        sr_star = std_sr * ((1 - gamma_em) * z_val
                            + gamma_em * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))

    # Estadístico: P(SR < SR_obs | H0)
    if std_sr > 0:
        psr = float(stats.norm.cdf((sharpe_obs - sr_star) / std_sr))
    else:
        psr = 0.0

    return psr


# --- Corrección Holm-Bonferroni ---

def holm_bonferroni(p_values):
    """Devuelve p-values ajustados por Holm-Bonferroni."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(p_values[idx] * (n - rank), 1.0)
    # Garantizar monotonicidad
    for i in range(1, n):
        idx = order[i]
        idx_prev = order[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[idx_prev])
    return adjusted


# --- Ejecución ---

if __name__ == "__main__":
    print("=" * 60)
    print("MODULO 5 — Inferencia estadistica")
    print("=" * 60)

    results = pd.read_parquet(os.path.join(OUT_DIR, "backtest_results.parquet"))
    with open(os.path.join(OUT_DIR, "best_config.json"), "r") as f:
        config = json.load(f)

    n_trials = config["tuning_meta"]["n_trials_total"]
    strategies = ["SPY_BH", "EqWeight", "TSMom12m", "LogReg", "LightGBM"]
    n_obs = len(results)

    print(f"Dias: {n_obs}, N_trials: {n_trials}, B: {B}, Bloque: {BLOCK_MEAN}")
    print()

    # 1. Bootstrap CI del Sharpe
    print("Intervalos de confianza del Sharpe (stationary bootstrap)...")
    ci_results = {}
    for strat in strategies:
        obs_sharpe = sharpe_ratio(results[strat].values)
        median_sr, lo, hi = bootstrap_sharpe_ci(results[strat])
        ci_results[strat] = {
            "sharpe_obs": round(obs_sharpe, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        }
        print(f"  {strat:15s}: Sharpe={obs_sharpe:.3f}  IC95=[{lo:.3f}, {hi:.3f}]")

    # 2. Test de Ledoit-Wolf: cada estrategia vs SPY
    print()
    print("Test de Ledoit-Wolf (H0: Sharpe_i = Sharpe_SPY)...")
    lw_pairs = ["EqWeight", "TSMom12m", "LogReg", "LightGBM"]
    lw_results = {}
    raw_pvals = []

    for strat in lw_pairs:
        diff, pval, tstat = ledoit_wolf_test(results[strat], results["SPY_BH"])
        lw_results[strat] = {"diff": round(diff, 4), "t_stat": round(tstat, 4),
                             "p_value": round(pval, 4)}
        raw_pvals.append(pval)
        print(f"  {strat:15s} vs SPY: diff={diff:.3f}, t={tstat:.3f}, p={pval:.4f}")

    # Corrección Holm-Bonferroni
    adj_pvals = holm_bonferroni(np.array(raw_pvals))
    print()
    print("P-values ajustados (Holm-Bonferroni):")
    for i, strat in enumerate(lw_pairs):
        lw_results[strat]["p_adj"] = round(float(adj_pvals[i]), 4)
        print(f"  {strat:15s}: p_adj={adj_pvals[i]:.4f}")

    # 3. Test auxiliar: LightGBM vs TS-Momentum
    print()
    diff_aux, pval_aux, tstat_aux = ledoit_wolf_test(results["LightGBM"], results["TSMom12m"])
    print(f"LightGBM vs TS-Mom 12m: diff={diff_aux:.3f}, t={tstat_aux:.3f}, p={pval_aux:.4f}")

    # 4. Deflated Sharpe Ratio (solo modelos ML)
    print()
    print("Deflated Sharpe Ratio (N_trials={})...".format(n_trials))
    dsr_results = {}
    for strat in ["LogReg", "LightGBM"]:
        ret = results[strat].values
        # DSR usa el Sharpe por observacion (diario), no anualizado
        daily_sr = ret.mean() / ret.std() if ret.std() > 0 else 0.0
        annual_sr = daily_sr * np.sqrt(252)
        sk = float(pd.Series(ret).skew())
        ku = float(pd.Series(ret).kurtosis() + 3)  # excess -> raw
        dsr = deflated_sharpe_ratio(daily_sr, n_obs, n_trials, sk, ku)
        dsr_results[strat] = {
            "sharpe_obs_annual": round(annual_sr, 4),
            "sharpe_obs_daily": round(daily_sr, 4),
            "dsr": round(dsr, 4),
            "skewness": round(sk, 4),
            "kurtosis": round(ku, 4),
        }
        sig = "significativo" if dsr > 0.95 else "no significativo"
        print(f"  {strat:15s}: SR_diario={daily_sr:.4f}, SR_anual={annual_sr:.3f}, "
              f"DSR={dsr:.4f} ({sig} al 5%)")

    # 5. Guardar
    output = {
        "bootstrap_ci": ci_results,
        "ledoit_wolf_vs_spy": lw_results,
        "ledoit_wolf_lgbm_vs_tsmom": {
            "diff": round(diff_aux, 4), "t_stat": round(tstat_aux, 4),
            "p_value": round(pval_aux, 4)
        },
        "deflated_sharpe": dsr_results,
        "params": {
            "B": B, "block_mean": BLOCK_MEAN, "alpha": ALPHA,
            "n_trials": n_trials, "n_obs": n_obs, "seed": SEED,
        },
    }

    out_path = os.path.join(OUT_DIR, "inference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"Resultados guardados: {out_path}")
    print()
    print("Modulo 5 completado. Siguiente: m06_robustness.py")
