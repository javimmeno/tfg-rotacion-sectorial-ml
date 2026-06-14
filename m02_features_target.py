"""
Módulo 2 — Feature engineering y variable objetivo
====================================================
Construye las variables explicativas (sec. 3.4) y el target de ranking
transversal a 21 días (sec. 3.3) sobre el panel de precios.

Todas las features usan exclusivamente ventanas hacia atrás.
Solo las filas de fin de mes se conservan como observaciones del modelo.

Entrada:  data/prices.parquet
Salida:   data/dataset.parquet   — panel (sector, fecha) con features + target
          Informe de dimensiones y estadísticas en consola
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 0. Parámetros
# ──────────────────────────────────────────────
SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
BENCHMARK = "SPY"
TARGET_HORIZON = 21  # días hábiles (~1 mes)
OUT_DIR = "data"

# ──────────────────────────────────────────────
# 1. Cargar precios
# ──────────────────────────────────────────────
print("=" * 60)
print("MÓDULO 2 — Features y target")
print("=" * 60)

prices = pd.read_parquet(os.path.join(OUT_DIR, "prices.parquet"))
print(f"Precios cargados: {prices.shape[0]} días × {prices.shape[1]} tickers\n")

# Retornos diarios (log-returns para aditabilidad en ventanas)
log_ret = np.log(prices / prices.shift(1))
# Retornos simples (para el target y métricas económicas)
simple_ret = prices.pct_change()


# ──────────────────────────────────────────────
# 2. Funciones auxiliares
# ──────────────────────────────────────────────
def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Pendiente normalizada de regresión lineal sobre ventana móvil.
    Se normaliza dividiendo por la media de la ventana para que sea
    comparable entre activos con distintos niveles de precio."""
    def _slope(arr):
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]
        mean = arr.mean()
        return slope / mean if mean != 0 else np.nan
    return series.rolling(window).apply(_slope, raw=True)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Volatilidad Parkinson (range-based) sobre ventana móvil."""
    log_hl = np.log(high / low)
    return np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2)))


# ──────────────────────────────────────────────
# 3. Construir features por sector (diarias)
# ──────────────────────────────────────────────
print("Construyendo features...")

all_frames = []

for ticker in SECTOR_TICKERS:
    px = prices[ticker]
    lr = log_ret[ticker]
    sr = simple_ret[ticker]

    feat = pd.DataFrame(index=prices.index)
    feat["ticker"] = ticker

    # --- TENDENCIA (5 features) ---
    # Distancia del precio a medias móviles
    for w in [20, 60, 200]:
        ma = px.rolling(w).mean()
        feat[f"dist_ma_{w}"] = (px - ma) / ma

    # Pendiente normalizada de regresión lineal
    for w in [60, 120]:
        feat[f"slope_{w}"] = rolling_slope(px, w)

    # --- MOMENTUM (8 features) ---
    # Retornos acumulados (usamos log-returns sumados = ln(P_t/P_{t-w}))
    vol_21 = lr.rolling(21).std() * np.sqrt(252)  # vol anualizada para escalar

    for w in [21, 63, 126, 252]:
        cum_ret = lr.rolling(w).sum()
        feat[f"ret_{w}"] = cum_ret
        # Momentum escalado por volatilidad (risk-adjusted)
        feat[f"ret_{w}_vs"] = cum_ret / vol_21.replace(0, np.nan)

    # --- VOLATILIDAD (3 features) ---
    feat["vol_21"] = lr.rolling(21).std() * np.sqrt(252)
    feat["vol_63"] = lr.rolling(63).std() * np.sqrt(252)

    # Parkinson (necesita High y Low — no los tenemos en precios ajustados)
    # Aproximación: usamos la desviación intradiaria estimada con retornos diarios
    # como volatilidad de rango alternativa (ratio vol_21 / vol_63)
    feat["vol_ratio"] = feat["vol_21"] / feat["vol_63"].replace(0, np.nan)

    # --- FORTALEZA RELATIVA (5 features) ---
    # Retorno del sector menos retorno medio del universo
    eq_mean_lr = log_ret[SECTOR_TICKERS].mean(axis=1)
    for w in [21, 63, 126, 252]:
        sector_cum = lr.rolling(w).sum()
        universe_cum = eq_mean_lr.rolling(w).sum()
        feat[f"rs_{w}"] = sector_cum - universe_cum

    # Z-score transversal del momentum a 252 días
    # (se calcula después, en el paso transversal)

    all_frames.append(feat)

# Unir en panel largo
daily_panel = pd.concat(all_frames, ignore_index=False)
daily_panel.index.name = "date"

# --- Z-SCORE TRANSVERSAL (1 feature) ---
# Se calcula sobre la sección transversal de cada fecha
print("Calculando z-score transversal del momentum 252d...")

# Pivotear ret_252 para calcular z-score cross-sectional
ret252_wide = daily_panel.pivot_table(index=daily_panel.index, columns="ticker", values="ret_252")
zscore_252 = ret252_wide.sub(ret252_wide.mean(axis=1), axis=0).div(
    ret252_wide.std(axis=1), axis=0
)

# Unpivotar y asignar al panel
zscore_long = zscore_252.stack().reset_index()
zscore_long.columns = ["date", "ticker", "zscore_mom_252"]
zscore_long = zscore_long.set_index("date")

daily_panel = daily_panel.reset_index()
daily_panel = daily_panel.merge(
    zscore_long.reset_index(), on=["date", "ticker"], how="left"
)
daily_panel = daily_panel.set_index("date")

# ──────────────────────────────────────────────
# 4. Construir target: ranking transversal a 21 días
# ──────────────────────────────────────────────
print("Construyendo target (ranking transversal a 21 días)...")

# Retorno futuro a 21 días por sector (usando retornos simples)
fwd_ret_wide = pd.DataFrame(index=prices.index)
for ticker in SECTOR_TICKERS:
    # Retorno acumulado de t+1 a t+21
    fwd_ret_wide[ticker] = (prices[ticker].shift(-TARGET_HORIZON) / prices[ticker]) - 1

# Ranking transversal: 1 = peor, 9 = mejor (para lambdarank, mayor = mejor)
fwd_rank_wide = fwd_ret_wide.rank(axis=1, method="average")

# Pasar a panel largo
fwd_rank_long = fwd_rank_wide.stack().reset_index()
fwd_rank_long.columns = ["date", "ticker", "target_rank"]

fwd_ret_long = fwd_ret_wide.stack().reset_index()
fwd_ret_long.columns = ["date", "ticker", "fwd_ret_21"]

daily_panel = daily_panel.reset_index()
daily_panel = daily_panel.merge(fwd_rank_long, on=["date", "ticker"], how="left")
daily_panel = daily_panel.merge(fwd_ret_long, on=["date", "ticker"], how="left")
daily_panel = daily_panel.set_index("date")

# ──────────────────────────────────────────────
# 5. Filtrar a fin de mes (máscara de decisión, sec. 3.2.1)
# ──────────────────────────────────────────────
print("Aplicando máscara de fin de mes...")

# Identificar último día hábil de cada mes
month_end_mask = daily_panel.index.to_series().groupby(
    daily_panel.index.to_period("M")
).transform("max") == daily_panel.index

dataset = daily_panel[month_end_mask].copy()

# ──────────────────────────────────────────────
# 6. Limpieza final
# ──────────────────────────────────────────────
# Lista de features (excluir ticker, target, fwd_ret)
feature_cols = [c for c in dataset.columns
                if c not in ["ticker", "target_rank", "fwd_ret_21"]]

# Eliminar filas donde las features no estén completas
# (primeras ~252 sesiones / ~12 meses no tienen ret_252 calculable)
before = len(dataset)
dataset = dataset.dropna(subset=feature_cols)
after = len(dataset)
print(f"Filas eliminadas por NaN en features: {before - after}")

# Eliminar filas sin target (últimas 21 sesiones)
before2 = len(dataset)
dataset = dataset.dropna(subset=["target_rank"])
after2 = len(dataset)
print(f"Filas eliminadas por target futuro no disponible: {before2 - after2}")

# ──────────────────────────────────────────────
# 7. Guardar
# ──────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "dataset.parquet")
dataset.to_parquet(out_path, engine="pyarrow")

# ──────────────────────────────────────────────
# 8. Informe
# ──────────────────────────────────────────────
n_dates = dataset.index.nunique()
n_tickers = dataset["ticker"].nunique()
n_features = len(feature_cols)

print()
print("=" * 60)
print("RESUMEN DEL DATASET")
print("=" * 60)
print(f"  Observaciones   : {len(dataset)} ({n_tickers} sectores x {n_dates} fechas)")
print(f"  Fechas           : {dataset.index.min().date()} → {dataset.index.max().date()}")
print(f"  Features         : {n_features}")
print(f"  Target           : target_rank (ranking 1-9, mayor = mejor retorno futuro)")
print(f"  Columna auxiliar : fwd_ret_21 (retorno futuro, solo para evaluacion)")
print()
print("  Lista de features:")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i:2d}. {col}")
print()

# Estadísticas descriptivas de las features
print("  Estadísticas descriptivas (features):")
print(dataset[feature_cols].describe().round(4).to_string())
print()

# Distribución del target
print("  Distribución del target_rank:")
print(dataset["target_rank"].describe().round(2).to_string())
print()

print(f"  Archivo guardado: {out_path}")
print(f"  Tamaño: {os.path.getsize(out_path) / 1024:.0f} KB")
print()
print("Módulo 2 completado. Siguiente: m03_tuning.py")
