"""
Módulo 1 — Adquisición y verificación de datos
================================================
Descarga precios ajustados diarios de los 9 ETF sectoriales SPDR + SPY
desde Yahoo Finance, realiza controles de calidad y guarda un snapshot
en formato Parquet con hash SHA-256 para reproducibilidad.

Universo (sec. 3.2 de la memoria):
  XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY  (9 sectoriales)
  SPY  (benchmark)

Periodo: 1999-01-01 a 2026-05-30

Salida:
  data/prices.parquet   — panel de precios ajustados (columnas = tickers)
  data/prices_sha256.txt — hash del archivo para trazabilidad
  Informe de calidad impreso en consola
"""

import os
import hashlib
import datetime as dt

import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────
# 1. Parámetros
# ──────────────────────────────────────────────
TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY"]
START = "1999-01-01"
END = "2026-05-30"
OUT_DIR = "data"

# ──────────────────────────────────────────────
# 2. Descarga
# ──────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("MÓDULO 1 — Adquisición de datos")
print("=" * 60)
print(f"Tickers : {TICKERS}")
print(f"Periodo : {START} → {END}")
print()

# Descargar precios ajustados por dividendos y splits
print("Descargando datos de Yahoo Finance...")
raw = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    auto_adjust=True,   # precios ya ajustados
    progress=True,
)

# Extraer solo el cierre ajustado
prices = raw["Close"].copy()

# Asegurar que las columnas están en el orden correcto
prices = prices[TICKERS]

print(f"\nDimensiones descargadas: {prices.shape[0]} días × {prices.shape[1]} tickers")
print(f"Rango de fechas: {prices.index[0].date()} → {prices.index[-1].date()}")
print()

# ──────────────────────────────────────────────
# 3. Controles de calidad (sec. 3.2)
# ──────────────────────────────────────────────
print("-" * 60)
print("CONTROLES DE CALIDAD")
print("-" * 60)

# 3a. Valores ausentes
missing = prices.isnull().sum()
print("\n[1] Valores ausentes por ticker:")
for t in TICKERS:
    status = "✓ OK" if missing[t] == 0 else f"⚠ {missing[t]} NaN"
    print(f"    {t:4s}: {status}")

total_missing = missing.sum()
if total_missing > 0:
    print(f"\n    → Rellenando {total_missing} NaN con forward-fill (último precio conocido)")
    prices = prices.ffill()
    # Si quedan NaN al inicio (antes del primer precio), backfill
    remaining = prices.isnull().sum().sum()
    if remaining > 0:
        prices = prices.bfill()
        print(f"    → Backfill aplicado para {remaining} NaN iniciales")

# 3b. Continuidad de fechas (no hay gaps de más de 5 días naturales)
date_diffs = pd.Series(prices.index).diff().dropna()
max_gap = date_diffs.max()
print(f"\n[2] Continuidad temporal:")
print(f"    Gap máximo entre sesiones: {max_gap.days} días naturales", end="")
if max_gap.days <= 7:
    print(" ✓ OK (fin de semana, festivos o cierre puntual de mercado)")
else:
    print(f" ⚠ Gap inusual — revisar")

# 3c. Precios no negativos / no nulos
min_prices = prices.min()
print(f"\n[3] Precio mínimo por ticker:")
for t in TICKERS:
    status = "✓ OK" if min_prices[t] > 0 else "⚠ Precio ≤ 0"
    print(f"    {t:4s}: ${min_prices[t]:.2f} {status}")

# 3d. Retornos extremos (detectar posibles errores de datos)
returns = prices.pct_change().dropna()
max_abs_ret = returns.abs().max()
print(f"\n[4] Retorno diario absoluto máximo por ticker:")
for t in TICKERS:
    flag = " ⚠ >20%" if max_abs_ret[t] > 0.20 else ""
    print(f"    {t:4s}: {max_abs_ret[t]*100:.1f}%{flag}")

# 3e. Verificación de consistencia SPY vs media sectorial
# La suma ponderada por capitalización no es posible sin datos de cap,
# pero la media equiponderada debería correlacionar alto con SPY
sector_tickers = [t for t in TICKERS if t != "SPY"]
eq_weighted_ret = returns[sector_tickers].mean(axis=1)
spy_ret = returns["SPY"]
corr = eq_weighted_ret.corr(spy_ret)
print(f"\n[5] Correlación (media equiponderada sectores vs SPY): {corr:.4f}", end="")
if corr > 0.90:
    print(" ✓ OK (consistencia razonable)")
else:
    print(" ⚠ Correlación baja — revisar datos")

# ──────────────────────────────────────────────
# 4. Guardar snapshot
# ──────────────────────────────────────────────
parquet_path = os.path.join(OUT_DIR, "prices.parquet")
prices.to_parquet(parquet_path, engine="pyarrow")

# Hash SHA-256
sha256 = hashlib.sha256()
with open(parquet_path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        sha256.update(chunk)
hash_hex = sha256.hexdigest()

hash_path = os.path.join(OUT_DIR, "prices_sha256.txt")
with open(hash_path, "w", encoding="utf-8") as f:
    f.write(f"# SHA-256 de {parquet_path}\n")
    f.write(f"# Generado: {dt.datetime.now().isoformat()}\n")
    f.write(f"# Periodo: {START} → {END}\n")
    f.write(f"# Tickers: {', '.join(TICKERS)}\n")
    f.write(f"{hash_hex}\n")

print()
print("-" * 60)
print("SNAPSHOT GUARDADO")
print("-" * 60)
print(f"  Archivo : {parquet_path}")
print(f"  Tamaño  : {os.path.getsize(parquet_path) / 1024:.0f} KB")
print(f"  SHA-256 : {hash_hex}")
print(f"  Hash en : {hash_path}")
print()

# ──────────────────────────────────────────────
# 5. Resumen final
# ──────────────────────────────────────────────
print("=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  Días hábiles : {prices.shape[0]}")
print(f"  Tickers      : {prices.shape[1]} ({', '.join(prices.columns.tolist())})")
print(f"  Primer día   : {prices.index[0].date()}")
print(f"  Último día   : {prices.index[-1].date()}")
print(f"  NaN restantes: {prices.isnull().sum().sum()}")
print()
print("Módulo 1 completado. Siguiente: m02_features.py")
