"""
Módulo 7 - Figuras para el capítulo 5

Genera:
  fig_equity_curves.png    - curvas de equity de las 5 estrategias
  fig_sharpe_ci.png        - Sharpe con intervalos de confianza bootstrap
  fig_subperiod.png        - retornos anuales por estrategia
  fig_robustness.png       - resumen de sensibilidad (k, coste, horizonte)
  fig_feature_importance.png - importancia de features del último modelo LightGBM
  fig_drawdowns.png        - drawdowns comparados

Entrada: data/*.parquet, data/*.json
Salida:  figures/*.png
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

OUT_DIR = "figures"
DATA_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 26122003
SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FEATURE_COLS = [
    "dist_ma_20", "dist_ma_60", "dist_ma_200", "slope_60", "slope_120",
    "ret_21", "ret_21_vs", "ret_63", "ret_63_vs", "ret_126", "ret_126_vs",
    "ret_252", "ret_252_vs", "vol_21", "vol_63", "vol_ratio",
    "rs_21", "rs_63", "rs_126", "rs_252", "zscore_mom_252",
]

# Estilo general
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "SPY_BH": "#1f77b4",
    "EqWeight": "#aec7e8",
    "TSMom12m": "#ff7f0e",
    "LogReg": "#2ca02c",
    "LightGBM": "#d62728",
}
LABELS = {
    "SPY_BH": "SPY B&H",
    "EqWeight": "Equiponderado",
    "TSMom12m": "TS-Mom 12m",
    "LogReg": "LogReg L2",
    "LightGBM": "LightGBM",
}

# Cargar datos
backtest = pd.read_parquet(os.path.join(DATA_DIR, "backtest_results.parquet"))
equity = (1 + backtest).cumprod()

with open(os.path.join(DATA_DIR, "inference.json"), "r") as f:
    inference = json.load(f)
with open(os.path.join(DATA_DIR, "robustness.json"), "r") as f:
    robustness = json.load(f)

strategies = ["SPY_BH", "EqWeight", "TSMom12m", "LogReg", "LightGBM"]


# ───── Figura 1: Curvas de equity ─────
print("Generando fig_equity_curves.png...")

fig, ax = plt.subplots(figsize=(10, 5))
for strat in strategies:
    ax.plot(equity.index, equity[strat],
            label=LABELS[strat], color=COLORS[strat],
            linewidth=1.5 if strat in ["SPY_BH", "LightGBM"] else 1.0,
            alpha=1.0 if strat in ["SPY_BH", "LightGBM"] else 0.7)

ax.set_ylabel("Valor de la cartera (base 1.0)")
ax.set_xlabel("")
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# Sombrear 2022 (bear market)
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
           alpha=0.08, color="gray")
ax.text(pd.Timestamp("2022-06-15"), ax.get_ylim()[0] * 1.05,
        "2022", ha="center", fontsize=8, color="gray")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_equity_curves.png"), bbox_inches="tight")
plt.close()


# ───── Figura 2: Sharpe con IC bootstrap ─────
print("Generando fig_sharpe_ci.png...")

ci = inference["bootstrap_ci"]
fig, ax = plt.subplots(figsize=(8, 4.5))

x_pos = np.arange(len(strategies))
sharpes = [ci[s]["sharpe_obs"] for s in strategies]
lows = [ci[s]["ci_lo"] for s in strategies]
highs = [ci[s]["ci_hi"] for s in strategies]
errors_lo = [s - l for s, l in zip(sharpes, lows)]
errors_hi = [h - s for s, h in zip(sharpes, highs)]

bars = ax.bar(x_pos, sharpes,
              color=[COLORS[s] for s in strategies],
              edgecolor="white", linewidth=0.5, width=0.6)
ax.errorbar(x_pos, sharpes, yerr=[errors_lo, errors_hi],
            fmt="none", ecolor="black", capsize=4, linewidth=1.2)

ax.axhline(y=ci["SPY_BH"]["sharpe_obs"], color=COLORS["SPY_BH"],
           linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(y=0, color="black", linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels([LABELS[s] for s in strategies], fontsize=9)
ax.set_ylabel("Sharpe Ratio (anualizado)")
ax.text(0.98, 0.97, "Barras de error: IC 95% bootstrap\n(B=5000, bloque=21d)",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_sharpe_ci.png"), bbox_inches="tight")
plt.close()


# ───── Figura 3: Subperiod analysis ─────
print("Generando fig_subperiod.png...")

sub = robustness["subperiod"]
years = sorted(sub.keys())
strats_sub = ["SPY_BH", "LightGBM", "TSMom12m"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel izquierdo: retornos anuales
x = np.arange(len(years))
width = 0.25
for i, s in enumerate(strats_sub):
    vals = [sub[y][s]["return"] * 100 for y in years]
    ax1.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s])

ax1.set_xticks(x + width)
ax1.set_xticklabels(years)
ax1.set_ylabel("Retorno anual (%)")
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.legend(fontsize=8)

# Panel derecho: Sharpe por anio
for i, s in enumerate(strats_sub):
    vals = [sub[y][s]["sharpe"] for y in years]
    ax2.bar(x + i * width, vals, width, label=LABELS[s], color=COLORS[s])

ax2.set_xticks(x + width)
ax2.set_xticklabels(years)
ax2.set_ylabel("Sharpe Ratio")
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.legend(fontsize=8)

fig.suptitle("")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_subperiod.png"), bbox_inches="tight")
plt.close()


# ───── Figura 4: Resumen de robustez ─────
print("Generando fig_robustness.png...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Panel 1: sensibilidad a k
k_data = robustness["sensitivity_k"]
ks = sorted(k_data.keys())
ax = axes[0]
ax.bar(ks, [k_data[k]["Sharpe"] for k in ks],
       color=["#d62728" if "3" in k else "#888888" for k in ks],
       edgecolor="white", width=0.5)
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sensibilidad a k", fontsize=10)
ax.axhline(y=0, color="black", linewidth=0.5)

# Panel 2: sensibilidad al coste
c_data = robustness["sensitivity_cost"]
cs = sorted(c_data.keys(), key=lambda x: int(x.replace("bps", "")))
ax = axes[1]
ax.bar(cs, [c_data[c]["Sharpe"] for c in cs],
       color=["#d62728" if "10" in c else "#888888" for c in cs],
       edgecolor="white", width=0.5)
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sensibilidad al coste", fontsize=10)
ax.axhline(y=0, color="black", linewidth=0.5)

# Panel 3: sensibilidad al horizonte
h_data = robustness["sensitivity_horizon"]
hs = sorted(h_data.keys(), key=lambda x: int(x.replace("d", "")))
ax = axes[2]
ax.bar(hs, [h_data[h]["Sharpe"] for h in hs],
       color=["#d62728" if "21" in h else "#888888" for h in hs],
       edgecolor="white", width=0.5)
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sensibilidad al horizonte", fontsize=10)
ax.axhline(y=0, color="black", linewidth=0.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_robustness.png"), bbox_inches="tight")
plt.close()


# ───── Figura 5: Drawdowns ─────
print("Generando fig_drawdowns.png...")

fig, ax = plt.subplots(figsize=(10, 4))
for strat in ["SPY_BH", "LightGBM", "TSMom12m"]:
    eq = equity[strat]
    dd = (eq - eq.cummax()) / eq.cummax()
    ax.fill_between(dd.index, dd.values, 0,
                    alpha=0.3 if strat != "LightGBM" else 0.5,
                    color=COLORS[strat], label=LABELS[strat])
    ax.plot(dd.index, dd.values, color=COLORS[strat], linewidth=0.5)

ax.set_ylabel("Drawdown (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_drawdowns.png"), bbox_inches="tight")
plt.close()


# ───── Figura 6: Feature importance ─────
print("Generando fig_feature_importance.png...")

# Entrenar el último modelo (con todos los datos hasta 2023) para extraer importancia
dataset = pd.read_parquet(os.path.join(DATA_DIR, "dataset.parquet"))
dataset.index = pd.to_datetime(dataset.index)
train = dataset[dataset.index <= "2023-12-31"]

with open(os.path.join(DATA_DIR, "best_config.json"), "r") as f:
    config = json.load(f)
lgbm_params = config["lgbm"]["params"]

scaler = StandardScaler()
X = scaler.fit_transform(train[FEATURE_COLS].values)
y = train["target_rank"].values
groups = [9] * train.index.nunique()

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

model = lgb.train(lgb_p, lgb.Dataset(X, label=y, group=groups), num_boost_round=500)
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=FEATURE_COLS).sort_values(ascending=True)

# Agrupar por familia
family_colors = {}
for f in FEATURE_COLS:
    if f.startswith("dist_ma") or f.startswith("slope"):
        family_colors[f] = "#1f77b4"  # tendencia
    elif f.startswith("ret_"):
        family_colors[f] = "#ff7f0e"  # momentum
    elif f.startswith("vol"):
        family_colors[f] = "#2ca02c"  # volatilidad
    else:
        family_colors[f] = "#9467bd"  # fortaleza relativa

fig, ax = plt.subplots(figsize=(8, 6))
colors = [family_colors[f] for f in feat_imp.index]
ax.barh(range(len(feat_imp)), feat_imp.values, color=colors, edgecolor="white",
        height=0.7)
ax.set_yticks(range(len(feat_imp)))
ax.set_yticklabels(feat_imp.index, fontsize=8.5)
ax.set_xlabel("Importancia (gain)")

# Leyenda de familias
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1f77b4", label="Tendencia"),
    Patch(facecolor="#ff7f0e", label="Momentum"),
    Patch(facecolor="#2ca02c", label="Volatilidad"),
    Patch(facecolor="#9467bd", label="Fortaleza relativa"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_feature_importance.png"), bbox_inches="tight")
plt.close()

# ───── Figura 7: Rank IC fuera de muestra por año ─────
print("Generando fig_rank_ic_oos.png...")

# Rank IC anual del modelo LightGBM en el walk-forward fuera de muestra,
# recalculado a partir de las predicciones guardadas por m04.
preds = pd.read_parquet(os.path.join(DATA_DIR, "predictions.parquet"))
preds["date"] = pd.to_datetime(preds["date"])

ic_by_year = {}
for year in sorted(preds["date"].dt.year.unique()):
    yd = preds[preds["date"].dt.year == year]
    ics = []
    for _, g in yd.groupby("date"):
        ics.append(g[["target_rank", "pred_lgbm"]].corr(method="spearman").iloc[0, 1])
    ic_by_year[year] = float(np.nanmean(ics))

years = [str(y) for y in ic_by_year.keys()]
ic_vals = list(ic_by_year.values())
VALIDATION_IC = 0.054  # rank IC medio en validacion (Tabla 5)

fig, ax = plt.subplots(figsize=(9, 4.5))
bar_colors = [COLORS["LightGBM"] if v >= 0 else "#7f7f7f" for v in ic_vals]
ax.bar(years, ic_vals, color=bar_colors, edgecolor="white", width=0.65)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(VALIDATION_IC, color="#1f77b4", linestyle="--", linewidth=1.2,
           label=f"Rank IC en validación ({VALIDATION_IC:.3f})")
ax.set_ylabel("Rank IC medio anual")
ax.set_xlabel("Año (walk-forward fuera de muestra)")
ax.legend(loc="upper left", fontsize=8)
for i, v in enumerate(ic_vals):
    ax.text(i, v + (0.012 if v >= 0 else -0.012), f"{v:.2f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylim(min(ic_vals) - 0.05, max(max(ic_vals), VALIDATION_IC) + 0.05)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_rank_ic_oos.png"), bbox_inches="tight")
plt.close()

# ─────
print()
print(f"Todas las figuras guardadas en {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f:35s} ({size_kb:.0f} KB)")
print()
print("Modulo 7 completado. Pipeline finalizado.")
