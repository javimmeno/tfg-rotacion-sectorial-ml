"""
Módulo 3 — Purged k-fold CV y selección de hiperparámetros
============================================================
Selecciona la configuración óptima de LightGBM (lambdarank) y de
regresión logística (L2) sobre el periodo 2000-2019, usando purged
k-fold cross-validation con embargo equivalente al target de 21 días hábiles.

Protocolo (secs. 3.5 y 3.6.1):
  - 5 pliegues temporales sobre 2000-01 a 2019-12
    (los datos parten de 1999, pero la feature de 252 días deja la primera
     observación usable en ~enero de 2000)
  - Purging: elimina del train observaciones cuyo target solapa con test
  - Embargo: el target abarca 21 días hábiles (~1 mes); como las fechas de
    decisión son mensuales, basta purgar 1 fecha de decisión a cada lado del
    bloque de test para eliminar el solapamiento de etiquetas
  - Criterio: rank IC medio (Spearman entre ranking predicho y observado)
  - Random search N=40 para LightGBM, grid completo (5 puntos) para LogReg
  - Total: N_trials = 45 (input para el Deflated Sharpe Ratio)
  - Semilla: 42

Entrada:  data/dataset.parquet
Salida:   data/best_config.json   — configuración seleccionada
          data/tuning_log.csv     — log completo de todas las configuraciones
"""

import os
import json
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# 0. Parámetros (declarados a priori, sec. 3.6.1)
# ──────────────────────────────────────────────
SEED = 26122003
N_FOLDS = 5
# El target abarca 21 días hábiles (~1 mes). Como las fechas de decisión son
# mensuales, el índice de fechas (dates_list) ya está en pasos mensuales, así
# que el embargo debe expresarse en fechas de decisión, NO en días. Purgar
# 1 fecha a cada lado elimina el único mes cuya etiqueta solapa con el test.
# (Tratar 21 como pasos mensuales purgaría ~21 meses, lo cual era incorrecto.)
EMBARGO_STEPS = 1          # fechas de decisión mensuales (equivale al target de 21 días hábiles)
TUNING_END = "2019-12-31"  # el walk-forward empieza en 2020

# Espacio de búsqueda LightGBM (Tabla 3 de la memoria)
LGBM_SPACE = {
    "learning_rate":    [0.01, 0.05, 0.10],
    "num_leaves":       [15, 31, 63],
    "max_depth":        [-1, 4, 6],
    "min_data_in_leaf": [50, 100, 200],
    "feature_fraction": [0.70, 0.85, 1.00],
    "bagging_fraction": [0.70, 0.85, 1.00],
}
N_LGBM_RANDOM = 40  # configuraciones muestreadas

# Espacio LogReg
LOGREG_C = [0.01, 0.1, 1.0, 10.0, 100.0]

FEATURE_COLS = [
    "dist_ma_20", "dist_ma_60", "dist_ma_200", "slope_60", "slope_120",
    "ret_21", "ret_21_vs", "ret_63", "ret_63_vs", "ret_126", "ret_126_vs",
    "ret_252", "ret_252_vs", "vol_21", "vol_63", "vol_ratio",
    "rs_21", "rs_63", "rs_126", "rs_252", "zscore_mom_252",
]

OUT_DIR = "data"

# ──────────────────────────────────────────────
# 1. Cargar datos y filtrar periodo de tuning
# ──────────────────────────────────────────────
print("=" * 60)
print("MÓDULO 3 — Purged k-fold CV + Tuning")
print("=" * 60)

dataset = pd.read_parquet(os.path.join(OUT_DIR, "dataset.parquet"))
dataset.index = pd.to_datetime(dataset.index)

# Solo periodo de tuning (pre-2020)
tune_data = dataset[dataset.index <= TUNING_END].copy()
n_dates_tune = tune_data.index.nunique()
dates_tune = sorted(tune_data.index.unique())

print(f"Periodo de tuning: {dates_tune[0].date()} -> {dates_tune[-1].date()}")
print(f"Fechas de decisión: {n_dates_tune}")
print(f"Observaciones: {len(tune_data)} ({len(tune_data)//9} fechas x 9 sectores)")
print()

# ──────────────────────────────────────────────
# 2. Construir pliegues temporales con purging y embargo
# ──────────────────────────────────────────────
print("Construyendo pliegues con purging + embargo...")

# Dividir las fechas en N_FOLDS bloques contiguos
fold_size = n_dates_tune // N_FOLDS
fold_assignments = {}
for i, d in enumerate(dates_tune):
    fold_idx = min(i // fold_size, N_FOLDS - 1)
    fold_assignments[d] = fold_idx

tune_data["fold"] = tune_data.index.map(fold_assignments)


def get_purged_train_test(data, dates_list, test_fold, embargo=EMBARGO_STEPS):
    """Devuelve índices de train y test con purging y embargo.

    `embargo` se cuenta en FECHAS DE DECISIÓN (mensuales), no en días.
    Purging: elimina del train las fechas cuyo target (21 días hábiles) solapa
             con el inicio del bloque de test (las `embargo` fechas previas).
    Embargo: elimina del train las `embargo` fechas posteriores al fin del test.
    """
    test_mask = data["fold"] == test_fold
    test_dates = sorted(data[test_mask].index.unique())
    test_start = test_dates[0]
    test_end = test_dates[-1]

    # Todas las fechas que NO son del fold de test
    train_candidates = data[~test_mask].copy()

    # Purging: eliminar fechas de train cuyo target solapa con test
    # Si una obs tiene fecha t, su label usa datos hasta t+21 días hábiles
    # Si t + 21 >= test_start, hay solapamiento
    # Buscamos la fecha de purge: la fecha que está EMBARGO posiciones
    # antes de test_start en el calendario de fechas
    test_start_idx = dates_list.index(test_start)
    purge_start_idx = max(0, test_start_idx - embargo)
    purge_start_date = dates_list[purge_start_idx]

    # Embargo: eliminar fechas de train dentro de embargo días después de test_end
    test_end_idx = dates_list.index(test_end)
    embargo_end_idx = min(len(dates_list) - 1, test_end_idx + embargo)
    embargo_end_date = dates_list[embargo_end_idx]

    # Aplicar purging y embargo
    train_clean = train_candidates[
        (train_candidates.index < purge_start_date) |
        (train_candidates.index > embargo_end_date)
    ]

    test_set = data[test_mask]

    return train_clean, test_set


# Verificar los pliegues
for fold in range(N_FOLDS):
    train, test = get_purged_train_test(tune_data, dates_tune, fold)
    train_dates = sorted(train.index.unique())
    test_dates = sorted(test.index.unique())
    n_purged = len(tune_data[tune_data["fold"] != fold]) - len(train)
    print(f"  Fold {fold}: test {test_dates[0].date()}->{test_dates[-1].date()} "
          f"| train={len(train)//9} fechas | test={len(test)//9} fechas "
          f"| purged+embargo={n_purged//9} fechas")


# ──────────────────────────────────────────────
# 3. Función de evaluación: rank IC
# ──────────────────────────────────────────────
def rank_ic_by_date(y_true, y_pred, dates):
    """Calcula rank IC (Spearman) promedio por fecha.
    Para cada fecha, calcula la correlación entre ranking predicho
    y ranking observado sobre los 9 sectores."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "date": dates})
    ics = []
    for _, group in df.groupby("date"):
        if len(group) < 3:
            continue
        corr, _ = spearmanr(group["true"], group["pred"])
        if not np.isnan(corr):
            ics.append(corr)
    return np.mean(ics) if ics else 0.0


# ──────────────────────────────────────────────
# 4. Evaluar una configuración de LightGBM
# ──────────────────────────────────────────────
def evaluate_lgbm(params, data, dates_list):
    """Evalúa una configuración de LightGBM con purged k-fold CV.
    Devuelve rank IC medio y std entre folds."""
    fold_ics = []

    for fold in range(N_FOLDS):
        train, test = get_purged_train_test(data, dates_list, fold)

        X_train = train[FEATURE_COLS].values
        y_train = train["target_rank"].values
        X_test = test[FEATURE_COLS].values
        y_test = test["target_rank"].values

        # Estandarizar features (solo con estadísticos del train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Grupos para lambdarank (9 sectores por fecha)
        train_dates_unique = sorted(train.index.unique())
        train_groups = [9] * len(train_dates_unique)
        test_dates_unique = sorted(test.index.unique())
        test_groups = [9] * len(test_dates_unique)

        # Entrenar LightGBM ranker
        lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups)
        lgb_val = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=lgb_train)

        lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [3],
            "verbosity": -1,
            "seed": SEED,
            **params,
        }
        if params.get("bagging_fraction", 1.0) < 1.0:
            lgb_params["bagging_freq"] = 1

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

        model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=callbacks,
        )

        # Predecir y calcular rank IC
        preds = model.predict(X_test)
        ic = rank_ic_by_date(y_test, preds, test.index)
        fold_ics.append(ic)

    return np.mean(fold_ics), np.std(fold_ics)


# ──────────────────────────────────────────────
# 5. Evaluar una configuración de LogReg
# ──────────────────────────────────────────────
def evaluate_logreg(C, data, dates_list):
    """Evalúa regresión logística con penalización L2.
    Se entrena como clasificación ordinal simplificada:
    el score de cada sector es la probabilidad predicha de la clase
    positiva, y se rankea por ese score."""
    fold_ics = []

    for fold in range(N_FOLDS):
        train, test = get_purged_train_test(data, dates_list, fold)

        X_train = train[FEATURE_COLS].values
        y_train = train["target_rank"].values
        X_test = test[FEATURE_COLS].values
        y_test = test["target_rank"].values

        # Binarizar: top-3 = 1, resto = 0 (para LogReg)
        y_train_bin = (y_train >= 7).astype(int)  # ranks 7,8,9 = top-3

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=SEED)
        model.fit(X_train, y_train_bin)

        # Score = probabilidad de ser top-3 (se usa para rankear)
        preds = model.predict_proba(X_test)[:, 1]
        ic = rank_ic_by_date(y_test, preds, test.index)
        fold_ics.append(ic)

    return np.mean(fold_ics), np.std(fold_ics)


# ──────────────────────────────────────────────
# 6. Random search LightGBM (N=40)
# ──────────────────────────────────────────────
print()
print("-" * 60)
print(f"RANDOM SEARCH LightGBM (N={N_LGBM_RANDOM}, semilla={SEED})")
print("-" * 60)

rng = np.random.RandomState(SEED)
all_keys = list(LGBM_SPACE.keys())
lgbm_results = []

for i in range(N_LGBM_RANDOM):
    params = {k: rng.choice(LGBM_SPACE[k]) for k in all_keys}
    # Convertir tipos numpy a Python nativo para JSON
    params = {k: (int(v) if isinstance(v, (np.integer,)) else
                  float(v) if isinstance(v, (np.floating,)) else v)
              for k, v in params.items()}

    mean_ic, std_ic = evaluate_lgbm(params, tune_data, dates_tune)
    lgbm_results.append({
        "model": "LightGBM",
        "config_id": i + 1,
        "mean_rank_ic": round(mean_ic, 6),
        "std_rank_ic": round(std_ic, 6),
        **params,
    })
    print(f"  [{i+1:2d}/{N_LGBM_RANDOM}] IC={mean_ic:.4f} +/- {std_ic:.4f}  "
          f"lr={params['learning_rate']} leaves={params['num_leaves']} "
          f"depth={params['max_depth']}")

# ──────────────────────────────────────────────
# 7. Grid search LogReg (N=5)
# ──────────────────────────────────────────────
print()
print("-" * 60)
print(f"GRID SEARCH LogReg (N={len(LOGREG_C)})")
print("-" * 60)

logreg_results = []
for j, C in enumerate(LOGREG_C):
    mean_ic, std_ic = evaluate_logreg(C, tune_data, dates_tune)
    logreg_results.append({
        "model": "LogReg",
        "config_id": N_LGBM_RANDOM + j + 1,
        "mean_rank_ic": round(mean_ic, 6),
        "std_rank_ic": round(std_ic, 6),
        "C": C,
    })
    print(f"  [{j+1}/{len(LOGREG_C)}] C={C:6.2f}  IC={mean_ic:.4f} +/- {std_ic:.4f}")

# ──────────────────────────────────────────────
# 8. Selección de la mejor configuración
# ──────────────────────────────────────────────
print()
print("=" * 60)
print("SELECCIÓN FINAL")
print("=" * 60)

all_results = lgbm_results + logreg_results
results_df = pd.DataFrame(all_results)

# Mejor LightGBM: mayor mean_rank_ic, desempate por menor std
best_lgbm_df = results_df[results_df["model"] == "LightGBM"].copy()
best_lgbm_df = best_lgbm_df.sort_values(
    ["mean_rank_ic", "std_rank_ic"], ascending=[False, True]
)
best_lgbm = best_lgbm_df.iloc[0]

# Verificar regla de desempate (5%)
if len(best_lgbm_df) > 1:
    top2 = best_lgbm_df.head(2)
    diff_pct = abs(top2.iloc[0]["mean_rank_ic"] - top2.iloc[1]["mean_rank_ic"]) / \
               abs(top2.iloc[0]["mean_rank_ic"]) * 100 if top2.iloc[0]["mean_rank_ic"] != 0 else 0
    if diff_pct < 5:
        # Desempate por menor varianza
        best_lgbm = top2.sort_values("std_rank_ic").iloc[0]
        print(f"  Desempate aplicado (diferencia < 5%): se elige config con menor std")

# Mejor LogReg
best_logreg_df = results_df[results_df["model"] == "LogReg"].copy()
best_logreg = best_logreg_df.sort_values("mean_rank_ic", ascending=False).iloc[0]

print(f"\n  Mejor LightGBM (config #{int(best_lgbm['config_id'])}):")
print(f"    Rank IC medio: {best_lgbm['mean_rank_ic']:.4f} +/- {best_lgbm['std_rank_ic']:.4f}")
lgbm_params_final = {k: best_lgbm[k] for k in LGBM_SPACE.keys()
                     if k in best_lgbm.index}
for k, v in lgbm_params_final.items():
    print(f"    {k}: {v}")

print(f"\n  Mejor LogReg (config #{int(best_logreg['config_id'])}):")
print(f"    Rank IC medio: {best_logreg['mean_rank_ic']:.4f} +/- {best_logreg['std_rank_ic']:.4f}")
print(f"    C: {best_logreg['C']}")

# ──────────────────────────────────────────────
# 9. Guardar
# ──────────────────────────────────────────────
# Configuración final
config = {
    "lgbm": {
        "params": {k: (int(v) if isinstance(v, (np.integer, int)) and k != "feature_fraction"
                       else float(v))
                   for k, v in lgbm_params_final.items()},
        "mean_rank_ic": float(best_lgbm["mean_rank_ic"]),
        "std_rank_ic": float(best_lgbm["std_rank_ic"]),
    },
    "logreg": {
        "C": float(best_logreg["C"]),
        "mean_rank_ic": float(best_logreg["mean_rank_ic"]),
        "std_rank_ic": float(best_logreg["std_rank_ic"]),
    },
    "tuning_meta": {
        "n_trials_total": len(all_results),
        "n_lgbm": N_LGBM_RANDOM,
        "n_logreg": len(LOGREG_C),
        "n_folds": N_FOLDS,
        "embargo_steps": EMBARGO_STEPS,
        "embargo_days_equiv": 21,
        "seed": SEED,
        "tuning_period": f"{dates_tune[0].date()} -> {dates_tune[-1].date()}",
    },
}

config_path = os.path.join(OUT_DIR, "best_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# Log completo
log_path = os.path.join(OUT_DIR, "tuning_log.csv")
results_df.to_csv(log_path, index=False, encoding="utf-8")

print(f"\n  Configuración guardada: {config_path}")
print(f"  Log completo: {log_path}")
print(f"  N_trials = {len(all_results)} (input para DSR)")
print()
print("=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  LightGBM rank IC: {best_lgbm['mean_rank_ic']:.4f}")
print(f"  LogReg rank IC:   {best_logreg['mean_rank_ic']:.4f}")
print(f"  Diferencia:       {best_lgbm['mean_rank_ic'] - best_logreg['mean_rank_ic']:.4f}")
print()
print("Módulo 3 completado. Siguiente: m04_walkforward.py")
