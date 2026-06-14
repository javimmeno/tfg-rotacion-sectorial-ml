# Rotación sectorial táctica con machine learning

Trabajo de Fin de Grado (GRETI, ESEIAAT - UPC). Pipeline reproducible que
evalúa si un modelo de aprendizaje automático puede mejorar una estrategia
de rotación entre ETF sectoriales SPDR frente a comprar y mantener el S&P 500.

El resultado central es honesto: sobre la ventana extendida 2012-2026 el
modelo no supera al índice de forma estadísticamente significativa después
de costes. El valor del trabajo reside en el marco experimental, diseñado
para evitar el sobreajuste de backtest habitual en este tipo de estudios.

## Universo y datos

- 9 ETF sectoriales SPDR (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) y SPY como benchmark.
- Precios ajustados diarios desde Yahoo Finance (1999-2026).
- Decisiones mensuales (último día hábil), target de ranking transversal a 21 días.

## Metodología

- Selección de hiperparámetros con purged k-fold cross-validation con embargo.
- Walk-forward con reentrenamiento anual sobre periodo reservado.
- Costes de transacción y penalización de la rotación.
- Inferencia con stationary bootstrap, test de Sharpe de Ledoit-Wolf,
  Deflated Sharpe Ratio y corrección de Holm-Bonferroni.
- Hiperparámetros congelados antes del test; sin iteración tras ver resultados fuera de muestra.

## Estructura del pipeline

Los módulos se ejecutan en orden:

| Módulo | Función |
|---|---|
| `m01_data_acquisition.py` | Descarga y verifica los precios; guarda un snapshot con hash SHA-256. |
| `m02_features_target.py` | Construye las 21 features y el target de ranking. |
| `m03_tuning.py` | Purged k-fold CV y selección de hiperparámetros. |
| `m04_walkforward_backtest.py` | Walk-forward y backtest de las 5 estrategias con costes. |
| `m05_inference.py` | Inferencia estadística sobre los resultados. |
| `m06_robustness.py` | Pruebas de robustez (horizonte, k, coste, subperiodos). |
| `m07_figures.py` | Genera las figuras del capítulo 5. |
| `m08_extended_window.py` | Validación sobre la ventana extendida 2012-2026. |

## Reproducibilidad

- Semilla fija (26122003) en todos los módulos.
- Datos públicos y dependencias fijadas en `requirements.txt`.

## Uso

```
pip install -r requirements.txt
python m01_data_acquisition.py
python m02_features_target.py
python m03_tuning.py
python m04_walkforward_backtest.py
python m05_inference.py
python m06_robustness.py
python m07_figures.py
python m08_extended_window.py
```

Cada script se ejecuta en menos de un minuto en un equipo doméstico.

## Autor

Javier Marín Moreno - TFG dirigido por Daniel Fernández Martínez (ESEIAAT - UPC), 2026.

## Licencia

MIT. Ver el archivo [LICENSE](LICENSE).
