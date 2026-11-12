# Fase 08 — estado reproducible y ejecución segura (12-11-2026)

**Rama de trabajo:** `experiment/study-a-v2-integracion-20261112`.
**Origen:** `experiment/study-a-v2-protocol-20261109` (`f93d61dd34c55cc9b10b9ebb50f61c4e7a91a953`).
**No mezclar resultados v1 con v2; no sobrescribir `main`, `models/`, salidas previas ni el manuscrito Fase 06A.**

## 1. Implementado en esta rama

- `trainer.py`: concurso de cinco familias (XGBoost, LightGBM, CatBoost, RandomForest, HistGradientBoosting); los ensayos de Optuna usan **solo** bloque `tune` y umbral fijo 0.50. El umbral final se selecciona **solo** con `calibrate`. El preprocesador se ajusta exclusivamente sobre `train` (70/15/15). No refit con tune/calibrate, deliberadamente para evitar su reutilización en ajuste.
- `run_comparison_v2.py`: tres políticas; evaluación `test-then-train`; solicitud k → comparación/promoción k+1 → primer uso k+2. Modelo inicial común. Archivos experimentales nuevos, métricas por ventana, FP/FN, tiempos adicionales, medición CPU/RSS opcional y logs de candidatos. No se hace hot-swap real en A.
- `prepare_epochs.py`: separa IEEE-CIS cronológicamente en tres épocas de origen disjuntas (opción configurable); genera versiones natural y de cambio sintético de covariables. No presupone que sean muestras bancarias independientes.
- `run_study_v2.py`: combina épocas, escenarios, semillas y alternativas de periodicidad en ejecución secuencial; reanuda corridas cerradas y consolida resultados descriptivos.
- `tests/test_study_a_v2_integration.py`: smoke tests sobre datos sintéticos. CI GitHub registra las verificaciones; estos tests **no son** validación de rendimiento IEEE-CIS.

## 2. Atención: error de procedencia en Estudio B histórico

Al leer `src/prepare_data.py` se confirmó que:
```python
X_lote1.assign(Class=y_lote1).to_csv("data/prod_lote1_normal.csv", index=False)
X_lote1.assign(Class=y_lote1).to_csv("data/prod_lote3_post_retrain.csv", index=False)
```
Por tanto, **lote 3 es una copia del lote 1**, NO un lote cronológicamente posterior ni una validación independiente. Además, el baseline funcional se preparó mediante particiones aleatorias estratificadas. El CatBoost del workflow histórico no puede recibir un test «posterior estricto» sin comprobar primero qué datos estuvieron disponibles en su entrenamiento/selección. Para la evidencia nueva de B se debe reconstruir un protocolo con corte temporal anterior al reentrenamiento y reservar datos futuros inaccesibles durante entrenamiento, ajuste del umbral, selección y promoción. Conservar F1=0.9619 solo como evidencia funcional sobre lote reutilizado.

## 3. Comandos Windows/PowerShell para el usuario (NO ejecutar hasta comprobar entorno)

Desde la raíz del repositorio:
```powershell
git status -sb
git fetch origin
git switch experiment/study-a-v2-integracion-20261112
git pull --ff-only origin experiment/study-a-v2-integracion-20261112
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-experiments.txt pytest
.\venv\Scripts\python.exe -m pytest -q tests/test_study_a_v2.py tests/test_study_a_v2_integration.py
```

Las pruebas con datos sintéticos no requieren IEEE-CIS. En cambio **para el benchmark** deben estar disponibles localmente los archivos crudos originales `train_transaction.csv` y `train_identity.csv`. `data/datos_entrenamiento.csv` no sirve para v2 porque ya perdió TransactionDT y TransactionID.

Preparar épocas (no sobrescribe una carpeta existente no vacía):
```powershell
.\venv\Scripts\python.exe -m src.experiments.v2.prepare_epochs --transactions train_transaction.csv --identity train_identity.csv --output data/experiments/raw-v2-epochs --epochs 3 --max-rows 120000 --windows 8
```

**Primero**, hacer piloto de una época y una semilla, un trial por familia; carpeta de salida nueva:
```powershell
.\venv\Scripts\python.exe -m src.experiments.v2.run_comparison_v2 --input data/experiments/raw-v2-epochs/epoch_01/natural --output data/experiments/results-v2-integrated-smoke-42 --seed 42 --trials 1 --every 4 --profile-resources
```

Si pasa, analizar `configuration.json`, `comparison.csv`, `summary.csv` y `candidate_registry.json` antes del estudio ampliado. **No interpretar un piloto de 1 trial como la conclusión final.**

Estudio exploratorio de varias épocas/semillas (mucho más pesado; solo con tiempo/hardware disponible y presupuesto previamente fijado):
```powershell
.\venv\Scripts\python.exe -m src.experiments.v2.run_study_v2 --epochs-dir data/experiments/raw-v2-epochs --output data/experiments/study-a-v2-confirmatory --seeds 42 43 44 --every 2 3 4 --trials 3 --profile-resources
```
Para una comparación más profunda, estudiar 15 trials por familia, SIEMPRE igual para todas las políticas/corridas comparadas y sin escoger el presupuesto según quién gane. Las tres semillas no sustituyen épocas cronológicas diferentes. Un límite máximo de intentos (`--max-attempts`) **no garantiza que ambas políticas ejecuten exactamente el mismo número de reentrenamientos**: el análisis de presupuestos estrictamente iguales todavía requiere control o emparejamiento adicional.

## 4. Criterios de aceptación antes de atribuir resultados a v2

1. Pasar los tests existentes y de integración.
2. Verificar que todos los `TransactionID` de historia/holdout/ventanas sean únicos en cada escenario, y que no exista regresión de `TransactionDT`.
3. Verificar que el preprocesador solo ajuste el bloque train, que Optuna no observe calibrate y que la promoción use una ventana posterior al ajuste del candidato.
4. Confirmar que el modelo promovido solo se usa desde k+2. Registrar eventos y fallos por política/ventana.
5. Guardar la configuración, versión de código, métricas, consumo y evidencia bruta sin sobrescribir carpetas.
6. Reportar diferencias de F1 en conjunto con FP, FN, entrenamiento, CPU, RSS, número de intentos y promociones.
7. No afirmar ahorro cloud, ventaja estadísticamente significativa ni generalización hasta demostrarlo.
8. Conservar el título y el Word Fase 06A, usando resaltado amarillo solo para modificaciones nuevas y verificables.

## 5. Pendientes reales

- Ejecutar el piloto y el benchmark con **archivos crudos IEEE-CIS** de la computadora del autor. Los registros de CI solamente prueban coherencia del software con datos sintéticos.
- Ejecutar sensibilidad de umbrales y periodos, controles de presupuestos equivalentes reales, retrasos de etiquetas y análisis de dependencia temporal.
- Reconstruir el **Estudio B con datos posteriores independientes** y medir latencia completa, consumo/costo cloud real y fallos/rollback; no basta con renombrar `prod_lote3_post_retrain.csv`.
- Integrar métricas v2 verificadas en la tesis y discutir si los resultados sustentan o no el objetivo de eficiencia.
- MLflow y mejoras de observabilidad son sugerencias técnicas, no requisitos para afirmar validez.
