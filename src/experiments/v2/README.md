# Estudio A v2 — integración probada con datos sintéticos (12/11/2026)

**Rama:** `experiment/study-a-v2-integracion-20261112`. **No modificar main ni v1.**

## Estado real

- `raw_windows.py`: crea datos crudos y cronológicos sin preprocesamiento anticipado.
- `temporal.py`: separa historia previa 70/15/15 para selección de hiperparámetros y ajuste inicial del preprocesador.
- `controller.py`: política estática, periódica y por eventos; solicita en k, revisa/promueve en k+1, usa el modelo nuevo desde k+2.
- `trainer.py`: Optuna compara **las cinco familias** (XGBoost, LightGBM, CatBoost, RandomForest, HistGradientBoosting). Usa solamente `tune` de historia previa para elegir parámetros; al actualizar, ajusta el modelo y el preprocesador finales con historia + el **70% inicial del lote causal k**, y calibra el umbral con el **30% final no utilizado para fit**. Para el baseline inicial usa `calibrate` del split 70/15/15.
- `run_comparison_v2.py`: ejecuta las tres políticas de forma local sobre las mismas ventanas e idéntico baseline; persiste métricas, rechazos, tiempo y artefactos experimentales.
- `prepare_epochs.py`: genera escenarios natural/sintético a partir de épocas cronológicas disjuntas del dataset original.
- `run_study_v2.py`: consolida épocas × escenarios × semillas × periodicidades, sin afirmaciones de independencia estadística ni de costo cloud.

**CI:** 20 pruebas unitarias y de integración aprobadas con datos sintéticos. **No hay aún resultados IEEE-CIS v2**; falta ejecutar el benchmark real y verificar la calidad y los costos.

## Lectura y pasos

Ver `docs/FASE_08_ESTADO_Y_COMANDOS.md` para comandos PowerShell, procedimientos de no sobrescritura, explicación del lote 3 del Estudio B y criterios de aceptación.

```powershell
.\venv\Scripts\python.exe -m pytest -q tests/test_study_a_v2.py tests/test_study_a_v2_integration.py
.\venv\Scripts\python.exe -m src.experiments.v2.prepare_epochs --transactions train_transaction.csv --identity train_identity.csv --output data/experiments/raw-v2-epochs --epochs 3 --max-rows 120000 --windows 8
.\venv\Scripts\python.exe -m src.experiments.v2.run_comparison_v2 --input data/experiments/raw-v2-epochs/epoch_01/natural --output data/experiments/results-v2-integrated-smoke-42 --seed 42 --trials 1 --every 4
```

Los CSV originales IEEE-CIS se conservan localmente, no se agregan al repositorio. Es preciso comprobar su procedencia y el orden de `TransactionDT` antes de cualquier corrida científica.

**Ninguna prueba sintética sustituye resultados reales, ni el número de semillas implica independencia temporal.** Un cap de intentos no demuestra presupuesto verdaderamente igualado. La demostración funcional del Estudio B sigue siendo un experimento separado.
