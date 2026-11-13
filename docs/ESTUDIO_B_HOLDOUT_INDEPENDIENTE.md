# Estudio B: evaluación temporal independiente sin modificar producción

## Alcance de esta rama

Rama: `experiment/study-b-holdout-independiente-20261112`, derivada de la rama validada del Estudio A.

**No reemplaza** el resultado histórico de GitHub Actions Run 36989119260, ni cambia `main`, `src/prepare_data.py`, el watcher o los modelos activos. El estudio histórico probó hot-swap; su F1=0.9619 se calculó sobre el lote causal adaptado, no es generalización.

Este protocolo agrega una **prueba de generalización independiente sobre otro entrenamiento local**. Al usar nuevos splits cronológicos, los modelos originales B *no* pueden presentarse directamente como modelos entrenados con este protocolo: su split antiguo era aleatorio y se desconoce independencia temporal. El evaluador externo puede puntuar modelos antiguos pero no probar por sí mismo independencia del entrenamiento.

## Qué se reserva

Se submuestrean sistemáticamente hasta 120,000 filas de IEEE-CIS, manteniendo cronología. Se dividen sin solapamiento en:

- Baseline inicial: primeras 50%; único bloque usado para ajustar el `FraudFeaturePreprocessor`.
- Lote nominal: 10% siguiente.
- Lote causal/adaptación: 20% siguiente; se aplica *shift* sintético multiplicativo a `TransactionAmt` y `C13`, sin cambiar etiquetas.
- Lote futuro sellado: 20% final. Dos versiones, natural y shift sintético, **comparten las mismas transacciones** y no son muestras independientes entre sí.

Se guarda un `manifest.json` con hashes SHA256 de cada archivo y hashes de los identificadores originales, recuentos, positivos y orden temporal. El futuro se prepara **antes de entrenar**. No se utiliza para selección de familia, hiperparámetros, threshold ni promoción.

## Uso en Windows PowerShell

En el repositorio local, después de guardar cambios anteriores:

```powershell
git status -sb
git fetch origin
git switch experiment/study-b-holdout-independiente-20261112
git pull --ff-only origin experiment/study-b-holdout-independiente-20261112

python -m pytest -q tests/test_study_b_independent.py

python -m src.experiments.study_b_independent prepare `
  --transactions train_transaction.csv `
  --identity train_identity.csv `
  --output data/experiments/study-b-temporal-holdout-20261112 `
  --max-rows 120000 --seed 42
```

Este primer comando NO entrena modelos ni modifica archivos `models/`.

### Evaluación local con AutoML B

Para ejecutar dos entrenamientos en carpetas separadas y evaluar con las mismas transacciones futuras:

```powershell
python -m src.experiments.study_b_independent local-run `
  --prepared data/experiments/study-b-temporal-holdout-20261112 `
  --trials 1 --seed 42
```

Guarda:

- `local_validation/model_before/` y `local_validation/model_after/`: artefactos distintos, modelo y manifiesto por versión.
- `local_validation/comparison_future.json`: F1, precision, recall, TP/FP/FN/TN antes y después en futuro natural y alterado.
- `local_validation/provenance.json`: evidencias de entrada, configuraciones y límites de la prueba.

El script trabaja con una **COPIA de la base** en `local_validation/working_training.csv`, ya que el `src/auto_tune.py` de B concatena el lote causal sobre el archivo de base recibido. El archivo `baseline_train.csv` original permanece intacto.

**Advertencia metodológica:** para permitir comparar los dos modelos incluso si alguno falla una restricción de calidad, este estudio local usa `umbral_guardrail=0.0`. Por eso NO demuestra que el modelo final hubiera sido aprobado o promovido por el guardrail productivo `F1>=0.50`. Mantiene, además, la validación interna estratificada histórica de `auto_tune.py`; el criterio temporal aplica a los bloques externos, no a su validación interna. Se deben mostrar también resultados negativos. Optuna con 1 trial es solo prueba piloto; aumentar a 3 si hay recursos, describiéndolo como presupuesto reducido.

### Evaluar manualmente dos modelos de confianza

```powershell
python -m src.experiments.study_b_independent evaluate `
  --prepared data/experiments/study-b-temporal-holdout-20261112 `
  --before data/experiments/study-b-temporal-holdout-20261112/local_validation/model_before/MODELO_ANTERIOR.pkl `
  --after data/experiments/study-b-temporal-holdout-20261112/local_validation/model_after/MODELO_NUEVO.pkl `
  --output data/experiments/study-b-temporal-holdout-20261112/comparacion_extra.json
```

Sustituir las rutas de ejemplo por las rutas reales leídas desde `active_model.json`. **Cargar archivos `.pkl` de terceros puede ejecutar código: utilizar exclusivamente modelos de confianza.**

## Lo que este protocolo SÍ y NO prueba

- **SÍ:** que, con splits cronológicos nuevos, la muestra de prueba final se preparó antes de entrenar los modelos y permaneció fuera de la selección; y que el cambio de modelo puede compararse sobre las mismas transacciones futuras.
- **NO:** que la mejora sea positiva; que el modelo se haya promocionado realmente por guardrail; que GitHub Actions/Docker ejecutaron el nuevo split, ni que exista ahorro cloud.
- **NO:** que `future_natural` y `future_shifted` sean dos muestras independientes.
- **NO:** que el F1 del lote post-hot-swap histórico de octubre haya sido medido sobre datos nuevos.

Para una validación **operacional integrada** adicional, habría que usar los artefactos de este protocolo con una instancia aislada del watcher y GitHub Actions, fijar la aprobación del candidato y capturar el flujo alerta→workflow→SHA→modelo activo. No hacerlo en `main` sin aislar el ambiente, porque el monitor y el workflow B pueden hacer commits y push.

Los archivos de los lotes futuros **NUNCA deben copiarse a `data/ingesta_viva`**, pues el watcher podría desencadenar otro reentrenamiento. Usar `evaluate` es evaluación de solo lectura sin despachos.

## Relevancia de las observaciones de Rosario

- El modelo **estático del Estudio A** es la línea base comparativa, y la periodicidad es otra referencia. Este nuevo protocolo B es una validación de predicción sobre datos independientes de *entrenamiento local nuevo*, no reemplaza las baselines.
- La reevaluación sobre el lote que disparó la alerta no sirve para estimar generalización.
- El costo de reentrenar y la calidad se presentan por separado; los segundos de GitHub Actions disponibles siguen siendo tiempo parcial de una ejecución histórica.
- No extrapolar ahorro cloud, significancia estadística, validez bancaria, ni óptimo global de los cinco modelos.
