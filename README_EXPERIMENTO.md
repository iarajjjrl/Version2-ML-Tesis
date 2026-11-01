# Experimento comparativo — piloto (rama experimento-comparativo)

Esta implementación NO altera los scripts de preparación originales, ni el directorio `data/` existente, ni los artefactos `models/`, ni dispara GitHub Actions. Crea un conjunto independiente bajo `data/experiments/`, excluido de Git.

## 1. Preparar ventanas cronológicas

Ejecutar **desde la raíz** del repositorio, con los CSV originales `train_transaction.csv` y `train_identity.csv` presentes.

```powershell
.\venv\Scripts\python.exe src\experiments\prepare_windows.py --max-rows 120000 --windows 8 --seed 42
```

El script ordena por `TransactionDT`, utiliza el 50% inicial para entrenamiento, el siguiente 10% como holdout histórico de promoción y divide el 40% restante en ocho ventanas de **igual duración temporal** (no igual tamaño). Las métricas de una ventana se calculan **antes** de incorporar sus etiquetas al historial de entrenamiento. El preprocesador se ajusta exclusivamente sobre la primera partición.

Para un *stress test* separado, con modificación determinística de dos características de las ventanas 3 a 8:

```powershell
.\venv\Scripts\python.exe src\experiments\prepare_windows.py --max-rows 120000 --windows 8 --seed 42 --inject-drift-from 3 --output data\experiments\synthetic
```

Esta inyección se denomina **Data Drift sintético**, no Concept Drift. Es un escenario intervenido, no evidencia de un cambio natural de comportamiento del fraude.

## 2. Ejecutar piloto con presupuesto bajo

```powershell
.\venv\Scripts\python.exe src\experiments\run_comparison.py --input data\experiments\chronological --output data\experiments\results-pilot --trials 1 --every 4 --f1-trigger 0.50
```

**Importante:** aunque `--trials 1` sea un smoke test, ejecuta Optuna sobre cinco algoritmos en el entrenamiento base y en los ciclos que se activen, por lo que **puede tardar y consumir memoria/CPU**. El presupuesto final debe ser común y fijado antes de reportar resultados. No ejecutes `auto_tune.py` directamente para este experimento: ese script, con su configuración predeterminada, escribe en `models/`.

`periodic` representa un intervalo fijo de **cuatro ventanas temporales**; si cada ventana representa un periodo equivalente, funciona como una simulación de programación regular. No demuestra la ejecución real de una tarea cron del sistema operativo.

El modo `event` utiliza `F1 < 0.50` como disparador, **no PSI o KS**. `static` nunca reentrena. Las tres estrategias comparten exactamente un baseline inicial, misma preparación y mismo algoritmo de búsqueda. Los archivos de salida de cada corrida son independientes.

## 3. Resultados

- `data/experiments/results-pilot/comparison.csv`: métricas por política y ventana.
- `data/experiments/results-pilot/static.csv`, `periodic.csv`, `event.csv`: resultados separados.
- `data/experiments/results-pilot/configuration.json`: parámetros, supuesto de etiquetas, limitaciones.
- Modelos y CSV de entrenamiento temporales bajo `data/experiments/results-pilot/`.

Las columnas `f1`, `precision`, `recall`, falsos positivos/negativos, número de reentrenamientos, `training_wall_seconds` y `training_process_cpu_seconds` permiten una inspección inicial. **No interpretar `training_process_cpu_seconds` como CPU total de todo el sistema**: no captura de forma exhaustiva todas las actividades de librerías o procesos secundarios. Tampoco se ha cuantificado todavía costo monetario ni recuperación end-to-end de GitHub Actions.

Si deseas repetir una corrida, cambia `--output` (p. ej. `results-pilot-2`). El runner rechaza reutilizar una carpeta de resultados no vacía para proteger la trazabilidad.

## 4. Limitaciones que deben explicarse en la tesis

1. Es un **piloto**, no una validación estadística. Faltan múltiples seeds, diferentes escenarios y análisis de sensibilidad.
2. Se supone que **las etiquetas están disponibles al finalizar cada ventana**. Fraude real tiene retrasos de etiquetado.
3. El AutoML reutilizado hace validación interna estratificada aleatoria; debe evaluarse una validación interna temporal antes de una publicación científica definitiva.
4. La comparación de promoción utiliza un holdout histórico fijo; podría no reflejar la nueva distribución tras drift. Hay que mejorar el guardrail con validación independiente y representativa del periodo de actualización.
5. No ejecuta GitHub Actions ni mide su latencia total, costo de transferencia, espera de runner o tiempo de despliegue. Esos componentes deben instrumentarse aparte si se afirma optimización de recursos *cloud*.
6. No existe todavía una modelización real del cooldown de 180 segundos en las ventanas simuladas.
7. El gasto inicial común de entrenar el baseline se informa por separado. Para comparar costo total, deberá sumarse de modo consistente a las tres políticas.

**No mezclar las cifras de este piloto con la antigua demostración XGBoost → CatBoost sobre el mismo lote reentrenado.** Es un protocolo de evaluación diferente, con holdout temporal y ventanas futuras.

## 5. Relación con la arquitectura

La implementación existente con Docker y GitHub Actions permanece funcional en `main`. Este piloto se enfoca en responder la pregunta científica de **cuándo conviene reentrenar**. Una vez revisada la comparabilidad de resultados, el siguiente trabajo consiste en medir también el pipeline real (monitoreo, búsqueda, entrenamiento, publicación, validación y actualización).
