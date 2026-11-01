# EXPERIMENTO COMPARATIVO (piloto v2)

OBJETIVO: comparar static / periodic / event con los mismos datos, presupuesto de AutoML y un modelo inicial común.
La comparación es LOCAL. No modifica main, los datos originales ni models/; NO mide costos de GitHub Actions ni Docker.

## 0. Actualizar rama local
En VS Code PowerShell, desde Version2-ML-Tesis:

    git status -sb
    git fetch origin
    git merge --ff-only origin/experimento-comparativo

## 1. Pruebas unitarias (antes de entrenar)

    .\venv\Scripts\python.exe -m unittest discover -s tests -p "test_experiment_policies.py" -v

## 2. Ventanas cronológicas
Ya ejecutaste esta preparación. Si ya existen los archivos en data/experiments/chronological, NO necesitas repetirla.

    .\venv\Scripts\python.exe src\experiments\prepare_windows.py --max-rows 120000 --windows 8 --seed 42

Genera los 60 000 registros iniciales, un holdout histórico de 12 000 y ocho ventanas posteriores de duración equivalente en TransactionDT, sin mezcla temporal. El preprocesador usa solo el entrenamiento inicial. Los datos originales quedan intactos.

## 3. Piloto simple v2 (un trial por algoritmo)

    .\venv\Scripts\python.exe src\experiments\run_comparison.py --input data\experiments\chronological --output data\experiments\results-v2-simple --trials 1 --seed 42 --every 4 --f1-trigger 0.50 --consecutive-low 1 --cooldown-windows 0 --min-improvement 0.01

Se evalúa cada ventana ANTES de incorporar su información. Si corresponde reentrenar, el 70% inicial de esa ventana puede entrar al entrenamiento y el 30% final sirve de validación independiente del candidato. Solo se promueve si supera el F1 del modelo vigente por al menos 0.01 en esa fracción reservada. El rendimiento real posterior se mide en ventanas siguientes, no sobre el lote de adaptación.

Se usa validación interna cronológica en AutoML; Optuna recibe una semilla y el mismo presupuesto de búsquedas para todas las políticas. No se compara con un ser humano.

## 4. Piloto controlado v2: persistencia + cooldown por ventanas

    .\venv\Scripts\python.exe src\experiments\run_comparison.py --input data\experiments\chronological --output data\experiments\results-v2-controlled --trials 1 --seed 42 --every 4 --f1-trigger 0.50 --consecutive-low 2 --cooldown-windows 1 --min-improvement 0.01

Exige dos ventanas consecutivas con F1 bajo y bloquea reentrenamiento en la siguiente ventana tras un intento. Esto NO es el cooldown real de 180 s del watcher: aquí se simulan ventanas temporales, no una ejecución real de reloj.

## 5. Resumir y generar gráficos

    .\venv\Scripts\python.exe src\experiments\summarize.py --results data\experiments\results-v2-simple
    .\venv\Scripts\python.exe src\experiments\summarize.py --results data\experiments\results-v2-controlled

Resultados: comparison.csv (todas las ventanas y decisiones), summary.csv (promedio de F1 por ventana y F1 agregado sobre transacciones), f1_by_window.png, training_time_by_window.png, configuration.json y baseline_metrics.json.

## 6. Interpretación científica de los resultados
1. Ocho ventanas, 120000 registros, la división 50/10/40, F1=0.50, intervalo de cuatro ventanas, margen 0.01, dos ventanas consecutivas y cooldown de una ventana son CONFIGURACIONES EXPERIMENTALES. Deben justificarse mediante comparación o análisis de sensibilidad; no son normas del sector financiero.
2. Un trial por algoritmo comprueba ejecución, NO permite inferencia estadística ni afirmaciones sobre el mejor método.
3. Cada política comparte baseline, preprocesador, dataset y código de AutoML. La única intervención es la política de actualización.
4. Las etiquetas se suponen disponibles al final de cada ventana; el retraso de fraude real es una limitación.
5. CPU de un proceso y segundos de entrenamiento no equivalen al costo completo de nube. Faltan monitoreo completo, uso RAM, cola de GitHub Actions, publicación y despliegue.
6. Un holdout reciente sirve para aprobar el candidato, pero sus etiquetas contribuyeron a la condición de disparo. Solo las ventanas siguientes son una evaluación prospectiva independiente.
7. El preprocesador de features se mantiene congelado después del entrenamiento inicial; este aspecto también limita la capacidad adaptativa.
8. Periodic cada cuatro ventanas equivale a un calendario simulado de periodos fijos, NO prueba la ejecución real de Cron.
9. No mezclar las cifras del piloto antiguo con los nuevos resultados: cambiaron el diseño, el corte temporal, el split interno, la semilla de Optuna y la promoción.
10. Para el informe final: presupuesto fijo de búsqueda mayor, múltiples semillas y escenarios, análisis de sensibilidad, consumo real y pruebas estadísticas con unidades de análisis independientes.

Si la carpeta --output ya existe y no está vacía, usa un nombre NUEVO. Nunca borres el resultado del piloto anterior. No ejecutar auto_tune.py en forma aislada para el experimento, porque su configuración predeterminada escribe en models/.

SIGUIENTE PASO: pruebas unitarias, comparación simple, resumen, comparación controlada, resumen y decisión explícita de qué políticas/umbrales llevar a evaluación definitiva.


## 7. Revisión del cierre temporal (v3)

Un experimento que evalúa solo ocho ventanas NO debería sumar al costo un
reentrenamiento posterior a la ventana 8, porque ninguna ventana posterior
permite medir su aporte. Desde el commit de revisión de horizonte, el runner
omite esa actualización final por defecto y registra motivo
"end_of_horizon_not_retrained".

Si se pretende medir un sistema operativo que CONTINUARÁ recibiendo más
ventanas, puede añadirse --train-after-final-window, pero ese tiempo debe
interpretarse como mantenimiento para el horizonte siguiente, no como costo
de servir las ocho ventanas ya evaluadas.

**No mezclar los costos de results-v2-* con results-v3-*.** Para el análisis
comparativo de horizonte finito ejecuta nuevas corridas:

    .\venv\Scripts\python.exe src\experiments\run_comparison.py --input data\experiments\chronological --output data\experiments\results-v3-simple --trials 1 --seed 42 --every 4 --f1-trigger 0.50 --consecutive-low 1 --cooldown-windows 0 --min-improvement 0.01

    .\venv\Scripts\python.exe src\experiments\run_comparison.py --input data\experiments\chronological --output data\experiments\results-v3-controlled --trials 1 --seed 42 --every 4 --f1-trigger 0.50 --consecutive-low 2 --cooldown-windows 1 --min-improvement 0.01

    .\venv\Scripts\python.exe src\experiments\summarize.py --results data\experiments\results-v3-simple
    .\venv\Scripts\python.exe src\experiments\summarize.py --results data\experiments\results-v3-controlled

## 8. Sensibilidad del costo por errores de fraude

Permite ensayar razones hipotéticas costo(FN)/costo(FP), por ejemplo 1, 2,
5, 10, 20. Son UNIDADES RELATIVAS, no tarifas bancarias ni soles.

    .\venv\Scripts\python.exe src\experiments\error_cost_sensitivity.py --results data\experiments\results-v3-simple
    .\venv\Scripts\python.exe src\experiments\error_cost_sensitivity.py --results data\experiments\results-v3-controlled

Usar error_cost_sensitivity.csv solo para explorar el intercambio entre
detección de fraude y falsas alarmas; no afirmar ahorro económico real.
