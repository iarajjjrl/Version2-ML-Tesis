# Estudio A — Protocolo de evaluación revisado (v2)

**Proyecto:** Proceso automatizado basado en eventos para el reentrenamiento y optimización de recursos en un framework de MLOps. **El título se mantiene sin cambios.**

**Fecha:** 2026-11-09. **Estado:** especificación fijada **antes de ejecutar v2**; todavía no implementada ni validada experimentalmente. **No es una preregistración independiente:** las decisiones se formulan después de inspeccionar v1, y las observaciones de v1 ya son conocidas. Cualquier cambio posterior debe tener versión, motivo y fecha anteriores a la corrida afectada.

**Versión v1 de referencia:** commit `580a4b5ce317ba894f882ed8158d6b547e4d6ef7`, rama `archive/study-a-v1-20261109`. La rama `experiment/study-a-v2-protocol-20261109` parte del mismo commit. Las seis corridas originales se preservan en `runs.zip` con hash registrado en el manifiesto de procedencia; los modelos binarios no se agregan al repositorio. `main` es una rama distinta que ha avanzado después de v1; no se modificará aquí.

## 1. Preguntas e hipótesis a contrastar

**Pregunta principal:** ¿Qué compromiso observable entre calidad predictiva y esfuerzo computacional producen tres políticas de mantenimiento —estática, periódica y disparada por degradación supervisada de F1— cuando se evalúan en los mismos flujos temporales y con la misma búsqueda por intento?

- **H1 (calidad, exploratoria):** la política por eventos puede sostener un F1 agregado mayor que la política estática o ciertas periodicidades; no se presume superioridad estadística.
- **H2 (recursos, exploratoria):** los cambios de rendimiento deben analizarse junto al costo **total** de entrenamiento, intentos fallidos, monitorización y actualizaciones. No se presupone que eventos consuma menos recursos.
- **H3 (momento, secundaria):** condicionado a un presupuesto máximo común de actualizaciones, activar por degradación podría producir una trayectoria distinta de rendimiento respecto de agendas periódicas o aleatorias. No se atribuirá causalidad al momento si los números reales de intentos difieren.

**Resultado principal:** vector por corrida `(F1 agregado sobre ventanas de evaluación, CPU-segundos acumulados del mantenimiento medido)`; se analizará como compromiso multiobjetivo, no como una única cifra de “optimización”. Complementarios: Precision, Recall, FP, FN, PR-AUC cuando haya probabilidades, F1 por ventana, reentrenamientos solicitados/ejecutados, promociones/rechazos, duración de entrenamiento de pared, RSS pico y latencia de despliegue solo si está efectivamente instrumentada.

## 2. Datos y bloqueo temporal

1. Dataset público: IEEE-CIS Fraud Detection (`train_transaction.csv`, `train_identity.csv`, unidos por `TransactionID`), ordenado por `TransactionDT` y luego `TransactionID`. El estudio v1 usó 590 540 filas originales y seleccionó 120 000 de forma sistemática, sin alterar el orden.
2. Para la **comparación principal v2**, conservar inicialmente esos mismos 120 000 registros para aislar el efecto de corregir la metodología; 60 000 primeras filas son historia inicial, 12 000 siguientes son **holdout histórico diagnóstico**, y las 48 000 finales son horizonte futuro, dividido en ocho ventanas de igual **duración relativa**, no igual cantidad de filas. No cambiar a la vez el dataset, las reglas y los splits sin registrar el efecto.
3. El bloque histórico de 60 000 registros se divide internamente y cronológicamente, para cada ajuste, en 70 % ajuste del modelo/preprocesador, 15 % selección de hiperparámetros/familia y 15 % calibración del umbral. Las 12 000 filas de holdout histórico no participan en estas selecciones; sirven de comprobación diagnóstica inicial y su desempeño no se usa para elegir parámetros después de verlo.
4. Para cada reentrenamiento, solo se emplean observaciones y etiquetas conocidas **hasta ese momento**, con nuevas subdivisiones cronológicas internas de 70/15/15. Imputaciones, codificaciones, conteos por tarjeta, estadísticos de importes y otras transformaciones deben ajustarse exclusivamente sobre el bloque de ajuste correspondiente; forman parte del artefacto versionado junto al clasificador. No ajustar estadísticas con ventanas futuras.
5. Las predicciones de cada ventana se almacenan antes de que sus etiquetas afecten una actualización (*test-then-train*). Las métricas de evaluación del periodo no se vuelven a presentar como validación independiente de un candidato seleccionado sobre ese mismo periodo.
6. El escenario **natural** mantiene covariables originales. El escenario **sintético** conserva etiquetas e introduce, desde ventana 3, factores por fila entre 2 y 3 en `TransactionAmt` y `C13` antes de transformar. Denominarlo *perturbación sintética de covariables*, nunca evidencia directa de concept drift.
7. La comparación principal conserva el supuesto de disponibilidad de etiquetas al cerrar cada ventana y lo reconoce como limitación. Un análisis secundario simulará un retraso de **una ventana completa** y volverá a ejecutar las decisiones con información efectivamente disponible. Ninguna regla podrá usar etiquetas aún no reveladas.

## 3. Modelos y presupuesto de selección

- Competidores: XGBoost, LightGBM, CatBoost, Random Forest, HistGradientBoosting, sin aumentar el conjunto según el rendimiento observado en test.
- Búsqueda: Optuna `TPESampler`, con `n_trials=15` por familia y por intento, `n_startup_trials=5`, semillas de búsqueda fijadas. El objetivo de selección será **average precision (AP/área bajo curva precisión–recall no interpolada)** en el bloque de hiperparámetros; **el umbral de clasificación** se elige **después**, usando solo el bloque cronológico de calibración, maximizando F1 en una rejilla de 0.15 a 0.85 con incrementos de 0.01. Es una decisión de diseño, no una cifra que la literatura imponga.
- **Misma búsqueda por intento** para todas las políticas y escenarios: cinco familias, mismos espacios de parámetros, versión de bibliotecas, configuración de threads, criterio de desempate y semilla de la corrida. La política que entrene más veces consumirá mayor presupuesto total: eso es un resultado, no una desigualdad del presupuesto **por intento**.
- Antes del benchmark principal se harán pruebas funcionales con datos pequeños y una corrida diagnóstica de tiempo, **sin utilizar su F1 para ajustar umbrales**. El costo registrado comprenderá toda la búsqueda y preparación de candidato, aunque este sea rechazado.
- El criterio de promoción principal conservará inicialmente `delta_F1 >= 0.01` sobre evaluación independiente de candidato vs vigente y no aceptará empates. Se reportarán también positivos disponibles, FP/FN y márgenes de aceptación. El margen 0.01 es **exploratorio** y tendrá sensibilidad secundaria; no es garantía de mejora verdadera ni un umbral normativo. Una regla adicional de riesgo por falsos negativos requerirá umbral operativo sustentado antes de aplicarse, no se inventará.

## 4. Secuencia de disparo, entrenamiento, promoción y evaluación

- **Congelar una política al comienzo de cada corrida** y asignarle el mismo modelo inicial, transformador y fuente temporal.
- Al cierre de ventana `k` se califica el **modelo que estaba activo durante esa ventana**; solo las etiquetas ya disponibles pueden activar un intento. La decisión no conoce ventanas `k+1...`.
- Si corresponde un intento, el candidato se entrena con historia accesible hasta `k`, sin reutilizar el conjunto externo de evaluación de la próxima ventana.
- Para evitar la superposición entre datos de disparo y promoción, el candidato **permanece en espera** durante `k+1`: los datos de esa siguiente ventana se puntúan con el incumbente para la evaluación prequential y, al disponer de etiquetas, se utilizan como conjunto de aceptación **del candidato**. La aceptación no modifica las predicciones ya registradas de `k+1`. Si se aprueba, el nuevo modelo entra en servicio desde `k+2`. Esto incluye un retraso de una ventana por validación; **se aplica idénticamente** a todas las políticas con mantenimiento.
- Se permitirá **un candidato pendiente a la vez**. Un disparo adicional mientras exista un candidato pendiente se registra como diferido/no ejecutado; no se inventará una ejecución retroactiva. Tanto intentos solicitados como ejecutados deben permanecer en el log. En el caso base, el cooldown de un periodo comienza al realizar el intento; se documentará exactamente su interacción con la espera de promoción.
- Dado el horizonte de ocho ventanas, no se ejecutarán intentos que no puedan ser **promovidos y evaluados en una ventana posterior** dentro del horizonte; registrar disparos descartados por final de horizonte. Esto difiere expresamente de v1 (que omitía principalmente la ventana 8) y puede reducir el número de intentos observables. Comparar ambos protocolos sin fingir equivalencia.
- **No mezclar evaluación y aceptación:** las métricas de una ventana previa a la promoción corresponden al incumbente; no se presentará un F1 del candidato sobre su conjunto de promoción como rendimiento final independiente. El conjunto de evaluación final de un modelo promocionado empieza después de su activación.

## 5. Políticas de la comparación principal

1. **Estática:** no reentrena; sí genera predicciones temporales para evaluar F1, separando costo de evaluación de costo de operación.
2. **Periódicas:** periodos `every={2,3,4}` ventanas, con intentos nominales en ventanas múltiplos del periodo; respetan espera de promoción, etiqueta disponible y límite de horizonte. Registrar solicitudes omitidas para no atribuir actualizaciones no ejecutadas.
3. **Eventos:** umbral F1 `0.50`, persistencia `2` ventanas consecutivas, cooldown `1` ventana, con la regla de candidato pendiente descrita. Se registra si la señal corresponde a desempeño crónicamente bajo; no etiquetar el control como detector de PSI o concept drift.

La comparación primaria **no iguala artificialmente** los números de intentos: frecuencia y costo total son efectos observables de cada estrategia. No habrá conclusión de “optimización” basada solo en F1.

## 6. Análisis de momento de reentrenamiento con presupuesto controlado

- Presupuesto máximo fijo **K=2** intentos de reentrenamiento dentro del horizonte, definido como una restricción de ingeniería para v2 y **después de haber observado los promedios de v1**. Por tanto, este análisis es diagnóstico/exploratorio, no una prueba prospectiva independiente.
- Comparar política de evento con K=2, una agenda periódica con K=2 (e.g. ventanas 3 y 6, sujeta a elegibilidad) y **cinco agendas aleatorias predefinidas** con semillas `[1001,1002,1003,1004,1005]`, muestreadas sin reemplazo de ventanas elegibles cronológicamente.
- El máximo autorizado K no es necesariamente el número **ejecutado**: reportar intentos realmente completados, promociones y costo. Si difieren, no atribuir diferencias únicamente al momento; presentar comparaciones estratificadas descriptivas.
- La aleatoriedad de la agenda no puede utilizar información futura de etiquetas para escoger sus ventanas; la secuencia se fija antes de simular cada flujo.

## 7. Sensibilidad y robustez separadas del resultado principal

1. Umbrales F1 `{0.40,0.45,0.55}` con otros parámetros de evento en valores principales; además comparar cooldown `{0,2}` con umbral 0.50 y persistencia 2. No seleccionar el “mejor” sobre las mismas ventanas para luego declararlo test independiente.
2. Granularidad temporal de **12 ventanas** sobre el mismo horizonte, con frecuencia periódica convertida a **duración relativa**, no trasladando mecánicamente `every=4` como si fuera idéntico.
3. Tres semillas de búsqueda iniciales `{42,43,44}` para comparar con v1; si el presupuesto lo permite, ampliar a `{45,46}` en el análisis de estabilidad. Las semillas sobre la misma trayectoria no son réplicas temporales independientes.
4. Distintas muestras sistemáticas del mismo dataset pueden aportar sensibilidad al muestreo, **pero no independencia respecto del periodo histórico**. Nunca denominar sus resultados validación externa en bancos reales.
5. Retraso de una ventana en disponibilidad de etiquetas: simulación adicional que retrasa tanto disparos supervisados como promoción. El caso principal de etiquetas inmediatas no se presentará como operacionalmente realista.

## 8. Métricas, instrumentación y análisis estadístico

- **Predictivas:** F1 agregado sobre predicciones test-then-train de todas las ventanas, F1 por ventana, Precision, Recall, FP/FN, AP/PR-AUC sobre puntuaciones si se guardan probabilidades, y peor ventana. Mantener claramente separado promedio de F1 por ventana vs F1 sobre transacciones agregadas.
- **Esfuerzo computacional:** tiempo total del proceso de mantenimiento, CPU-segundos acumulados del árbol de procesos, RSS pico (no sumable), tamaño de datos de entrenamiento, cantidad y duración de intentos aprobados/rechazados; gasto adicional de monitoreo y generación de señales si se instrumenta; tiempo de espera/runner/cloud **solo** para B si se registra. Registrar entrenamiento inicial como costo común separado y evaluación del benchmark como costo externo al sistema.
- **Registro obligatorio:** JSON de configuración y metadatos (hash código/datos, CPU/hardware, SO, Python, paquetes), CSV por ventana con política/modelo/etiquetas temporales, puntuaciones por transacción cuando privacidad y espacio lo permitan, tiempos, CPU/RSS, intentos/decisiones/motivos, cambios y artefactos de modelo, hash de cada salida.
- **Diseño estadístico:** reportar diferencias pareadas descriptivas por trayectoria/escenario y sensibilidad; `media±desviación` de semillas representa **variabilidad del entrenamiento**, no distribución de entornos independientes. No aplicar t o Wilcoxon a tres semillas como si fueran flujos independientes. Si se generan más trayectorias temporalmente distintas y predicciones pareadas, definir el remuestreo por bloques y sus supuestos **antes** de calcular intervalos de confianza. Con ocho ventanas no prometer significancia robusta.
- **Interpretación:** una política puede mejorar F1 y consumir más recursos; se reportará como compromiso, no ahorro. Para sostener ahorro operacional total se requiere el costo completo de monitorización, selección, espera y despliegue, no solo la duración del entrenamiento local.

## 9. Estudio B separado

La validación con Docker y GitHub Actions es una **demostración funcional de arquitectura**, no comparación A. Sus reglas no se homologan retrospectivamente a A. Una ampliación posterior deberá puntuar **lotes realmente posteriores y no usados para adaptación** y medir latencia evento→runner→artefacto→verificación→inferencia, con trazabilidad de reintentos y fallos. `MLflow` se tratará como recomendación de instrumentación todavía no implementada, salvo prueba en código.

## 10. Plan de pruebas previo a las corridas largas

- Pruebas del orden temporal: todo `fit` de preprocesador tiene tiempo máximo menor que el de tune, calibration y evaluación; ninguna transformación se ajusta con filas de evaluación futura.
- Pruebas de separación: tune no calibra threshold; calibración no selecciona familia; aceptación del candidato ocurre en `k+1`, y la evaluación independiente del modelo promocionado desde `k+2`.
- Casos de borde: cooldown, persistencia, umbral exactamente igual a 0.50, empates exactos, mejora inferior a 0.01, rechazo, ausencia de clases positivas, candidato pendiente, solicitud al final del horizonte, disponibilidad de etiquetas retrasadas, logs incompletos.
- Verificación de igualdad por corrida: mismo modelo y preprocesador inicial, misma secuencia de ventanas, mismo presupuesto por intento, semilla y configuraciones fijadas; decisiones distintas solo por política.
- Reproducibilidad: scripts reconstruyen agregados desde resultados sin cargar modelos; se prueba que cifras nuevas no sobrescriben ni alteran el archivo v1.

## 11. Condiciones de aceptación del protocolo (antes de resultados)

El estudio v2 se considerará **técnicamente ejecutable** únicamente si (a) las pruebas de integridad y separación pasan, (b) existe una configuración congelada y se genera manifiesto por corrida, (c) se preservan las salidas v1, (d) pueden reconstruirse las métricas desde logs y predicciones, (e) se documentan reentrenamientos fallidos y latencia de aceptación, y (f) los modelos solo ven información disponible en el punto temporal correspondiente. **No existe un criterio prefijado de que eventos “gane”; un resultado negativo seguirá siendo reportable.**

**Próxima acción:** instrumentar las pruebas unitarias y refactorizar el experimento **solo en la rama v2**. No ejecutar aún benchmark completo ni editar las conclusiones cuantitativas de la Fase 06.