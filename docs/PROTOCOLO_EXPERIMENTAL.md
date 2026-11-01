# Protocolo de evaluación comparativa de mantenimiento MLOps

## Pregunta

¿En qué condiciones una política basada en eventos mejora el equilibrio entre calidad predictiva y costo de mantenimiento de modelos de fraude, en comparación con una política sin actualizaciones y una política periódica?

## Objeto de estudio y variables

- Caso: IEEE-CIS Fraud Detection. Características seleccionadas y preprocesador fijo del prototipo.
- Independiente: política de actualización {estática, periódica cada cuatro ventanas, eventos con dos ventanas consecutivas F1 < 0.50 y cooldown una ventana}.
- Dependientes de calidad: F1 agrupado sobre todas las transacciones, media F1 por ventana, Recall, falsos negativos y falsos positivos.
- Dependientes de recursos: número de entrenamientos, promociones y segundos de entrenamiento. Con psutil opcional: CPU de proceso y descendientes y memoria RSS máxima aproximada.
- Controladas: mismo conjunto de datos, orden de ventanas, modelo inicial, cinco clasificadores candidatos, presupuesto y semilla de Optuna por repetición, regla de promoción y entorno informático.

## Hipótesis a contrastar, no resultados asumidos

H1. La política basada en eventos controlada ofrece un mejor equilibrio calidad/costo que la política periódica en al menos alguno de los escenarios evaluados.

H2. La política controlada puede evitar entrenamientos innecesarios frente a un evento simple sin histéresis, con una pérdida de calidad cuantificable. Comparar en un análisis separado; no seleccionar umbrales a posteriori para maximizar F1.

H3. La preferencia entre políticas cambia cuando aumenta el costo relativo de falsos negativos frente a falsos positivos.

## Diseño cronológico

1. Ordenar IEEE-CIS por TransactionDT.
2. Seleccionar una muestra sistemática reproducible que preserve el orden, o emplear la totalidad con capacidad computacional suficiente. Explicar el sesgo potencial del submuestreo.
3. Ajustar el preprocesador exclusivamente con los datos de entrenamiento inicial (50% temporal). Reservar 10% temporal para evaluar baseline; dividir el 40% temporal posterior en ocho intervalos de igual duración.
4. Compartir un modelo inicial idéntico para las tres políticas dentro de cada corrida.
5. Ejecutar evaluación por ventanas: predecir primero, calcular las métricas y decidir la actualización cuando las etiquetas estén disponibles. Para entrenar un candidato, separar la última ventana observada en 70% de adaptación y 30% de promoción; no usar ese 30% en ajuste del candidato.
6. Evaluar el rendimiento predictivo del nuevo modelo solamente sobre ventanas posteriores a su promoción. No contar como costo del horizonte cerrado una actualización tras la última ventana, excepto si se declara horizonte operativo continuado.
7. Registrar todas las métricas y reglas de disparo con trazabilidad por escenario, semilla y commit.

## Escenarios bloqueados antes de las ejecuciones

A. Natural: sin inyección artificial, cambios presentes en la secuencia original.

B. Sintético: introducir variaciones deterministas en TransactionAmt y C13 a partir de la ventana 3, manteniendo la etiqueta y la semilla de inyección fija en 42. Esto es un stress test de cambio de covariables, no una prueba de Concept Drift real. No afirmar fraude adicional si no se alteraron etiquetas.

Se repite cada escenario con semillas de AutoML 42, 43 y 44 para estudiar sensibilidad a la búsqueda. **Estas repeticiones comparten el mismo conjunto temporal** y NO constituyen muestras independientes del proceso económico; reportar rangos/desviaciones descriptivos sin usar pruebas de significancia que supongan independencia.

Para ampliar robustez posteriormente: prueba sobre la totalidad de registros y varios bloques temporales de evaluación respetando orden y embargo si las features temporales lo requieren.

## Presupuesto computacional

- Fase inicial: tres trials de Optuna POR modelo, cinco modelos candidatos, cada vez que corresponde reentrenar. Mismo presupuesto por estrategia y repetición.
- La carga de entrenamiento depende del número de actualizaciones y ese costo debe computarse; no ajustar presupuestos de búsqueda entre estrategias.
- Estimar memoria, tiempos y CPU si se activa --profile-resources. La medición RSS/CPU es local y muestreada; su validez NO se extiende automáticamente a GitHub Actions.
- Congelar la configuración del estudio en study_config.json, almacenando también versión de Python, plataforma y SHA del commit de Git.

## Regla de promoción

El candidato solo se publica si mejora por al menos 0.01 puntos absolutos de F1 al incumbente sobre el segmento reservado de la ventana. Un empate no promueve. Es un guardrail experimental sujeto a sensibilidad; NO un estándar bancario.

La política desencadenante no modifica las features del clasificador ni selecciona un algoritmo arbitrariamente: las tres reutilizan el mismo AutoML y presupuesto en cualquier intento.

## Limitaciones que NO pueden omitirse

- Etiquetas de fraude no siempre están disponibles en tiempo real; aquí se suponen al cierre de la ventana.
- Evaluar F1 en toda la ventana determina disparo y usa información de etiquetas, incluso del segmento reservado para promoción. Esa sección no entra en el entrenamiento, pero la decisión de abrir un ciclo ya depende de las etiquetas de la ventana.
- Una única secuencia temporal, ocho ventanas y tres semillas de AutoML NO permiten atribuir significación estadística general.
- El preprocesador permanece congelado; su drift interno no se corrige automáticamente.
- No se instrumenta latencia end-to-end de GitHub Actions, cola, transferencia, seguridad, despliegue o gasto de nube; esto requiere una validación operacional independiente.
- Los costos relativos de FP y FN son hipótesis de sensibilidad, no cifras reales del banco.

## Entregables de evidencia

- study_config.json y manifest.json de preparación.
- comparison.csv y summary.csv por corrida.
- study_runs.csv y study_aggregate.csv por escenario y semilla.
- Gráficos F1 y tiempo acumulado por ventana.
- Matriz de error operativo FN/FP y sensibilidad de costo relativo.
- Documentación del pipeline real GitHub Actions/Docker y telemetría end-to-end, en etapa separada.

## Regla editorial

No escoger la configuración después de inspeccionar los resultados y llamarla de antemano óptima. Diferenciar hipótesis, simulación, resultados del prototipo y validación operacional. Es aceptable que eventos tenga mayor costo computacional si su beneficio predictivo lo justifica; documentar el tradeoff sin forzar la conclusión.
