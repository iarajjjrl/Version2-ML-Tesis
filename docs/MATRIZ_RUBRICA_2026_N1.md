# Matriz de respuesta a rúbrica SI II — Nota 1 (entrega de octubre 2026)

Fuente primaria: PDF de dos páginas aportado por el autor: 20262 SI2 Rúbrica Nota 1 IanRamirez(1).pdf. Documento presentado: Tesis_SEM2_Ian_Ramirez(1).docx, título «Proceso automatizado basado en eventos para el reentrenamiento y optimización de recursos en un framework de MLOps».

Calificación del curso: 10/20 indicado en portada de la rúbrica. La nota tiene componentes documento (5.46) y sustentación oral (4.8) bajo el sistema de ponderación de la rúbrica; no inferir una relación aritmética adicional de estos dos números sin la fórmula institucional.

| Área | Observación docente | Estado de la evidencia | Acción verificable |
|---|---|---|---|
| Introducción | Brecha y contribución no claras; afirmaciones generales sin citas; revisión de tiempos verbales | Manuscrito plantea pregunta y objetivo (sección 1), pero novedad frente a políticas existentes requiere contraste | Reescribir 2–3 párrafos de problema, vacío específico y contribución, con 3–5 antecedentes directos verificados |
| Estado del arte | Sintetizar y comparar, no describir autores aisladamente; verificar citas/referencias, coherencia y evitar tabla reiterativa | Referencias y tabla existentes; comparación a menudo narrativa por autor | Matriz comparativa por tipo de disparador, latencia de etiquetas, costo, promoción y limitaciones; verificar DOI y citas |
| Marco teórico | Fundamentar con fuentes válidas | Terminología Data Drift/Concept Drift, PSI, KS, F1; fragmentos de notación matemática requieren QA visual del Word | Verificar fórmulas, diferenciación PSI/KS vs F1 y justificación de métrica |
| Metodología | Dataset, decisiones justificadas, comparación equivalente, costo completo, promoción independiente, sensibilidad F1/cooldown/etiquetas | Comparador cronológico implementado y estudio 18 filas; aún falta validación de parametrización, ruptura de leakage y costo end-to-end | Corregir pipeline y fijar protocolo definitivo con evidencia reproducible |
| Experimentación | Falta replicabilidad; redacción en pasado | La última versión presenta prueba end-to-end original y reserva comparación como futura. Ahora hay comparaciones y config.json verificables | Integrar experimentos nuevos con tables/figuras, parámetros, SHA de commit, límites y análisis de incertidumbre |
| Acta de asesoría | Incorporar MLflow para experimentos, métricas y versiones; CPU/GPU/RAM; tiempo/costo y utilización, asignación/liberación de recursos; diagrama de ciclo; comparar estático y eventos | CSV, Git manifest y profiler parcial, pero sin MLflow ni elasticidad/FinOps demostrado | Definir MLflow tracking e instrumentación; GPU no aplicable si ejecución CPU; presentar mediciones reales o limitar afirmación de asignación dinámica |
| Ética | Documentar fuente/licencia del dataset público | Manuscrito incluye Anexo A con URL Kaggle y describe anonimización | Verificar licencias y condiciones de uso, y conservar evidencia de procedencia |
| Oral | Demostrar dominio y responder sustentado | Nota de rúbrica y observaciones reflejan necesidad de dominio conceptual | Preparar explicación del pipeline, F1/PSI, validación, label delay, crítica de novelty, pruebas de error y costos |

## Correcciones de coherencia manuscrito-código

1. En sección 3.7 el documento presenta «tipo de clasificador como variable categórica del espacio de búsqueda». Código actual realiza **cinco estudios Optuna independientes**, no un solo study con variable categórica. Corregir la redacción o rediseñar el espacio.
2. En sección 3.7 se afirma optimización TPE. El código usa TPESampler, pero **con 3 trials y 10 startup trials por defecto no utiliza el componente TPE adaptativo**. El documento previo describe la formulación teórica, pero NO atribuir su resultado al TPE adaptativo hasta cambiar presupuesto.
3. Metodología de la última entrega describe baseline, lotes nominal/drift y GitHub Actions como avance y reserva experimentos comparativos para la siguiente fase. Esa fase **ya se ejecutó** en seis corridas; actualizar metodología, resultados, discusión y conclusiones con las métricas auditadas.
4. Resultados previos con F1=0.9619 después de reentrenar sobre el mismo lote se reconocen en el texto como recuperación funcional, no generalización. Mantener advertencia y no comparar ese valor con F1 del test cronológico.
5. No igualar un cooldown de 180 segundos del watcher real con una ventana del comparador retrospectivo.
6. Evitar afirmar implementación real de Cron para comparador: la política se activa tras una cantidad fija de ventanas de igual duración.

## Orden recomendado

1. Auditoría de resultados del estudio A (ver docs/ANALISIS_ESTUDIO_A.md).
2. Corrección leakage, búsqueda TPE, métricas y disponibilidad de etiquetas.
3. Validación operativa de trazabilidad y recursos, incluyendo MLflow si se acuerda implementarlo como mecanismo medible.
4. Revisión integral Word: tiempos verbales, tablas, pies, fórmulas, referencias y ética.
5. Validación con asesor y seguimiento de observaciones por evidencia, sin optimizar la investigación en función de nota supuesta.
