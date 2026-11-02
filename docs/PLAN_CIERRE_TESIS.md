# Plan de cierre de tesis MLOps — Ingeniería de Sistemas

**Fecha de planificación:** 1 de noviembre de 2026  
**Estado:** plan de trabajo; NO constituye aprobación académica ni resultados definitivos.  
**Rama:** experimento-comparativo; mantener main intacta hasta validación final.  
**Objetivo:** consolidar un artículo científico publicable de la Universidad de Lima y preparar evidencia para la sustentación.

## 0. Ruta universitaria y control de versiones

**Ruta sugerida:** tesis, submodalidad artículo publicable. La carrera también ofrece artículo publicado. Esta decisión debe ser ratificada con asesor/Unidad de Grados y Títulos, pues el título profesional requiere bachiller, aprobación académica y trámites oficiales.

Enlace informativo de carrera: https://www.ulima.edu.pe/pregrado/ingenieria-de-sistemas  
Rúbrica oficial 2026: https://www.ulima.edu.pe/sites/default/files/2026-01/gyt_cis_2026_rubrica_art_publicable.pdf  
Contacto de Ingeniería de Sistemas: gytsistemas@ulima.edu.pe

**Por confirmar con universidad:** elegibilidad actual del alumno, vigencia de procedimiento, reconocimiento del artículo desarrollado en Seminario de Investigación II, asesor/especialista, documentos, plazos y sustentación. No extrapolar automáticamente reglamentos de promociones anteriores.

## 0.1. Auditoría del manuscrito y rúbrica del curso (BLOQUEANTE)

Se revisó una versión anterior del artículo FULL_SEM2_Ian_Ramírez(1).docx y el reporte del evaluador de Seminario de Investigación II (octubre 2026). Hay afirmaciones que requieren rectificación o evidencia antes de presentar: significancia t de Student sin prueba corroborada, reducción de costos cloud sin medición completa, Concept Drift tratado como alteración de covariables y cumplimiento bancario atribuido solo a SHA256. **No reutilizar automáticamente sus conclusiones.**

Plan detallado: docs/OBSERVACIONES_CRITICAS_MANUSCRITO.md.

Adicionalmente, conservar correspondencia entre el objetivo original sobre MLOps, calidad y mantenibilidad y el comparador de políticas. No convertir silenciosamente la tesis en un simple benchmark de clasificadores; incluir validación operacional y contraste MLOps/DevOps.

## 1. Qué afirmamos / qué todavía no demostramos

**Demostrado técnicamente por ejecución local de scripts:**
- Creación de ocho ventanas cronológicas IEEE-CIS con 120 000 registros sistemáticamente seleccionados, sin inyección en escenario natural.
- Ejecución de comparaciones estática, periódica y basada en eventos, con resultados piloto V3 verificables.
- Script de estudio completado para escenarios natural/sintético, semillas 42/43/44, tres trials de Optuna por algoritmo. La bitácora sola no aporta las métricas agregadas: se requieren archivos CSV.
- Pruebas unitarias del controlador y monitor de recursos: 9/9 pasaron.

**No demostrado:**
- Ventaja estadística general, costo real de nube, rendimiento bancario en tiempo real, ausencia de todas las fugas de validación, disponibilidad instantánea de etiquetas o generalización a otros datasets.
- Que PSI/KS disparen el experimento de comparación: en monitor.py son descriptivos, mientras que run_comparison.py usa F1.

## 2. Gate A — Auditar evidencia del estudio existente (PRIORIDAD 1)

**Entradas locales que debe facilitar el autor:** data/experiments/study-a/study_runs.csv, study_aggregate.csv, study_config.json, y —solo si vemos anomalías— comparison.csv y summary.csv de cada corrida. No pedir tokens ni credenciales.

**Análisis:**
1. Verificar seis combinaciones de escenario/semilla, tres políticas por combinación, cobertura de ocho ventanas y comparabilidad de columnas.
2. Comparar F1 agregado y F1 por ventana, Recall, FN, FP y gastos de entrenamiento; inspeccionar semillas que cambian conclusiones.
3. Describir media y dispersión entre seeds como sensibilidad a la optimización, NO como seis muestras independientes del fenómeno de fraude.
4. Revisar los recursos (RSS/CPU) y la política con menor error para distintos ratios FN/FP.
5. Comprobar drift sintético contra natural; no presentarlo como concepto real.
6. Congelar tablas de resultados con SHA de commit y versiones de paquetes.

**Criterio de cierre:** resumen técnico con todas las cifras trazables, diferencias y limitaciones; decidir solo después cuáles pruebas complementarias tienen valor.

## 3. Gate B — Robustecer validez del experimento (PRIORIDAD 2)

- **Disponibilidad temporal de etiquetas:** separar predicción, monitoreo sin etiqueta (PSI/KS) y señal supervisada (F1) cuando existan etiquetas confirmadas. Probar al menos un supuesto de retraso y documentarlo.
- **Seleccionar y evaluar modelos sin fugas:** evitar utilizar el mismo conjunto interno para optimización de hiperparámetros y calibración del umbral sin validación adicional; validar en bloques posteriores. Mantener comparación equilibrada entre políticas.
- **Políticas relevantes:** estática, reentrenamiento periódico fijo y evento controlado; si se agrega detector PSI/KS o ADWIN, debe probarse como variante con reglas y umbrales previamente definidos.
- **Datos:** conservar protocolo natural/sintético; evaluar representatividad de la submuestra de 120 000 y, si recursos lo permiten, repetir sobre más registros del dataset completo.
- **Análisis de sensibilidad:** umbrales F1, persistencia, cooldown, periodicidad, mejora de promoción y presupuesto Optuna. No escoger hiperparámetros del controlador solo porque favorezcan evento.
- **Estimación:** métricas por ventana y agregadas; PR-AUC y ROC-AUC si se dispone de probabilidades; documentar rangos e incertidumbre cuando exista replicación defendible.
- **Pruebas:** añadir controles automatizados de no filtración temporal, etiquetas, compatibilidad de columnas y escenarios reproducibles.

**Criterio de cierre:** protocolo congelado que permita a otra persona reproducir datos, split, condiciones, resultados y conclusiones.

## 4. Gate C — Mantenibilidad y MLOps operativo (PRIORIDAD 3)

Inspección/control de src/folder_watcher.py, src/monitor.py, src/stream_sintetico.py, .github/workflows/retrain.yml, Dockerfile y manifests.

Pruebas controladas:
1. Evaluación normal; registrar latencia y versión activa.
2. Drift con desencadenamiento supervisado/no supervisado según etiquetas; registrar tiempo detección -> solicitud -> job -> nuevo modelo -> hot-swap.
3. Candidato sin mejora: rechazado sin reemplazar modelo vigente.
4. Artefacto corrupto o hash inválido: no publicarlo; conservar versión anterior.
5. Fallo de Actions/red temporal: manejo seguro y trazabilidad, sin pérdidas silenciosas.
6. Reinicio del watcher e ingesta de lotes repetidos: idempotencia.
7. Control de secretos: nunca commitear .env ni tokens, no compartir logs con credenciales, revisar que flujos git no expongan URL autenticada.

**Métricas:** tasa de éxito de actualizaciones, tiempo de detección, tiempo hasta recuperación, errores/rollback, reproducibilidad de manifiesto/artefactos, costo end-to-end medible. Distinguir simulación local de GitHub Actions real.

**Criterio de cierre:** matriz de pruebas con observación/log, resultado esperado y real, y limitaciones.

## 5. Gate D — Estado del arte, aporte y manuscrito (PRIORIDAD 4)

- Crear matriz de >=20 investigaciones pertinentes verificadas, idealmente peer reviewed y recientes. Columnas: cita, año, objetivo, escenario, política de drift/retraining, métricas, resultados, limitación, diferencia respecto al framework. Las 20 obras corresponden al descriptor de mayor nivel de la rúbrica 2026, no a una condición mínima universal de toda tesis.
- Diferenciación propuesta a validar: política de mantenimiento controlada + validación de candidatos + trazabilidad + comparación de calidad y costo de operación. No afirmar que el retraining event-driven es novedoso por sí mismo.
- Ajustar pregunta/objetivos a lo que realmente mide el experimento.
- Artículo: título/resumen; introducción con problema y contribución; estado del arte; metodología; implementación; resultados; discusión que compare bibliografía y resultados inesperados; conclusiones, limitaciones y recomendaciones; referencias y anexos replicables.
- Preparar diagrama reproducible, tablas, gráficos legibles, evidencia de origen de datos, commits, licencias/permisos y sección de ética.
- Revisión con profesor especialista; levantar observaciones registradas, comprobar estructura y envío según modalidad oficial.

**Criterio de cierre:** artículo completo editable + PDF de revisión + matriz de observaciones atendidas + anexos técnicos.

## 6. Gate E — Presentación y titulación

- Confirmar condición de bachiller y requisitos con Unidad de Grados y Títulos.
- Asegurar visto bueno del asesor y especialista y documentos oficiales actualizados.
- Preparar guion de sustentación y preguntas difíciles: causalidad, definición de drift, retraso de etiquetas, prueba temporal, baseline, F1 vs costo, por qué cinco algoritmos, reproducibilidad, fallos/rollback y originalidad.
- No confundir aprobación del artículo, aceptación editorial y obtención del título: son procesos diferentes.

## Orden de ejecución inmediato

1. Auditar study_runs.csv, study_aggregate.csv y study_config.json existentes; NO rehacer el estudio.
2. Registrar hallazgos y riesgos; escoger una sola lista de experimentos imprescindibles.
3. Reforzar evaluación temporal, disponibilidad de etiquetas y validez de promoción.
4. Validación operacional Docker/Actions.
5. Consolidar artículo y respaldarlo contra rúbrica 2026.
6. Revisión con asesor y trámite oficial.

**Regla de control:** no enviar artículos a revistas ni afirmar ahorro de recursos, despliegue bancario o novedad inédita sin la evidencia correspondiente. No tocar main sin aprobar integraciones.
