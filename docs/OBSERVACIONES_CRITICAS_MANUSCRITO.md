# Hallazgos críticos antes de presentar el artículo

Fecha de revisión: 2026-11-01. Esta revisión compara el código actual de la rama experimental con una versión anterior del manuscrito FULL_SEM2_Ian_Ramírez(1).docx y con las observaciones de la rúbrica del curso Seminario de Investigación II (primera nota de octubre de 2026). **No presume que esos borradores sean la versión vigente.**

## A. Afirmaciones que deben corregirse o demostrarse

| Prioridad | Afirmación del borrador anterior | Evaluación frente al código/evidencia | Tratamiento |
|---|---|---|---|
| Bloqueante | El modelo responde a Concept Drift inducido mediante cambios en montos y contadores | prepare_windows.py modifica covariables y conserva etiquetas: demuestra **data/covariate drift sintético**, NO alteración demostrada de P(y|X) | Corregir terminología; reservar Concept Drift para pruebas y definiciones específicas |
| Bloqueante | Una prueba t de Student demuestra superioridad estadística del framework | No se verificó tal prueba ni independencia de sus unidades de análisis. Los tres seeds actuales reutilizan la misma serie temporal | Eliminar/atenuar conclusión hasta ejecutar prueba válida con datos y supuestos verificados |
| Bloqueante | El framework optimiza drásticamente costos reales de infraestructura | Se midieron tiempos locales y algunos recursos; NO costo completo de nube, tarificación, monitoreo, cola y despliegue | Medir y reportar total, o describir como hipótesis/limitación |
| Bloqueante | Hash SHA256 garantiza inmutabilidad y cumplimiento bancario/SBS/PCI-DSS | SHA256 aporta verificación de integridad; por sí solo NO demuestra inmutabilidad, autenticidad ni cumplimiento regulatorio | Limitar afirmación; añadir control de firmas/acceso/auditoría si se desea cobertura mayor |
| Alta | PSI es el disparador de un experimento de tres políticas | monitor.py calcula PSI/KS pero dispara reentrenamiento por F1; el nuevo comparador usa solo F1 | Describir exactamente lo ejecutado o implementar/validar política alternativa |
| Alta | Validación en línea y sin interrupción de un banco real | La ingesta es batch CSV y watcher; sin evidencia de carga de pasarela real ni disponibilidad 100% | Describir como simulación por lotes y prueba de concepto |
| Alta | F1 se recuperó en lote reentrenado y demuestra generalización | La medición sobre lote reutilizado para entrenar puede ser optimista | Usar ventanas temporales FUTURAS no vistas para conclusión predictiva |
| Alta | El estudio es un experimento puro en producción | Diseño actual: estudio aplicado retrospectivo con simulación cronológica, intervenciones de políticas y drift controlado | Describir honestamente el diseño |
| Media | El costo de GitHub Actions son solamente 2:58 | No están verificados todos los tramos ni las tarifas/vigencia de la infraestructura | Separar duración de job, tiempo end-to-end y costo real |
| Media | Todos los algoritmos están optimizados de forma robusta con Optuna | El estudio actual utiliza 3 trials/modelo; es presupuesto pequeño | Reportar como búsqueda limitada y analizar sensibilidad o aumentar presupuesto si justificable |

## B. Observaciones específicas del evaluador del curso SI II

**Introducción:** contextualizar problema con presente, citando afirmaciones generales; delimitar brecha real y contribución a esa brecha.

**Estado del arte:** comparar hallazgos/metodologías/resultados entre autores, no acumular resúmenes; fortalecer fuentes y referencias cruzadas; evitar tabla reiterativa y teoría sin fuente.

**Metodología:** describir estructura de IEEE-CIS; justificar decisiones; comparar políticas bajo condiciones equivalentes; medir costo COMPLETO (monitoreo, búsqueda, cómputo, recuperación de servicio); validar promoción de modelo; sensibilidad al umbral F1, cooldown y retraso de etiquetas; redacción en pasado de lo ejecutado.

**Experimentación:** demostrar replicabilidad con código, versiones, semillas, esquema de datos, métricas, condiciones y resultados verificables.

**Mantenibilidad y observabilidad:** considerar MLflow para tracking y registro de versiones si resulta material a los objetivos; instrumentar CPU/GPU (reportar 'no aplica' sin GPU), RAM, latencias, errores y utilización de infraestructura; registrar evidencia antes/después de propuesta.

**Rúbrica de ética:** citar procedencia y permiso/condiciones del dataset público; nunca utilizar datos bancarios no autorizados.

## C. Puntos que NO deben sacrificarse por la comparación de modelos

La tesis presentada en el borrador es sobre un **framework MLOps basado en eventos orientado a rendimiento y eficiencia computacional**, no solamente un benchmark de F1. La validez de la contribución exige unir dos componentes: la evaluación de políticas y la validación operacional del sistema automatizado. Relacionar MLOps con DevOps, prácticas de versionado, reproducibilidad, monitoreo y gobernanza sin sobreafirmar un nivel de madurez.

## D. Acción inmediata

1. Solicitar los CSV ya existentes study_runs.csv, study_aggregate.csv, study_config.json. No realizar más experimentos antes de auditarlos.
2. Solicitar la versión más actual del manuscrito si difiere de FULL_SEM2_Ian_Ramírez(1).docx.
3. Hacer matriz afirmación -> experimento o fuente -> evidencia -> estado. Etiquetar cada párrafo de resultados como soportado, incompleto o contradictorio.
4. Después de cerrar la auditoría, aprobar solo las pruebas adicionales imprescindibles para el artículo publicable.
