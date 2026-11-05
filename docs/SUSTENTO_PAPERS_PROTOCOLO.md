# Sustento científico del protocolo MLOps — para discusión docente (5 nov. 2026)

**Propósito:** respaldar las decisiones de evaluación ANTES de modificar otra vez la arquitectura o reentrenar. Se complementa con el documento entregable de seis páginas y con `docs/PROTOCOLO_EXPERIMENTAL.md`.

## Matriz de fundamentos y decisiones

| Decisión | Fuente verificable | Respaldado por la fuente | Lo que sigue siendo una decisión nuestra |
|---|---|---|---|
| Comparar estático, periódico, eventos | Mahadevan y Mathioudakis (2024), [CARA](https://doi.org/10.1016/j.knosys.2024.111610); Regol et al. (2025), [ICML](https://proceedings.mlr.press/v267/regol25a.html) | La actualización presenta un compromiso entre degradación y costo de reentrenar | Frecuencia de 4 ventanas, disparador y costo relativo exactos |
| Evaluación en orden temporal | Gama et al. (2014), [ACM](https://doi.org/10.1145/2523813); [Scikit-learn time series](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit) | La validación temporal debe evitar entrenar sobre información futura | Ocho ventanas y división 50/10/40 |
| Justificar granularidad de ventanas | Bifet y Gavaldà (2007), [ADWIN](https://doi.org/10.1137/1.9781611972771.42) | La escala temporal afecta detectar cambios; existen ventanas adaptativas | No existe en esta fuente una recomendación universal de exactamente 8 |
| Fraude con etiqueta tardía | Dal Pozzolo et al. (2015), [IEEE](https://doi.org/10.1109/IJCNN.2015.7280527) y Dal Pozzolo et al. (2017), [TNNLS](https://doi.org/10.1109/TNNLS.2017.2736643) | La evaluación realista debe distinguir feedback disponible y etiquetas retrasadas | Retraso concreto en ventanas aún por especificar y probar |
| Métricas en desbalance | Saito y Rehmsmeier (2015), [PLOS ONE](https://doi.org/10.1371/journal.pone.0118432) | Curvas PR son informativas en clasificación desbalanceada | Límite F1<0.50 no es un estándar |
| Arquitectura MLOps | Kreuzberger et al. (2023), [IEEE Access](https://doi.org/10.1109/ACCESS.2023.3262138); [Google Cloud MLOps](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | Monitoreo, validación de datos/modelo, metadata, CT/CI/CD | Implementación particular GitHub Actions y Docker |
| Calidad y aceptación | [ISO/IEC 25010:2023](https://www.iso.org/standard/78176.html) y [25023:2016](https://www.iso.org/standard/35747.html) | Características de calidad y medición de productos software | La norma NO fija límites universales en ms, % o segundos de cooldown |
| Validación sin fuga | [Scikit-learn 1.7, common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html); [tuning threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) | Aprender preprocesado solo de entrenamiento, no reutilizar datos de ajuste para medir generalización | Split/calibración interno aún requieren refactor y pruebas |
| Optimización de modelos | Bergstra et al. (2011), [NeurIPS](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization); [Optuna TPESampler](https://optuna.readthedocs.io/en/latest/reference/samplers/generated/optuna.samplers.TPESampler.html) | Búsqueda de hiperparámetros puede mejorar modelos; predeterminado `n_startup_trials=10` | Los tres trials actuales caen en muestreo inicial, no TPE adaptativo |
| Seguimiento de experimentos | [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) | Permite registrar runs, parámetros, métricas, versiones y artefactos | Integración real con pipeline aún pendiente |
| Dataset y tiempo relativo | [Kaggle IEEE-CIS Data Description](https://www.kaggle.com/c/ieee-fraud-detection/data) | `TransactionDT` es timedelta de referencia, NO una fecha calendario | Hay que calcular período observado con `manifest.json` |

## Tres dimensiones de métricas

1. Calidad predictiva: F1, Precision, Recall, PR-AUC y matriz de confusión (FP/FN).
2. Rendimiento operativo: latencia inferencia p50/p95, demora detección, tiempo disparo→job→artefacto→modelo disponible y tasa de promoción/rollback segura.
3. Recursos: tiempo de AutoML, CPU, RAM máxima, trabajos fallidos, monitoreo, transferencia, espera de runner y costo cloud donde sea medible.

**Importante:** `study_aggregate.csv` demuestra en el estudio A que el método eventos consigue F1 agregado medio mayor que el periódico en ambos escenarios, pero también utiliza más tiempo de entrenamiento. Las tres semillas usan una serie temporal compartida y no garantizan significación estadística.

## Verificación de duración real de ocho ventanas (sin entrenar)

Desde `experimento-comparativo`:

```powershell
git fetch origin
git merge --ff-only origin/experimento-comparativo
.\venv\Scripts\python.exe src\experiments\report_window_horizon.py --manifest data\experiments\study-a\input\natural\manifest.json
```

Registrar en una tabla: duración observada en días, casos y fraudes por ventana, y periodo total. Para la defensa NO decir un año ni ocho meses sin corroborar.

## Criterios de aceptación a instrumentar

- Baseline estático: no actualizar modelo; medir sus resultados con la misma secuencia.
- Predicción antes de reentrenamiento: usar etiquetas cuando estén realmente disponibles.
- Guardrail: ningún empate o degradación del candidato reemplaza al modelo activo.
- Artefacto corrupto/inesperado: rechazar actualización y mantener servicio.
- Trazabilidad: relacionar datos, trigger, búsqueda, validación, manifiesto y modelo activado.
- Operación: medir duración completa de detección hasta disponibilidad del nuevo modelo, sin confundir simulador local con GitHub Actions real.

**Estado:** principios científicamente sustentados; parámetros numéricos sujetos a investigación y requisitos. El documento es una hoja de ruta, NO un reporte definitivo ni una afirmación de cumplimiento de ISO.
