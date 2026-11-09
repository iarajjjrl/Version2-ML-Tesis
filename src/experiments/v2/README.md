# Study A v2 — implementación parcial (sin benchmark)

**Fecha:** 2026-11-09. **Rama:** `experiment/study-a-v2-protocol-20261109`. **v1 no modificada.**

Este directorio incorpora los primeros componentes del protocolo revisado. **No es todavía un benchmark v2 integrado** y no debe utilizarse para sustituir las cifras exploratorias originales. El script v1 `src/experiments/run_study.py` permanece sin cambios y **sigue ejecutando v1**, no v2.

## Archivos

- `raw_windows.py`: crea archivos crudos, con `TransactionDT` y `TransactionID` conservados. No ajusta ningún preprocesador y rechaza sobrescribir una carpeta de salida no vacía.
- `temporal.py`: crea bloques cronológicos 70/15/15 (fit/tune/calibration) y solo aplica `fit` del preprocesador al bloque inicial. Necesita columnas crudas, no los CSV transformados de v1.
- `controller.py`: controla eventos, periodicidades, cooldown, presupuesto de intentos y la secuencia temporal: solicitud en k, evaluación del candidato en k+1 y uso del modelo promocionado desde k+2. Tras promocionar un nuevo modelo, descarta el historial de F1 bajo del modelo anterior.
- `tests/test_study_a_v2.py`: pruebas unitarias con datos sintéticos pequeños; no entrenan modelos ni necesitan IEEE-CIS.

## Pruebas locales

Desde la raíz del repositorio, con pandas, numpy, scikit-learn y pytest instalados:

```bash
python -m pytest -q tests/test_study_a_v2.py
```

**Verificación de desarrollo:** 16 pruebas superadas en una copia local del código de estos módulos el 2026-11-09; se comprobó que el SHA de los blobs publicados en GitHub coincide con la versión local. **No se ha ejecutado CI en GitHub, ni un entrenamiento completo con el dataset real.**

## Preparación de datos crudos (solo cuando se necesite)

```bash
python -m src.experiments.v2.raw_windows \
  --transactions train_transaction.csv \
  --identity train_identity.csv \
  --output data/experiments/raw-v2/natural \
  --max-rows 120000 --windows 8
```

Para el escenario sintético, usar **otra carpeta de salida** e indicar `--inject-drift-from 3`. Este comando solo prepara datos; no crea un modelo. Requiere los archivos originales de IEEE-CIS de manera local.

## Bloqueos antes de un nuevo benchmark

1. Falta crear el entrenador v2 que seleccione hiperparámetros exclusivamente con el bloque `tune`, calibre el umbral exclusivamente en `calibrate` y guarde el preprocesador junto con el modelo activo.
2. Falta conectar el entrenador y el controlador en un nuevo ejecutor `run_comparison_v2.py`, con registros, fallos, control de tamaño de presupuestos, medición de recursos y resultados por ventana.
3. Faltan simulación de retraso de etiquetas, agendas aleatorias con presupuesto controlado, sensibilidad y validación con suficientes trayectorias.
4. Falta el Estudio B operacional ampliado, que es independiente de A.

**Ningún script v1 ha sido modificado y no se han reejecutado corridas.** El título del manuscrito permanece intacto.
