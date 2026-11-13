# Protocolo adicional: F1 de referencia para disparador por eventos (12-11-2026)

## Alcance y control

**Rama aislada:** `experiment/study-a-v2-relative-reference-20261112`.
Es una **variante prospectiva**, no un reemplazo retrospectivo del estudio con umbral absoluto F1 < 0.50. Los resultados de `experiment/study-a-v2-integracion-20261112` y los CSV ya generados permanecen intactos. No alterar el manuscrito ni afirmar mejoras antes de ejecutar las pruebas sobre IEEE-CIS.

### Tres conceptos distintos de baseline

1. **Modelo inicial congelado:** se entrena con el historial común; todas las políticas empiezan con el mismo artefacto.
2. **Política estática:** brazo comparador sin actualizaciones, mide el beneficio predictivo de mantener modelos.
3. **F1 de referencia por versión:** mide degradación del desempeño del modelo actualmente desplegado; NO es una política comparativa en sí misma.

### Regla del evento relativa

- **Referencia inicial:** F1 del modelo inicial sobre `diagnostic_holdout_raw.csv`, posterior al historial inicial y ajeno a tuning/calibración/entrenamiento.
- **Umbral:** `F1_efectivo = F1_referencia × (1 - relative_drop)`. Hipótesis predefinida de sensibilidad `relative_drop=0.15`, equivalente a 15% de degradación relativa; no equivale a restar 0.15 puntos absolutos.
- **Persistencia:** dos ventanas consecutivas por debajo del umbral (mismo supuesto de etiquetas disponibles al cierre de cada ventana).
- **Promoción:** candidato entrenado tras k, comparado contra el vigente en k+1, y si gana se activa en k+2.
- **Nueva referencia:** NO reutilizar el F1 de `calibrate` ni el F1 de la ventana de promoción k+1. Se fija en la **primera ventana puntuable del nuevo modelo, k+2**, que no se utilizó para entrenarlo, calibrarlo ni promocionarlo. La búsqueda de degradación empieza en k+3.
- **F1 igual a cero en la primera ventana de un modelo nuevo:** no se inventa un umbral relativo; la variante registra `reference_unavailable_zero_f1`. Es una limitación que debe reportarse, o gestionarse con un guardrail absoluto predefinido en investigación posterior.
- Las condiciones de `cooldown`, máximo de intentos y última ventana continúan idénticas.

**Cuidado:** comparar el F1 de ventanas con tasas de fraude distintas puede inducir alertas por cambio de prevalencia; registrar número de positivos, Recall, Precision, FP/FN y, en extensiones futuras, intervalos de incertidumbre. La ventana k+2 proporciona una observación independiente del ajuste/promoción pero no una referencia estadísticamente estable por sí misma. Evaluar robustez a referencias basadas en múltiples ventanas futuras, sin usar información futura en el momento de decisión.

## Verificación de código

```powershell
# Solo después de terminar cualquier proceso experimental actual.
git status -sb
git fetch origin
git switch experiment/study-a-v2-relative-reference-20261112
git pull --ff-only origin experiment/study-a-v2-relative-reference-20261112
python -m pytest -q tests/test_study_a_v2.py tests/test_study_a_v2_integration.py tests/test_study_a_v2_relative_reference.py
```

## Primer piloto relativo (época 1 natural, un trial por familia)

```powershell
python -m src.experiments.v2.run_comparison_v2 `
  --input data/experiments/raw-v2-epochs/epoch_01/natural `
  --output data/experiments/piloto-v2-epoch01-event-relative-15pct `
  --seed 42 `
  --trials 1 `
  --every 4 `
  --event-mode relative `
  --relative-drop 0.15 `
  --profile-resources
```

La salida tendrá las políticas `static`, `periodic`, `event_relative` y las columnas `reference_f1`, `trigger_threshold`. Conservar pilotos absolutos separados en carpetas distintas.

## Estudio exploratorio de tres épocas, siempre con salida nueva

```powershell
python -m src.experiments.v2.run_study_v2 `
  --epochs-dir data/experiments/raw-v2-epochs `
  --output data/experiments/estudio-a-v2-referencia-relativa-15pct `
  --seeds 42 `
  --every 4 `
  --trials 3 `
  --event-mode relative `
  --relative-drop 0.15 `
  --profile-resources
```

Comparar estrictamente con `estudio-a-v2-seed42-3trials` sobre las **mismas épocas, semilla, trials y periódico cada 4**, y distinguir `event` absoluto de `event_relative`. Las épocas comparten origen IEEE-CIS; los resultados no prueban generalización poblacional ni estadística significativa.

Este protocolo no modifica GitHub Actions ni demuestra costos cloud. La validación independiente del Estudio B sigue pendiente.
