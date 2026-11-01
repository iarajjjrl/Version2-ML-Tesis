# Registro de resultados piloto V3 (1 de noviembre de 2026)

## Procedencia y configuración

Resultados compartidos mediante terminal PowerShell del usuario. Estado: pruebas unitarias 8/8 correctas, corrida `results-v3-controlled` terminada, scripts de resumen y costo de errores ejecutados.

- Datos: IEEE-CIS, submuestra sistemática cronológica de 120 000 transacciones; entrenamiento 60 000, holdout inicial 12 000, ocho ventanas posteriores con 48 000 transacciones.
- AutoML: cinco algoritmos, un trial de Optuna por algoritmo, semilla 42. Solo prueba piloto.
- Política periódica: cada cuatro ventanas temporales.
- Política event-driven controlada: activar si F1 < 0.50 en dos ventanas consecutivas, espera de una ventana entre activaciones y promoción cuando un candidato mejora el F1 en al menos 0.01 sobre muestra reciente separada.
- El entrenamiento después de la ventana final queda OMITIDO del costo observado.

## Métricas del horizonte estudiado

| Política | Media F1 | F1 agregado | Media Recall | FN | FP | Intentos de actualización | Promociones | Entrenamiento adicional (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Estática | 0.458586 | 0.462851 | 0.364206 | 1123 | 388 | 0 | 0 | 0.000000 |
| Periódica | 0.463576 | 0.465003 | 0.378406 | 1103 | 441 | 1 | 1 | 16.926533 |
| Eventos controlada | 0.475490 | 0.479816 | 0.411453 | 1043 | 542 | 1 | 1 | 19.657384 |

Costo inicial común del baseline: 16.07 s aproximadamente (sin contabilizar en los tiempos adicionales de la tabla).

**Diferencias eventos - periódico:** media F1 +0.011914; FN -60; FP +101; entrenamiento adicional +2.730851 s. **Diferencias eventos - estático:** media F1 +0.016904; FN -80; FP +154; entrenamiento adicional +19.657384 s.

## Sensibilidad a costo relativo del error (hipotético)

Con costo(FP)=1, costo(FN)=r, costo de error = FP + r * FN. No se trata de moneda real.

| r | Estática | Periódica | Eventos | Mínimo |
|---:|---:|---:|---:|---|
| 1 | 1511 | 1544 | 1585 | Estática |
| 2 | 2634 | 2647 | 2628 | Eventos |
| 5 | 6003 | 5956 | 5757 | Eventos |
| 10 | 11618 | 11471 | 10972 | Eventos |
| 20 | 22848 | 22501 | 21402 | Eventos |

La estrategia de eventos supera a la periódica en **costos de error exclusivamente** si r > 101/60 = 1.6833; no suma gasto de CPU ni costos operativos del sistema. Si se consideran cómputo y errores, el umbral depende de precios verificables que aún no se midieron.

## Interpretación y limitaciones

1. Los resultados son de **una sola serie temporal** y un trial de Optuna por algoritmo; NO existe inferencia estadística ni evidencia suficiente de superioridad universal.
2. El F1 medio por ventana y el F1 agregado sobre todas las transacciones son métricas distintas y se reportan por separado.
3. El tiempo de entrenamiento es local y no incluye el costo total de monitoreo, RAM, procesamiento en GitHub Actions, transferencia, espera, publicación ni despliegue.
4. La disponibilidad de etiquetas al cierre de cada ventana es un supuesto que debe hacerse explícito.
5. Los umbrales F1=0.50, persistencia de dos ventanas, cooldown de una y margen de promoción 0.01 aún son hipótesis de diseño. Hace falta análisis de sensibilidad predefinido.
6. El no reentrenamiento tras la última ventana corresponde a un horizonte finito; un sistema productivo que continúa operando podría asumir un costo fuera de este horizonte.
7. Hace falta escenario de cambio de distribución controlado y repeticiones del presupuesto de búsqueda; debe informarse estabilidad y variación.

**Próximo hito:** cerrar un protocolo de comparación con escenarios natural/sintético, presupuesto fijo, repeticiones y medición de recursos. Congelar ese protocolo antes de analizar los resultados definitivos.
