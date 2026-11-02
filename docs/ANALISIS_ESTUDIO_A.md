# Auditoría del estudio A — resultados recibidos y revisados

Origen: archivos locales study_runs.csv, study_aggregate.csv y study_config.json aportados por el autor el 1 de noviembre de 2026. No se copiaron las tablas originales al repositorio. Evidencia reproducible: SHA del commit del entorno 580a4b5ce317ba894f882ed8158d6b547e4d6ef7; Python 3.13.17; pandas 3.0.3; scikit-learn 1.7.2; Optuna 4.9.0; Windows 11.

## Diseño observado

- 18 filas: 2 escenarios × 3 semillas de AutoML (42, 43, 44) × 3 políticas.
- 120 000 registros, ocho ventanas; entrenamiento inicial en los primeros 60 000 y validación histórica 12 000; 48 000 en simulación futura.
- El escenario sintético cambia monto y C13 desde la ventana 3, sin cambiar etiquetas.
- Event: dos ventanas F1<0.50; cooldown una ventana. Periódica: cada cuatro ventanas. Estático: sin actualizaciones.
- Solo 3 trials por algoritmo. **CRÍTICO:** TPESampler usa por defecto n_startup_trials=10, por lo que tres trials de cada estudio son muestras iniciales aleatorias; no hay optimización TPE adaptativa en estas ejecuciones. Fuente: documentación oficial Optuna, TPESampler.
- La serie temporal es única. Cambiar seed solo prueba sensibilidad a búsquedas, no constituye una réplica independiente del proceso de fraude.

## Resultados descriptivos: promedio de tres seeds

| Escenario | Política | F1 agregado | F1 medio/ventana | Recall medio | FN | FP | Actualizaciones | Entreno local (s) | Pico RSS aprox (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Natural | static | 0.45310 | 0.44812 | 0.35047 | 1146.3 | 367.7 | 0 | 0.00 | n/d |
| Natural | periodic | 0.46307 | 0.46129 | 0.38475 | 1090.3 | 493.7 | 1.0 | 29.47 | 483.1 |
| Natural | event | 0.46864 | 0.46396 | 0.38829 | 1081.7 | 486.0 | 1.67 | 56.18 | 555.2 |
| Sintético | static | 0.43316 | 0.42632 | 0.32618 | 1187.7 | 346.0 | 0 | 0.00 | n/d |
| Sintético | periodic | 0.43623 | 0.43810 | 0.37261 | 1110.0 | 614.7 | 1.0 | 27.84 | 481.1 |
| Sintético | event | 0.45739 | 0.45177 | 0.37123 | 1111.0 | 460.7 | 2.33 | 73.48 | 589.1 |

Las cantidades decimales de FN y FP resultan de promediar tres ejecuciones; cada ejecución individual tiene conteos enteros. La memoria del estático queda sin dato en CSV: **NO es 0 MiB**. El costo inicial de entrenar baseline (común dentro de cada corrida) no aparece en columna de costo de actualizaciones.

## Comparación por semilla: F1 agregado

| Escenario | Seed | Estático | Periódico | Event |
|---|---:|---:|---:|---:|
| Natural | 42 | 0.46285 | 0.46500 | 0.47982 |
| Natural | 43 | 0.44260 | 0.45627 | 0.45591 |
| Natural | 44 | 0.45384 | 0.46795 | 0.47020 |
| Sintético | 42 | 0.44453 | 0.45722 | 0.47797 |
| Sintético | 43 | 0.42509 | 0.44315 | 0.44476 |
| Sintético | 44 | 0.42985 | 0.40832 | 0.44944 |

Event supera el F1 agregado de Static en 6/6 combinaciones. Supera al periódico en 5/6, pero en Natural seed 43 obtiene 0.45591 frente a 0.45627. **No demuestra significación estadística**.

**Natural**: ventaja descriptiva promedio de F1 agregado event - periodic = +0.00557; tiempo +26.71 s; ~1.91 veces más tiempo de entrenamiento.

**Sintético**: F1 agregado event - periodic = +0.02116; tiempo +45.64 s; ~2.64 veces más tiempo. En escenario sintético Event no mejora Recall medio de Periodic (0.37123 versus 0.37261), y hay 1 FN más en promedio (1111 versus 1110), pero produce muchas menos falsas alarmas (460.7 versus 614.7).

**Consecuencia:** NO sostener que Event ahorra cómputo frente al periódico ni que mejora todas las métricas. Existe un compromiso entre F1, falsas alarmas, fraudes perdidos y uso de recursos.

## Alertas metodológicas bloqueantes

1. **3 trials no alcanzan la fase adaptativa TPE**, pues el sampler tiene 10 intentos iniciales aleatorios por estudio. Aumentar y justificar presupuesto o describir como random search con Optuna; no cambiar retroactivamente la interpretación de resultados.
2. Las mediciones de CPU y RSS son aproximaciones de procesos locales durante entrenamiento; no incluyen facturación ni todo el ciclo CI/CD de GitHub Actions. Medir tiempo end-to-end y costo de monitoreo.
3. La variable F1 requiere etiquetas disponibles. Retraso de etiquetas y políticas PSI/KS no están validados en comparador actual.
4. El preprocesador aprende categorías/frecuencias sobre el conjunto baseline completo antes del split cronológico interno usado por Optuna. Eso implica fuga de información sobre estadísticas de la subsección de validación interna; se necesita separar fit del preprocesador por cada partición temporal relevante.
5. Cada estudio optimiza hiperparámetros y umbral de decisión en una misma validación interna, elevando riesgo de sobreajuste de selección.
6. No se controla la disponibilidad temporal de features transaccionales reales ni los límites de integración con el sistema antifraude. No afirmar producción bancaria.
7. El experimento no justifica todavía por qué ocho ventanas, 120 000 registros, cinco familias, umbral F1=0.50, cooldown 1 ventana y Delta F1=0.01. Protocolo y sensibilidad predefinidos antes del resultado final.
8. Al comparar 3 seeds sobre una misma secuencia, no realizar prueba t suponiendo que hay tres muestras independientes de datos.

## Recomendación inmediata

**No ejecutar más entrenamientos todavía.** Primero corregir leakage del preprocesador, separar selección/ajuste de umbral y validar correctamente TPE. Después preparar de una vez un presupuesto final y una lista mínima de escenarios, con evidencia operativa GitHub Actions/Docker. Actualizar metodología y capítulo experimental del artículo con las tablas de este análisis y distinguir claramente piloto de resultados definitivos.
