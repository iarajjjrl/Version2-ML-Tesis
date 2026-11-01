# Registro de resultados piloto v2 (1 de noviembre de 2026)

Este documento registra resultados aportados por el usuario. Son exploratorios, NO son resultados definitivos de tesis ni representan costos de nube.

## Configuración

- IEEE-CIS: 120 000 registros submuestreados cronológicamente, entrenamiento 60 000, holdout 12 000 y ocho ventanas futuras con 48 000 registros.
- Cinco algoritmos y UN trial de Optuna por algoritmo, semilla 42.
- Periodic: actualizar cada cuatro ventanas; Event: F1 por debajo de 0.50.
- Promoción: mejora mínima de F1 de 0.01 en el último 30% de una ventana, reservado de la actualización del candidato.
- Simple: 1 ventana de degradación, sin cooldown; controlado: 2 ventanas consecutivas, cooldown de 1 ventana.
- F1 medio: media aritmética de los F1 de las ocho ventanas. F1 agregado: F1 calculado con la suma de TP, FP y FN.

## Resultados tal como se ejecutaron

| Política | F1 medio | F1 agregado | Recall medio | FN | FP | Reentrenamientos | Promociones | Entreno (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 0.458586 | 0.462851 | 0.364206 | 1123 | 388 | 0 | 0 | 0 |
| Periodic (corrida simple) | 0.463576 | 0.465003 | 0.378406 | 1103 | 441 | 2 | 2 | 40.451716 |
| Event simple | 0.491926 | 0.495809 | 0.433834 | 1005 | 559 | 4 | 3 | 82.010486 |
| Periodic (corrida controlada) | 0.463576 | 0.465003 | 0.378406 | 1103 | 441 | 2 | 2 | 39.188214 |
| Event controlado | 0.475490 | 0.479816 | 0.411453 | 1043 | 542 | 2 | 2 | 46.229555 |

La política Event controlada redujo el número de intentos de 4 a 2 y los segundos de entrenamiento de 82.01 a 46.23 frente a Event simple (~43.6% menos), pero su F1 medio cayó de 0.4919 a 0.4755. Frente a Periodic, Event controlado obtuvo 60 FN menos, pero 101 FP más. No se conoce el costo monetario de esos errores, por lo que solo se pueden probar ratios de costos hipotéticos.

## Sesgo identificado y corregido después de estas ejecuciones

Ambas corridas contabilizaron un entrenamiento DESPUÉS de la ventana 8 (última ventana observada) para Periodic. La política Event controlada también reentrenó después de la ventana 8. No existe ventana posterior donde medir el aporte de estas actualizaciones; los costos informados incluyen trabajo fuera del horizonte de evaluación.

Se corrigió el código de experimentación de la rama para omitir por defecto la actualización final, con una opción explícita para incluirla en servicios que continúan fuera del horizonte. Las cifras de esta tabla son el registro histórico sin alterar. El análisis de horizonte finito requiere nuevas ejecuciones (v3), sin manipular artificialmente los resultados antiguos.

## Limitaciones y siguientes pasos

1. El presupuesto de un trial por algoritmo es solo un test funcional, no una validación de excelencia ni comparación estadística.
2. Los valores F1 0.50, intervalos de cuatro ventanas, cooldown de una ventana y mínimo de mejora 0.01 son parámetros experimentales, no estándares bancarios.
3. Se supone disponibilidad inmediata de etiquetas al cierre de la ventana; fraude real puede sufrir demoras.
4. Estos tiempos son locales: no incluyen costos completos de CPU, RAM, energía, GitHub Actions, colas ni despliegue.
5. Faltan pruebas con datos completos, escenario de drift sintético separado, análisis de sensibilidad, presupuesto de entrenamiento más amplio y repeticiones.
6. Los candidatos solo demuestran beneficio prospectivo en ventanas posteriores a la que dispara su actualización.

Fuente: resultados PowerShell y summary.csv compartidos en conversación el 1 de noviembre de 2026.
