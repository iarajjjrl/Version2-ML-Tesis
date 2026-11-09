# Estudio A — Referencia congelada de la ejecución exploratoria (v1)

**Fecha de registro:** 2026-11-09. **Estado:** referencia histórica; no se reejecutó ni reentrenó ningún modelo.

- **Repositorio:** `iarajjjrl/Version2-ML-Tesis`
- **Commit exacto de código:** `580a4b5ce317ba894f882ed8158d6b547e4d6ef7`
- **Rama de archivo:** `archive/study-a-v1-20261109` (creada sobre el commit exacto; no se agregan commits a esa rama).
- **Rama de desarrollo del protocolo v2:** `experiment/study-a-v2-protocol-20261109`, creada originalmente desde el mismo commit.
- **Archivo de resultados original:** `runs.zip` (adjunto a la conversación; no subido a GitHub porque contiene modelos grandes y snapshots de datos).
- **Bytes del ZIP:** 250845705.
- **SHA-256 completo de `runs.zip`:** `723a6e4c0fd155882c3d1be9415fe07f9d032156ba53afc152bbbd86e41e6845`.

## Configuración original documentada en `configuration.json`

IEEE-CIS Fraud Detection; 590540 filas originales; submuestra sistemática de 120000; 60000 para historia inicial; 12000 para holdout histórico; 8 ventanas futuras de igual duración relativa sobre 48000 filas; escenarios natural y perturbación sintética de covariables desde la ventana 3. Semillas 42, 43 y 44. Optuna: **3 trials por familia**, cinco familias. Periódica: cada 4 ventanas; evento: F1 menor que 0.50 durante dos ventanas, cooldown de una ventana. Promoción: mejora F1 >=0.01, partición final de ventana de 30 %. Perfilado de recursos activo; etiquetas supuestas disponibles al cierre de ventana.

## Huellas SHA-256 de los seis `comparison.csv` dentro del ZIP

| Escenario | Semilla | Ruta relativa a `runs.zip` | SHA-256 |
|---|---:|---|---|
| Natural | 42 | `runs/natural/seed_42/comparison.csv` | `ec81d2fa468ac3a6a0a4f7f606e9632fbd0d2cce4b57561cbe8e8472b51ba95f` |
| Natural | 43 | `runs/natural/seed_43/comparison.csv` | `a15cdd0426617b8ccdf194db3919a455e2e30ba84ede094c87a102300191b340` |
| Natural | 44 | `runs/natural/seed_44/comparison.csv` | `250c8efc088b696407008ba719d769a9ae05a27e22ca1043e19d26cc064392ca` |
| Sintético | 42 | `runs/synthetic/seed_42/comparison.csv` | `0a52c0e310920e08aec833be0c35e72a1da3c0a0caed4b5917eee1e2ccf5ee84` |
| Sintético | 43 | `runs/synthetic/seed_43/comparison.csv` | `278c54a1d393b30f5096aa6922566d85e797396c6cd14645238e988227f99ada` |
| Sintético | 44 | `runs/synthetic/seed_44/comparison.csv` | `9c61b363b83b959a32eb45b49f447ae40d0c2559679c915b883202350bb145aa` |

Los hashes de `configuration.json`, `baseline_metrics.json` y `summary.csv` de las seis corridas constan en el manifiesto completo de procedencia generado localmente (`Estudio_A_V1_Manifiesto_SHA256.json`). Los archivos binarios originales del dataset IEEE-CIS **no se recibieron** y su hash no pudo comprobarse aquí.

## Qué conserva y qué no conserva esta referencia

El commit y la rama de archivo preservan **código**. El SHA del ZIP verifica **la integridad de la copia específica de los resultados**, pero no constituye por sí mismo un respaldo remoto de esos bytes. El usuario debe conservar `runs.zip` de forma segura. Los modelos no fueron subidos a GitHub y GitHub `main` quedó intacta.

La comparación v1 es **exploratoria**. Sus números no serán reemplazados silenciosamente por v2; cualquier cambio de protocolo debe aparecer en un manifiesto separado, indicando commit, datos y motivo.