# Integración experimental del controlador de eventos y el prototipo MLOps

**Rama:** experiment/integracion-controlador-mlops-20261113

Esta implementación mantiene intactos el workflow histórico \`.github/workflows/retrain.yml\`, \`src/monitor.py\`, \`src/folder_watcher.py\`, \`main\` y \`models/active_model.json\`.

## Alcance demostrado por el código

Esta ruta experimental reutiliza SIN COPIAR sus reglas:

- \`src/experiments/v2/controller.py\` para solicitudes, persistencia temporal, cooldown y promoción.
- \`src/experiments/v2/trainer.py\` para selección entre cinco familias, ajuste cronológico, calibración y creación de un ModelBundle.
- \`src/mlops/controller_bridge.py\` para guardar y restaurar el controlador entre lotes.
- \`src/mlops/controlled_v2.py\` para entrenar M0 y procesar las mismas ventanas raw-v2 de forma secuencial.
- \`src/mlops/watch_controlled_v2.py\` para recibir copias comprobables de las ventanas de entrada.
- \`dockerfile.controlled_v2\` para ejecutar esta ruta en un contenedor separado.
- \`.github/workflows/verify-controlled-v2.yml\` para pruebas de integración en GitHub Actions sin publicar modelos.

El modelo inicial se comparte dentro de la comparación local del Estudio A. El prototipo inicia SU PROPIA sesión con el mismo entrenador y el mismo protocolo de datos. No debe afirmarse que comparte los mismos artefactos binarios ni los tiempos de entrenamiento medidos en A.

## Regla temporal compartida

1. La ventana k se evalúa con el modelo vigente antes de actualizarlo.
2. Dos ventanas seguidas con F1 menor que 0.50 habilitan una solicitud de mantenimiento.
3. La solicitud respeta un cooldown de una ventana y no se ejecuta sin al menos dos ventanas futuras disponibles.
4. El candidato se entrena con historia conocida hasta k. La ventana k+1 permanece fuera de ese entrenamiento.
5. En k+1 se compara F1 del candidato con el vigente. Solo se aprueba una mejora igual o superior a 0.01.
6. El candidato aprobado se emplea por primera vez para puntuar la ventana k+2.
7. El estado, los artefactos y las huellas SHA256 quedan bajo un directorio experimental independiente.

Estas reglas son decisiones experimentales, no estándares universales de MLOps.

## Ejecución local con el dataset real

Se requieren \`train_transaction.csv\` y \`train_identity.csv\` de IEEE-CIS en la raíz local del proyecto. Los archivos originales no se suben a GitHub.

Preparar las ventanas con la misma herramienta del Estudio A:

\`\`\`powershell
python -m src.experiments.v2.raw_windows --transactions train_transaction.csv --identity train_identity.csv --output data/experiments/raw-v2/prototipo --max-rows 120000 --windows 8
\`\`\`

Crear estado M0 separado:

\`\`\`powershell
python -m src.mlops.controlled_v2 bootstrap --raw-dir data/experiments/raw-v2/prototipo --store data/experiments/controlado-v2 --trials 3
\`\`\`

Procesar las ventanas una por una y comprobar estado entre cada ejecución:

\`\`\`powershell
python -m src.mlops.controlled_v2 step --store data/experiments/controlado-v2
python -m src.mlops.controlled_v2 status --store data/experiments/controlado-v2
\`\`\`

Cada ejecución de \`step\` consume solo la siguiente ventana. Repetir hasta ocho veces. No correr ocho comandos simultáneamente.

El archivo \`data/experiments/controlado-v2/state.json\` registra el F1, Precision, Recall, TP, FP, FN, TN, intentos, candidatos, promociones y tiempos de entrenamiento por ventana. Los modelos se guardan en \`data/experiments/controlado-v2/models/\`. El reentrenamiento puede tardar según el equipo utilizado.

## Docker aislado

\`\`\`powershell
docker build -f dockerfile.controlled_v2 -t tesis-controlado-v2 .
docker run --rm -v "\${PWD}/data:/app/data" tesis-controlado-v2 bootstrap --raw-dir /app/data/experiments/raw-v2/prototipo --store /app/data/experiments/controlado-v2 --trials 3
docker run --rm -v "\${PWD}/data:/app/data" tesis-controlado-v2 step --store /app/data/experiments/controlado-v2
\`\`\`

La carpeta \`/app/data\` debe mantenerse montada porque contiene el estado y los modelos. No se requiere GitHub Token.

Para la ingesta observada, ejecutar el watcher en otra terminal, con la imagen anterior y el mismo volumen:

\`\`\`powershell
docker run --rm --entrypoint python -v "\${PWD}/data:/app/data" tesis-controlado-v2 -m src.mlops.watch_controlled_v2 --store /app/data/experiments/controlado-v2 --incoming /app/data/ingesta_controlada --processed /app/data/procesados_controlados
\`\`\`

Copiar a \`data/ingesta_controlada\` el archivo \`window_01_raw.csv\` de la carpeta de preparación. Luego copiar \`window_02_raw.csv\` y así sucesivamente. El watcher comprueba que cada copia coincide mediante SHA256 con la ventana que corresponde. Nunca puntúa un lote que no corresponda al orden esperado.

## Pruebas automatizadas

\`\`\`powershell
python -m unittest discover -s tests -p "test_controlled_policy_bridge.py" -v
python -m unittest discover -s tests -p "test_controlled_adapter_v2.py" -v
\`\`\`

El primer grupo no necesita entrenadores ni librerías de ML. El segundo usa pandas y las librerías del proyecto, pero sustituye el entrenamiento por un modelo ficticio para probar el recorrido completo en segundos. GitHub Actions verifica ambos grupos en esta rama sin ejecutar entrenamientos de fraude.

## Límites y evidencias que faltan

- **No se ha modificado** el deployment histórico de \`main\`. Ese flujo antiguo sigue usando una activación y una validación diferentes.
- El workflow nuevo de GitHub Actions **solo comprueba las pruebas del controlador y del adaptador**. No es todavía reentrenamiento en nube ni publicación de un modelo.
- Falta ejecutar el prototipo aislado con las ocho ventanas IEEE-CIS reales y guardar sus registros antes de presentarlo como una implementación verificada con esos datos.
- Las etiquetas se asumen disponibles al terminar cada ventana. El retraso real de confirmación de fraude no está simulado.
- La comparación local mide tiempos de entrenamiento en el equipo local, no dólares en nube.
- No se pueden sustituir por estos resultados los experimentos históricos del Estudio A ni atribuirlos al funcionamiento de GitHub Actions.

El uso documental correcto es explicar que se ha desarrollado una **integración experimental del controlador del escenario por eventos con un prototipo local y contenedorizable**. Los resultados solo se incorporarán al capítulo de experimentación tras ejecutar y comprobar las pruebas correspondientes.
