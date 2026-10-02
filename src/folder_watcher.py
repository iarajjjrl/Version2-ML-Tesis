import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Mantiene la raíz del proyecto en sys.path al ejecutar este archivo directamente.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.monitor import evaluar_lote_produccion
from src.stream_sintetico import (
    esperar_github_actions_dinamico,
    ejecutar_git_pull_automatico,
)


INGESTA_DIR = os.path.join(PROJECT_ROOT, "data", "ingesta_viva")
PROCESADOS_DIR = os.path.join(PROJECT_ROOT, "data", "procesados")


class LoteIngestaHandler(FileSystemEventHandler):
    """Procesa lotes CSV y coordina la recuperación automática del modelo."""

    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".csv"):
            return

        print(f"[INGESTA] Nuevo lote detectado: {event.src_path}", flush=True)

        try:
            # Espera a que el sistema operativo complete la escritura del archivo.
            time.sleep(1)

            timestamp_evaluacion = datetime.now(timezone.utc).timestamp()
            registro = evaluar_lote_produccion(event.src_path)

            if registro.get("alerta_disparada") is True:
                print(
                    "[RECUPERACIÓN] Alerta disparada; esperando el reentrenamiento en la nube...",
                    flush=True,
                )
                exito_nube = esperar_github_actions_dinamico(timestamp_evaluacion)

                if exito_nube:
                    exito_pull = ejecutar_git_pull_automatico()
                    if exito_pull:
                        print(
                            "[HOT-SWAP] Nuevo modelo descargado; reevaluando el lote con el modelo activo.",
                            flush=True,
                        )
                        registro = evaluar_lote_produccion(event.src_path)
                        print(
                            f"[HOT-SWAP] Reevaluación completada: {registro}",
                            flush=True,
                        )
                    else:
                        print(
                            "[HOT-SWAP][ERROR] No se pudo descargar el nuevo modelo.",
                            flush=True,
                        )
                else:
                    print(
                        "[RECUPERACIÓN][ERROR] La ejecución de GitHub Actions no finalizó correctamente.",
                        flush=True,
                    )
            else:
                print(
                    f"[INGESTA] Lote evaluado; no se disparó alerta: {event.src_path}",
                    flush=True,
                )

            return registro
        except Exception as error:
            print(
                f"[INGESTA][ERROR] No se pudo procesar '{event.src_path}': {error}",
                flush=True,
            )
        finally:
            if os.path.isfile(event.src_path):
                try:
                    os.makedirs(PROCESADOS_DIR, exist_ok=True)
                    destino = os.path.join(PROCESADOS_DIR, os.path.basename(event.src_path))
                    if os.path.exists(destino):
                        nombre, extension = os.path.splitext(os.path.basename(event.src_path))
                        sufijo = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                        destino = os.path.join(PROCESADOS_DIR, f"{nombre}_{sufijo}{extension}")
                    shutil.move(event.src_path, destino)
                    print(f"[INGESTA] Lote movido a procesados: {destino}", flush=True)
                except Exception as error:
                    print(
                        f"[INGESTA][ERROR] No se pudo mover '{event.src_path}' a procesados: {error}",
                        flush=True,
                    )


def iniciar_vigilancia():
    os.makedirs(INGESTA_DIR, exist_ok=True)
    print(f"[INGESTA] Vigilando carpeta: {INGESTA_DIR}", flush=True)

    observer = Observer()
    observer.schedule(LoteIngestaHandler(), INGESTA_DIR, recursive=False)

    print("\n[INFO] Inicializando configuración GitOps del contenedor...", flush=True)
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")

    if token and repo:
        url_autenticada = f"https://{token}@github.com/{repo}.git"
        try:
            if not os.path.exists(os.path.join(PROJECT_ROOT, ".git")):
                subprocess.run(["git", "init"], cwd=PROJECT_ROOT, check=True)
                subprocess.run(
                    ["git", "remote", "add", "origin", url_autenticada],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
                subprocess.run(["git", "fetch", "origin", "main"], cwd=PROJECT_ROOT, check=True)
                subprocess.run(
                    ["git", "reset", "--mixed", "origin/main"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
                subprocess.run(
                    ["git", "branch", "-M", "main"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
                print("[INFO] Historial remoto sincronizado correctamente.", flush=True)
            else:
                subprocess.run(
                    ["git", "remote", "set-url", "origin", url_autenticada],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
        except Exception as error:
            # No imprimir el texto de la excepción: podría incluir la URL con el token.
            print(
                f"[ERROR] No se pudo configurar GitOps ({type(error).__name__}).",
                flush=True,
            )
    else:
        print("[ERROR] Faltan GITHUB_TOKEN o GITHUB_REPOSITORY; GitOps no está configurado.", flush=True)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INGESTA] Interrupción recibida; cerrando vigilancia...", flush=True)
    finally:
        observer.stop()
        observer.join()
        print("[INGESTA] Vigilancia detenida.", flush=True)


if __name__ == "__main__":
    iniciar_vigilancia()
