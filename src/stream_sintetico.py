import os
import sys
import time
import requests
import subprocess
import hashlib
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitor import evaluar_lote_produccion, obtener_huella_modelo

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _git_validado(argumentos):
    resultado = subprocess.run(["git", *argumentos], cwd=PROJECT_ROOT, capture_output=True, text=True)
    token = os.getenv("GITHUB_TOKEN", "")
    stdout = resultado.stdout.strip().replace(token, "[REDACTADO]") if token else resultado.stdout.strip()
    stderr = resultado.stderr.strip().replace(token, "[REDACTADO]") if token else resultado.stderr.strip()
    print(f"[GIT] git {' '.join(argumentos)} | returncode={resultado.returncode}", flush=True)
    if stdout:
        print(f"[GIT][stdout] {stdout}", flush=True)
    if stderr:
        print(f"[GIT][stderr] {stderr}", flush=True)
    return stdout if resultado.returncode == 0 else None


def _leer_modelo_activo(exigir_hash_manifiesto=True):
    with open(os.path.join(PROJECT_ROOT, "models", "active_model.json"), encoding="utf-8") as archivo:
        manifiesto = json.load(archivo)
    ruta_relativa = manifiesto.get("modelo_activo", "")
    ruta_modelo = os.path.abspath(os.path.join(PROJECT_ROOT, ruta_relativa))
    if os.path.commonpath([PROJECT_ROOT, ruta_modelo]) != PROJECT_ROOT or not os.path.isfile(ruta_modelo):
        raise ValueError(f"[HOT-SWAP][ERROR] Ruta de modelo inválida o inexistente: {ruta_relativa}")
    with open(ruta_modelo, "rb") as archivo:
        sha256 = hashlib.sha256(archivo.read()).hexdigest()
    if exigir_hash_manifiesto and manifiesto.get("sha256") != sha256:
        raise ValueError(f"[HOT-SWAP][ERROR] SHA-256 no coincide para {ruta_relativa}")
    modelo = joblib.load(ruta_modelo)
    algoritmo = getattr(modelo, "name", modelo.__class__.__name__)
    if exigir_hash_manifiesto and manifiesto.get("algoritmo") != algoritmo:
        raise ValueError(f"[HOT-SWAP][ERROR] Algoritmo no coincide: manifiesto={manifiesto.get('algoritmo')} artefacto={algoritmo}")
    version = manifiesto.get("version") or manifiesto.get("despliegue") or os.path.basename(ruta_modelo)
    if not version:
        raise ValueError("[HOT-SWAP][ERROR] El manifiesto no contiene timestamp/versión")
    commit_modelo = _git_validado(["rev-parse", "origin/main"])
    if not commit_modelo:
        raise ValueError("[HOT-SWAP][ERROR] No se pudo leer el commit remoto origin/main")
    return {"ruta": ruta_relativa, "sha256": sha256, "algoritmo": algoritmo,
            "version": version, "commit_modelo": commit_modelo}

def esperar_github_actions_dinamico(timestamp_disparo, timeout_segundos=400, intervalo_poll=8):
    """
    Monitoreo en 2 Fases:
    1. Engancha el Run ID específico generado tras el disparo.
    2. Monitorea exclusivamente ese Run ID hasta su estado COMPLETED.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    
    if not token:
        print("[AUTO-POLL]  No hay GITHUB_TOKEN en .env. No se puede monitorear la nube.")
        return False
        
    url_lista = f"https://api.github.com/repos/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print("\n[AUTO-POLL]  FASE 1: Aguardando a que GitHub Actions levante la máquina virtual...")
    time.sleep(5)
    
    run_id_objetivo = None
    tiempo_inicio = time.time()
    
    while time.time() - tiempo_inicio < 60:
        try:
            response = requests.get(url_lista, headers=headers)
            if response.status_code == 200:
                runs = response.json().get("workflow_runs", [])
                for r in runs:
                    status = r.get("status")
                    created_at_str = r.get("created_at")
                    if created_at_str:
                        dt_created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        if dt_created.timestamp() >= (timestamp_disparo - 15):
                            if status in ["queued", "in_progress"]:
                                run_id_objetivo = r.get("id")
                                print(f"[AUTO-POLL]  ¡Enganchado al Run ID activo: {run_id_objetivo}! Estado inicial: {status.upper()}")
                                break
                if run_id_objetivo:
                    break
        except Exception as e:
            print(f"[AUTO-POLL] Error de conexión: {e}")
            
        print("[AUTO-POLL]  Esperando registro del pod virtual en GitHub API...")
        time.sleep(5)
        
    if not run_id_objetivo:
        print("[AUTO-POLL]  No se detectó ninguna ejecución activa nueva. Asumiendo timeout o fallo de evento.")
        return False

    print(f"[AUTO-POLL]  FASE 2: Monitoreando el pipeline en la nube (Run ID: {run_id_objetivo})...")
    url_run_especifico = f"https://api.github.com/repos/{repo}/actions/runs/{run_id_objetivo}"
    
    while time.time() - tiempo_inicio < timeout_segundos:
        try:
            response = requests.get(url_run_especifico, headers=headers)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                conclusion = data.get("conclusion")
                
                if status == "completed":
                    print(f"[AUTO-POLL]  ¡El Run ID {run_id_objetivo} ha FINALIZADO! Conclusión: {conclusion.upper()}")
                    if conclusion == "success":
                        print("[AUTO-POLL] ⏸ Pausa de 5s para asentamiento de cambios en git...")
                        time.sleep(5)
                        return True
                    else:
                        return False
                else:
                    print(f"[AUTO-POLL]  Reentrenando modelo en la nube... Estado: {status.upper()}")
        except Exception as e:
            print(f"[AUTO-POLL] Error de conexión: {e}")
            
        time.sleep(intervalo_poll)
        
    print("[AUTO-POLL] ⏱ Timeout alcanzado esperando la finalización en GitHub Actions.")
    return False

def ejecutar_git_pull_automatico():
    print("[AUTO-PULL] Sincronizando únicamente models/ desde origin/main...", flush=True)
    if _git_validado(["fetch", "origin", "main"]) is None:
        return False
    if _git_validado(["checkout", "origin/main", "--", "models"]) is None:
        print("[AUTO-PULL][ERROR] Checkout de models/ falló; se cancela Hot-Swap y reevaluación.", flush=True)
        return False
    return True

def ejecutar_simulacion_stream():
    print("="*60)
    print("  SIMULADOR DE INGESTA TRANSACCIONAL EN TIEMPO REAL (MLOps)  ")
    print("="*60)
    
    escenarios = [
        {
            "nombre": "Lote #1: Tráfico Normal de Producción (Operación Nominal)",
            "ruta_csv": "data/prod_lote1_normal.csv"
        },
        {
            "nombre": "Lote #2: Inyección de Concept Drift (Ataque Evasivo / Estacionalidad)",
            "ruta_csv": "data/prod_lote2_drift.csv"
        },
        {
            "nombre": "Lote #3: Tráfico Post-Reentrenamiento (Validación de Recuperación)",
            "ruta_csv": "data/prod_lote3_post_retrain.csv"
        }
    ]
    
    for i, esc in enumerate(escenarios, 1):
        print(f"\n\n[STREAM] --- SIMULANDO VENTANA #{i}: {esc['nombre']} ---")
        
        ruta_lote = esc["ruta_csv"]
        if not os.path.exists(ruta_lote):
            print(f"[ERROR] No se encontró el archivo '{ruta_lote}'. Ejecuta 'python src/prepare_data.py' primero.")
            continue
        
        aprobado = False
        intentos = 0
        max_intentos = 3
        
        while not aprobado and intentos < max_intentos:
            intentos += 1
            if intentos > 1:
                print(f"\n[RE-EXAMEN]  Evaluando nuevamente la VENTANA #{i} tras reentrenamiento (Intento {intentos})...")
            
            timestamp_evaluacion = datetime.now(timezone.utc).timestamp()
            
            registro = evaluar_lote_produccion(ruta_lote)
            
            f1 = registro["f1_score_fraude"]
            psi = registro["psi_valor"]
            alerta = registro["alerta_disparada"]
            
            # Criterio de Aprobación MLOps:
            # Si el F1-Score supera la regla de calidad (0.50), el modelo
            # demuestra que sabe predecir con alta precisión sobre la nueva realidad.
            if f1 >= 0.50:
                print(f"[ÉXITO]  El modelo APROBÓ el lote con un F1-Score de {f1:.4f} >= 0.50. Avanzando...")
                aprobado = True
            else:
                print(f"[REPROBADO]  El modelo REPROBÓ el lote (F1: {f1:.4f} < 0.50 | PSI: {psi:.4f}).")
                
                if alerta:
                    print("[STREAM] Alerta enviada a la nube. Esperando solución de GitHub Actions...")
                    exito_nube = esperar_github_actions_dinamico(timestamp_disparo=timestamp_evaluacion)
                    
                    if exito_nube:
                        try:
                            modelo_anterior = _leer_modelo_activo(exigir_hash_manifiesto=False)
                            sincronizado = ejecutar_git_pull_automatico()
                            if not sincronizado:
                                break
                            modelo_nuevo = _leer_modelo_activo()
                            print(
                                f"[HOT-SWAP][COMPARACION] anterior_sha256={modelo_anterior['sha256']} "
                                f"anterior_algoritmo={modelo_anterior['algoritmo']} "
                                f"nuevo_sha256={modelo_nuevo['sha256']} "
                                f"nuevo_algoritmo={modelo_nuevo['algoritmo']} "
                                f"version={modelo_nuevo['version']} "
                                f"commit_modelo_anterior={modelo_anterior['commit_modelo']} "
                                f"commit_modelo_nuevo={modelo_nuevo['commit_modelo']}",
                                flush=True,
                            )
                            if modelo_nuevo["sha256"] == modelo_anterior["sha256"]:
                                print("[HOT-SWAP][ERROR] No se publicó un artefacto nuevo; no se reevaluará.", flush=True)
                                break
                        except Exception as error:
                            print(f"[HOT-SWAP][ERROR] Validación/sincronización fallida: {error}", flush=True)
                            break
                        print("[HOT-SWAP] Artefacto nuevo validado; reevaluando el mismo lote...", flush=True)
                    else:
                        print("[STREAM]  GitHub Actions no logró generar un modelo superior. Manteniendo modelo actual.")
                        break
                else:
                    print("[STREAM] En periodo de Cooldown FinOps o Nube Ocupada. No se enviará evento duplicado.")
                    break

if __name__ == "__main__":
    ejecutar_simulacion_stream()
