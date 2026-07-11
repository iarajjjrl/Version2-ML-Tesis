import os
import sys
import time
import requests
import subprocess
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitor import evaluar_lote_produccion, obtener_huella_modelo

load_dotenv()

def esperar_github_actions_dinamico(timestamp_disparo, timeout_segundos=400, intervalo_poll=8):
    """
    Monitoreo en 2 Fases:
    1. Engancha el Run ID específico generado tras el disparo.
    2. Monitorea exclusivamente ese Run ID hasta su estado COMPLETED.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    
    if not token:
        print("[AUTO-POLL] ❌ No hay GITHUB_TOKEN en .env. No se puede monitorear la nube.")
        return False
        
    url_lista = f"https://api.github.com/repos/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print("\n[AUTO-POLL] ⏳ FASE 1: Aguardando a que GitHub Actions levante la máquina virtual...")
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
                                print(f"[AUTO-POLL] 🎯 ¡Enganchado al Run ID activo: {run_id_objetivo}! Estado inicial: {status.upper()}")
                                break
                if run_id_objetivo:
                    break
        except Exception as e:
            print(f"[AUTO-POLL] Error de conexión: {e}")
            
        print("[AUTO-POLL] ⏳ Esperando registro del pod virtual en GitHub API...")
        time.sleep(5)
        
    if not run_id_objetivo:
        print("[AUTO-POLL] ⚠️ No se detectó ninguna ejecución activa nueva. Asumiendo timeout o fallo de evento.")
        return False

    print(f"[AUTO-POLL] ⏳ FASE 2: Monitoreando el pipeline en la nube (Run ID: {run_id_objetivo})...")
    url_run_especifico = f"https://api.github.com/repos/{repo}/actions/runs/{run_id_objetivo}"
    
    while time.time() - tiempo_inicio < timeout_segundos:
        try:
            response = requests.get(url_run_especifico, headers=headers)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                conclusion = data.get("conclusion")
                
                if status == "completed":
                    print(f"[AUTO-POLL] 🏁 ¡El Run ID {run_id_objetivo} ha FINALIZADO! Conclusión: {conclusion.upper()}")
                    if conclusion == "success":
                        print("[AUTO-POLL] ⏸️ Pausa de 5s para asentamiento de cambios en git...")
                        time.sleep(5)
                        return True
                    else:
                        return False
                else:
                    print(f"[AUTO-POLL] 🚀 Reentrenando modelo en la nube... Estado: {status.upper()}")
        except Exception as e:
            print(f"[AUTO-POLL] Error de conexión: {e}")
            
        time.sleep(intervalo_poll)
        
    print("[AUTO-POLL] ⏱️ Timeout alcanzado esperando la finalización en GitHub Actions.")
    return False

def ejecutar_git_pull_automatico():
    try:
        print("[AUTO-PULL] 📥 Ejecutando 'git pull' para descargar el nuevo modelo de producción...")
        resultado = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True)
        print(resultado.stdout)
        return True
    except Exception as e:
        print(f"[AUTO-PULL] Error al ejecutar git pull: {e}")
        return False

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
                print(f"\n[RE-EXAMEN] 🔄 Evaluando nuevamente la VENTANA #{i} tras reentrenamiento (Intento {intentos})...")
            
            timestamp_evaluacion = datetime.now(timezone.utc).timestamp()
            
            registro = evaluar_lote_produccion(ruta_lote)
            
            f1 = registro["f1_score_fraude"]
            psi = registro["psi_valor"]
            alerta = registro["alerta_disparada"]
            
            # Criterio de Aprobación MLOps:
            # Si el F1-Score supera la regla de calidad (0.50), el modelo
            # demuestra que sabe predecir con alta precisión sobre la nueva realidad.
            if f1 >= 0.50:
                print(f"[ÉXITO] ✅ El modelo APROBÓ el lote con un F1-Score de {f1:.4f} >= 0.50. Avanzando...")
                aprobado = True
            else:
                print(f"[REPROBADO] ❌ El modelo REPROBÓ el lote (F1: {f1:.4f} < 0.50 | PSI: {psi:.4f}).")
                
                if alerta:
                    print("[STREAM] Alerta enviada a la nube. Esperando solución de GitHub Actions...")
                    exito_nube = esperar_github_actions_dinamico(timestamp_disparo=timestamp_evaluacion)
                    
                    if exito_nube:
                        ejecutar_git_pull_automatico()
                        print("[STREAM] 🔄 Modelo actualizado descargado. Procediendo a RE-EVALUAR el mismo lote...")
                    else:
                        print("[STREAM] ⚠️ GitHub Actions no logró generar un modelo superior. Manteniendo modelo actual.")
                        break
                else:
                    print("[STREAM] En periodo de Cooldown FinOps o Nube Ocupada. No se enviará evento duplicado.")
                    break

if __name__ == "__main__":
    ejecutar_simulacion_stream()