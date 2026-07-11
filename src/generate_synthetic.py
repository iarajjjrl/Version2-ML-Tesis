import os
import sys
import time
import requests
import subprocess
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# Asegurar la ruta raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitor import evaluar_lote_produccion, obtener_huella_modelo

load_dotenv()

def esperar_github_actions_dinamico(timestamp_disparo, timeout_segundos=300, intervalo_poll=8):
    """
    Monitorea dinámicamente la API de GitHub Actions filtrando únicamente
    por ejecuciones creadas posterior al timestamp_disparo de la alerta.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    
    if not token:
        print("[AUTO-POLL] ❌ No hay GITHUB_TOKEN en .env. No se puede monitorear la nube.")
        return False
        
    url = f"https://api.github.com/repos/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print("\n[AUTO-POLL] ⏳ Conectando con GitHub API. Esperando registro de la nueva ejecución en la nube...")
    tiempo_inicio = time.time()
    
    while time.time() - tiempo_inicio < timeout_segundos:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                runs = data.get("workflow_runs", [])
                
                # Filtrar solo ejecuciones creadas DESPUÉS del disparo
                runs_nuevos = []
                for r in runs:
                    created_at_str = r.get("created_at")
                    if created_at_str:
                        dt_created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        ts_created = dt_created.timestamp()
                        if ts_created >= (timestamp_disparo - 10):
                            runs_nuevos.append(r)
                
                if runs_nuevos:
                    latest_run = runs_nuevos[0]
                    status = latest_run.get("status")
                    conclusion = latest_run.get("conclusion")
                    run_id = latest_run.get("id")
                    
                    if status in ["queued", "in_progress"]:
                        print(f"[AUTO-POLL] 🚀 Trabajo activo en la nube: {status.upper()}... (Run ID: {run_id})")
                    elif status == "completed":
                        print(f"[AUTO-POLL] 🏁 ¡GitHub Actions ha finalizado! Conclusión: {conclusion.upper()} (Run ID: {run_id})")
                        if conclusion == "success":
                            print("[AUTO-POLL] ⏸️ Pausa de 5 segundos para consolidar la actualización en GitHub...")
                            time.sleep(5)
                            return True
                        else:
                            return False
                else:
                    print("[AUTO-POLL] ⏳ Aguardando a que la API de GitHub registre el evento...")
        except Exception as e:
            print(f"[AUTO-POLL] Error de conexión con GitHub API: {e}")
            
        time.sleep(intervalo_poll)
        
    print("[AUTO-POLL] ⏱️ Timeout alcanzado esperando a GitHub Actions.")
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
        
        # BUCLE DE EXAMEN Y RE-EXAMEN (MLOps Loop)
        aprobado = False
        intentos = 0
        max_intentos = 3
        
        while not aprobado and intentos < max_intentos:
            intentos += 1
            if intentos > 1:
                print(f"\n[RE-EXAMEN] 🔄 Evaluando nuevamente la VENTANA #{i} tras reentrenamiento (Intento {intentos})...")
            
            timestamp_evaluacion = datetime.now(timezone.utc).timestamp()
            
            # Rendir examen sobre el lote de producción
            registro = evaluar_lote_produccion(ruta_lote)
            
            f1 = registro["f1_score_fraude"]
            psi = registro["psi_valor"]
            alerta = registro["alerta_disparada"]
            
            # Criterio de Aprobación
            if f1 >= 0.50 and psi <= 0.25:
                print(f"[ÉXITO] ✅ El modelo APROBÓ el lote (F1: {f1:.4f} >= 0.50, PSI: {psi:.4f} <= 0.25). Avanzando...")
                aprobado = True
            else:
                print(f"[REPROBADO] ❌ El modelo REPROBÓ el lote (F1: {f1:.4f}, PSI: {psi:.4f}).")
                
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