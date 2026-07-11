import time
import subprocess
from generate_synthetic import generar_lote_sintetico
from monitor import evaluar_lote_produccion
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def esperar_github_actions_dinamico(intervalo=10, timeout_minutos=5):
    """
    Consulta activamente la API de GitHub para saber cuándo termina 
    la ejecución del workflow efímero sin usar tiempos fijos.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    
    if not token:
        print("[AUTO-POLL] ⚠️ No se encontró GITHUB_TOKEN en el .env. Usando pausa fija de 90s...")
        time.sleep(90)
        return False
        
    url = f"https://api.github.com/repos/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print("\n[AUTO-POLL] Conectando con GitHub API para monitorear el reentrenamiento...")
    # Ventana de gracia corta para que GitHub registre el evento entrante
    time.sleep(5) 
    
    inicio_tiempo = time.time()
    while (time.time() - inicio_tiempo) < (timeout_minutos * 60):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                runs = response.json().get("workflow_runs", [])
                if runs:
                    ultima_run = runs[0]
                    status = ultima_run.get("status")          # queued, in_progress, completed
                    conclusion = ultima_run.get("conclusion")  # success, failure, startup_failure
                    run_id = ultima_run.get("id")
                    
                    if status in ["queued", "in_progress"]:
                        print(f"[AUTO-POLL] El pipeline en la nube está: {status.upper()}... Reintentando en {intervalo}s. (Run ID: {run_id})")
                    elif status == "completed":
                        print(f"[AUTO-POLL] 🏁 ¡GitHub Actions ha terminado! Conclusión: {conclusion.upper()}")
                        return conclusion == "success"
            else:
                print(f"[AUTO-POLL] Código de respuesta inesperado de GitHub API: {response.status_code}")
        except Exception as e:
            print(f"[AUTO-POLL] Error de conexión con la API: {e}")
            
        time.sleep(intervalo)
        
    print("[AUTO-POLL] ⏱️ Timeout alcanzado esperando el reentrenamiento.")
    return False

def ejecutar_git_pull_automatico():
    """Ejecuta git pull mediante Python para actualizar el modelo local."""
    print("\n[AUTO-PULL] Descargando el nuevo modelo reentrenado desde GitHub...")
    try:
        resultado = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[AUTO-PULL] ✅ Repositorio actualizado con éxito:")
        print(f"            {resultado.stdout.strip()}")
    except Exception as e:
        print(f"[AUTO-PULL] ❌ Error al ejecutar git pull: {str(e)}")

def ejecutar_streaming_sintetico():
    print("==========================================================")
    print("  SIMULADOR DE INGESTA TRANSACCIONAL SINTÉTICA (MLOps)    ")
    print("==========================================================\n")
    
    pasos_simulacion = [
        {"paso": 1, "drift": False, "descripcion": "Lote Sintético Normal - Operación Habitual"},
        {"paso": 2, "drift": True,  "descripcion": "Lote Sintético CON DRIFT - Ataque de Fraude Evasivo"},
        {"paso": 3, "drift": False, "descripcion": "Lote Sintético Post-Reentrenamiento (Validación)"},
    ]
    
    for estado in pasos_simulacion:
        print(f"\n[STREAM] --- SIMULANDO VENTANA #{estado['paso']}: {estado['descripcion']} ---")
        
        # 1. Generar la data sintética en tiempo real
        ruta_lote = generar_lote_sintetico(
            ruta_base_csv="data/datos_entrenamiento.csv",
            num_muestras=4000,
            proporcion_fraude=0.025,
            inyectar_drift=estado['drift']
        )
        
        # 2. Evaluar el modelo vigente contra la data recién generada
        evaluar_lote_produccion(ruta_lote, "models/modelo_vigente.pkl")
        
        # Si acabamos de simular el lote con Drift (Paso 2), activamos el sondeo dinámico
        if estado['drift']:
            print("\n[STREAM] Lanzando monitor dinámico de infraestructura efímera...")
            
            # Reemplazo de la pausa fija por la llamada al sondeo dinámico de la API
            exito_nube = esperar_github_actions_dinamico(intervalo=10)
            
            if exito_nube:
                # 3. Traer el nuevo modelo actualizado desde GitHub de forma automática si la nube fue exitosa
                ejecutar_git_pull_automatico()
            else:
                print("\n[AUTO-PULL] ❌ GitHub Actions falló o el Guardrail de Gobernanza rechazó el modelo en la nube.")
                print("[AUTO-PULL] Manteniendo el modelo actual en producción por resiliencia y continuidad del servicio.")
        else:
            print("[STREAM] Pausa de 10 segundos antes de la siguiente ingesta...")
            time.sleep(10)

if __name__ == "__main__":
    ejecutar_streaming_sintetico()