import pandas as pd
import numpy as np
import joblib
import time
import requests
import os
from datetime import datetime
from sklearn.metrics import f1_score
from dotenv import load_dotenv

# Cargar las variables configuradas en el archivo .env
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME")

UMBRAL_DRIFT_F1 = 0.75
RUTA_LOGS = "logs/telemetria_experimento.csv"

def registrar_telemetria(lote_evaluado, f1_obtenido, ttd_segundos, alerta_disparada):
    os.makedirs("logs", exist_ok=True)
    nuevo_registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lote_evaluado": lote_evaluado,
        "f1_score_fraude": round(f1_obtenido, 4),
        "ttd_segundos": round(ttd_segundos, 6),
        "alerta_disparada": alerta_disparada
    }
    df_registro = pd.DataFrame([nuevo_registro])
    if os.path.exists(RUTA_LOGS):
        df_registro.to_csv(RUTA_LOGS, mode='a', header=False, index=False)
    else:
        df_registro.to_csv(RUTA_LOGS, mode='w', header=True, index=False)
    print(f"[INFO] Telemetria registrada correctamente en '{RUTA_LOGS}'.")

def disparar_evento_github(f1_actual, lote_origen):
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    data = {
        "event_type": "data_drift_detected",
        "client_payload": {
            "f1_score": round(f1_actual, 4),
            "lote_origen": lote_origen
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 204:
            print("[INFO] Evento de alerta 'data_drift_detected' enviado con exito a GitHub API.")
        else:
            print(f"[ERROR] Fallo al notificar a GitHub API. Codigo HTTP: {response.status_code}")
            print(f"[ERROR] Respuesta de la API: {response.text}")
    except Exception as e:
        print(f"[ERROR] Excepcion durante la llamada Webhook a GitHub: {str(e)}")

def evaluar_lote_produccion(ruta_lote, modelo_path):
    print(f"\n[INFO] Evaluando telemetria de produccion sobre el lote: {ruta_lote}")
    
    if not os.path.exists(ruta_lote) or not os.path.exists(modelo_path):
        print(f"[ERROR] Archivos no encontrados para evaluar el lote.")
        return

    inicio_tiempo = time.time()
    df = pd.read_csv(ruta_lote)
    X = df.drop(columns=['Class'])
    y_real = df['Class']
    
    modelo = joblib.load(modelo_path)
    predicciones = modelo.predict(X)
    
    f1_actual = f1_score(y_real, predicciones, pos_label=1)
    ttd_segundos = time.time() - inicio_tiempo
    
    print(f"[RESULTADO] F1-Score (Fraude): {f1_actual:.4f} | Umbral: {UMBRAL_DRIFT_F1}")
    print(f"[RESULTADO] Drift Detection Time (TTD): {ttd_segundos:.6f} segundos")
    
    alerta_disparada = f1_actual < UMBRAL_DRIFT_F1
    
    if alerta_disparada:
        print(f"[ALERTA] Degradacion detectada: F1-Score ({f1_actual:.4f}) < Umbral ({UMBRAL_DRIFT_F1}).")
        disparar_evento_github(f1_actual, ruta_lote)
    else:
        print("[INFO] Estado del modelo estable. Rendimiento dentro de los parametros permitidos.")
        
    registrar_telemetria(ruta_lote, f1_actual, ttd_segundos, alerta_disparada)

if __name__ == "__main__":
    evaluar_lote_produccion("data/prod_dia1_normal.csv", "models/modelo_vigente.pkl")
    evaluar_lote_produccion("data/prod_dia2_drift.csv", "models/modelo_vigente.pkl")