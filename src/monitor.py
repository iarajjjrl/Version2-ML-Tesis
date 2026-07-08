import os
import time
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from dotenv import load_dotenv

# Cargar credenciales locales desde .env (GITHUB_TOKEN, etc.)
load_dotenv()

def consultar_ultimo_cooldown_alerta(ruta_log="logs/telemetria_experimento.csv", cooldown_segundos=180):
    """
    Verifica si ya se disparó una alerta recientemente para evitar 
    múltiples ejecuciones duplicadas en GitHub Actions (Cooldown).
    """
    if not os.path.exists(ruta_log):
        return False  # No hay historial, se permite la alerta
        
    try:
        df_log = pd.read_csv(ruta_log)
        alertas = df_log[df_log['alerta_disparada'] == True]
        if alertas.empty:
            return False
            
        ultima_alerta_str = alertas.iloc[-1]['timestamp']
        ultima_alerta_dt = datetime.strptime(ultima_alerta_str, "%Y-%m-%d %H:%M:%S")
        
        # Si la última alerta fue hace menos del tiempo de cooldown
        if datetime.now() - ultima_alerta_dt < timedelta(seconds=cooldown_segundos):
            return True  # Está en período de enfiamiento (Cooldown)
    except Exception:
        pass
    return False

def disparar_reentrenamiento_github():
    """Envía el evento Webhook HTTP a la API de GitHub Actions."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    
    if not token:
        print("[ERROR] No se encontró GITHUB_TOKEN en el archivo .env ni en las variables de entorno.")
        return False
        
    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"event_type": "data_drift_detected"}
    
    response = requests.post(url, json=data, headers=headers)
    return response.status_code in [200, 204]

def evaluar_lote_produccion(ruta_lote_csv, modelo_path="models/modelo_vigente.pkl", umbral_f1=0.75):
    """
    Evalúa el rendimiento del modelo sobre un lote de transacciones, 
    registra el tiempo de detección (TTD) y dispara alertas si hay Data Drift.
    """
    inicio_ttd = time.time()
    
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en '{modelo_path}'")
        
    df_lote = pd.read_csv(ruta_lote_csv)
    X_test = df_lote.drop(columns=['Class'])
    y_test = df_lote['Class']
    
    # Carga dinámica del modelo desde el disco (Asegura leer la versión más reciente)
    modelo = joblib.load(modelo_path)
    y_pred = modelo.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, pos_label=1)
    ttd_segundos = time.time() - inicio_ttd
    
    alerta_disparada = False
    
    print(f"\n[INFO] Evaluando telemetría sobre el lote: {ruta_lote_csv}")
    print(f"[RESULTADO] F1-Score (Fraude): {f1:.4f} | Umbral: {umbral_f1}")
    print(f"[RESULTADO] Drift Detection Time (TTD): {ttd_segundos:.6f} segundos")
    
    if f1 < umbral_f1:
        print(f"[ALERTA] Degradación detectada: F1-Score ({f1:.4f}) < Umbral ({umbral_f1}).")
        
        # Validar período de enfriamiento (Cooldown anti-spam)
        en_cooldown = consultar_ultimo_cooldown_alerta()
        if en_cooldown:
            print("[COOLDOWN] Alerta previa enviada recientemente. Omitiendo disparo duplicado a GitHub API.")
        else:
            exito = disparar_reentrenamiento_github()
            if exito:
                print("[INFO] Evento 'data_drift_detected' enviado con éxito a GitHub API.")
                alerta_disparada = True
            else:
                print("[ERROR] Falló el envío del evento a GitHub API.")
    else:
        print("[INFO] Estado del modelo estable. Rendimiento dentro de los parámetros permitidos.")
        
    # Registrar Telemetría en archivo CSV acumulativo
    os.makedirs("logs", exist_ok=True)
    ruta_log = "logs/telemetria_experimento.csv"
    
    registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lote_evaluado": ruta_lote_csv,
        "f1_score_fraude": round(f1, 4),
        "ttd_segundos": round(ttd_segundos, 6),
        "alerta_disparada": alerta_disparada
    }
    
    df_nuevo = pd.DataFrame([registro])
    if not os.path.exists(ruta_log):
        df_nuevo.to_csv(ruta_log, index=False)
    else:
        df_nuevo.to_csv(ruta_log, mode='a', header=False, index=False)
        
    print(f"[INFO] Telemetría registrada correctamente en '{ruta_log}'.")

if __name__ == "__main__":
    # Prueba individual por defecto al ejecutar este script directamente
    lote_prueba = "data/prod_dia1_normal.csv"
    if os.path.exists(lote_prueba):
        evaluar_lote_produccion(lote_prueba)