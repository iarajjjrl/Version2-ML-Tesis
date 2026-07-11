import os
import time
import hashlib
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from dotenv import load_dotenv

load_dotenv()

class ModelWithThreshold:
    """
    Wrapper de producción que aplica el umbral óptimo de probabilidad
    al ejecutar predict(), manteniendo compatibilidad total con joblib.
    """
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold
        
    def predict(self, X):
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)
        return self.base_model.predict(X)

def obtener_huella_modelo(ruta_modelo="models/modelo_vigente.pkl"):
    """
    Calcula la firma digital (SHA256) y el timestamp del archivo .pkl.
    """
    if not os.path.exists(ruta_modelo):
        return "INEXISTENTE", "N/A"
    
    try:
        hasher = hashlib.sha256()
        with open(ruta_modelo, 'rb') as f:
            hasher.update(f.read())
        hash_sha256 = hasher.hexdigest()[:12]
        
        mtime = os.path.getmtime(ruta_modelo)
        fecha_mod = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        return hash_sha256, fecha_mod
    except Exception:
        return "ERROR_HASH", "N/A"

def calcular_psi(esperado, actual, num_buckets=10):
    """
    Calcula el Índice de Estabilidad de la Población (PSI).
    """
    try:
        eps = 1e-4
        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(esperado, percentiles)
        buckets[0] = -np.inf
        buckets[-1] = np.inf
        
        counts_esp, _ = np.histogram(esperado, bins=buckets)
        counts_act, _ = np.histogram(actual, bins=buckets)
        
        pct_esp = counts_esp / len(esperado)
        pct_act = counts_act / len(actual)
        
        pct_esp = np.where(pct_esp == 0, eps, pct_esp)
        pct_act = np.where(pct_act == 0, eps, pct_act)
        
        psi_val = np.sum((pct_act - pct_esp) * np.log(pct_act / pct_esp))
        return float(psi_val)
    except Exception:
        return 0.0

def consultar_ultimo_cooldown_alerta(ruta_log="logs/telemetria_experimento.csv", cooldown_segundos=180):
    if not os.path.exists(ruta_log):
        return False
        
    try:
        df_log = pd.read_csv(ruta_log)
        if df_log.empty or 'alerta_disparada' not in df_log.columns:
            return False
            
        alertas = df_log[df_log['alerta_disparada'].astype(str).str.lower().isin(['true', '1'])]
        if alertas.empty:
            return False
            
        ultima_alerta_str = alertas.iloc[-1]['timestamp']
        ultima_alerta_dt = datetime.strptime(ultima_alerta_str, "%Y-%m-%d %H:%M:%S")
        
        if datetime.now() - ultima_alerta_dt < timedelta(seconds=cooldown_segundos):
            return True
    except Exception:
        pass
    return False

def hay_reentrenamiento_activo_en_nube():
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    if not token:
        return False
        
    url = f"https://api.github.com/repos/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            runs = response.json().get("workflow_runs", [])
            for run in runs:
                if run.get("status") in ["queued", "in_progress"]:
                    return True
    except Exception:
        pass
    return False

def disparar_reentrenamiento_github():
    if hay_reentrenamiento_activo_en_nube():
        print("[NUBE OCUPADA] ⏳ GitHub Actions ya tiene una ejecución activa. Omitiendo disparo duplicado.")
        return True

    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    if not token:
        print("[ERROR] No se encontró GITHUB_TOKEN en .env.")
        return False
        
    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"event_type": "data_drift_detected"}
    response = requests.post(url, json=data, headers=headers)
    return response.status_code in [200, 204]

def evaluar_lote_produccion(ruta_lote_csv, modelo_path="models/modelo_vigente.pkl", ruta_base_train="data/datos_entrenamiento.csv", umbral_f1=0.50, umbral_psi=0.25):
    inicio_ttd = time.time()
    
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en '{modelo_path}'")
        
    hash_modelo, fecha_mod_modelo = obtener_huella_modelo(modelo_path)
    
    df_lote = pd.read_csv(ruta_lote_csv)
    X_test = df_lote.drop(columns=['Class'])
    y_test = df_lote['Class']
    
    modelo = joblib.load(modelo_path)
    y_pred = modelo.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    psi_valor = 0.0
    if os.path.exists(ruta_base_train):
        df_base = pd.read_csv(ruta_base_train)
        if 'TransactionAmt' in df_base.columns and 'TransactionAmt' in X_test.columns:
            psi_valor = calcular_psi(df_base['TransactionAmt'], X_test['TransactionAmt'])
            
    ttd_segundos = time.time() - inicio_ttd
    alerta_disparada = False
    
    print(f"\n[INFO] Evaluando telemetría sobre el lote: {ruta_lote_csv}")
    print(f"[MODELO ACTIVO] SHA256: [{hash_modelo}] | Compilado/Actualizado: {fecha_mod_modelo}")
    print(f"[RESULTADO] F1-Score (Fraude): {f1:.4f} | Umbral Mínimo: {umbral_f1}")
    print(f"[RESULTADO] Population Stability Index (PSI): {psi_valor:.4f} | Umbral PSI: {umbral_psi}")
    print(f"[RESULTADO] Drift Detection Time (TTD): {ttd_segundos:.6f} segundos")
    
    hay_drift = (f1 < umbral_f1) or (psi_valor > umbral_psi)
    
    if hay_drift:
        print(f"[ALERTA] ⚠️ Degradación detectada (F1: {f1:.4f} < {umbral_f1} o PSI: {psi_valor:.4f} > {umbral_psi}).")
        
        en_cooldown = consultar_ultimo_cooldown_alerta()
        if en_cooldown:
            print("[COOLDOWN] Alerta previa enviada recientemente (< 180s). Omitiendo disparo duplicado a GitHub API.")
        else:
            exito = disparar_reentrenamiento_github()
            if exito:
                print("[INFO] Evento 'data_drift_detected' procesado correctamente.")
                alerta_disparada = True
            else:
                print("[ERROR] Falló el envío del evento a GitHub API.")
    else:
        print("[INFO] Estado del modelo estable. Rendimiento y PSI dentro de parámetros.")
        
    os.makedirs("logs", exist_ok=True)
    ruta_log = "logs/telemetria_experimento.csv"
    
    registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lote_evaluado": ruta_lote_csv,
        "modelo_sha256": hash_modelo,
        "f1_score_fraude": round(f1, 4),
        "psi_valor": round(psi_valor, 4),
        "ttd_segundos": round(ttd_segundos, 6),
        "alerta_disparada": alerta_disparada
    }
    
    df_nuevo = pd.DataFrame([registro])
    if not os.path.exists(ruta_log):
        df_nuevo.to_csv(ruta_log, index=False)
    else:
        df_nuevo.to_csv(ruta_log, mode='a', header=False, index=False)
        
    print(f"[INFO] Telemetría registrada correctamente en '{ruta_log}'.")
    return registro

if __name__ == "__main__":
    lote_prueba = "data/prod_lote1_normal.csv"
    if os.path.exists(lote_prueba):
        evaluar_lote_produccion(lote_prueba)