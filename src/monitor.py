import os
import time
import hashlib
import json
import sys
import subprocess
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from dotenv import load_dotenv

# Al ejecutar `python src/monitor.py`, Python prioriza `src/` en sys.path.
# Los artefactos serializados referencian `src.monitor.ModelWithThreshold`,
# por lo que el directorio raíz del proyecto debe estar disponible al cargarlos.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

class ModelWithThreshold:
    """
    Wrapper de producción con observabilidad de algoritmo y umbral dinámico.
    """
    def __init__(self, base_model, threshold=0.5, name="Modelo"):
        self.base_model = base_model
        self.threshold = threshold
        self.name = name
        
    def predict(self, X):
        if hasattr(self.base_model, "predict_proba"):
            probs = self.base_model.predict_proba(X)[:, 1]
            return (probs >= self.threshold).astype(int)
        return self.base_model.predict(X)

def obtener_huella_modelo(ruta_modelo="models/modelo_vigente.pkl"):
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

def disparar_reentrenamiento_github(ruta_lote_causante=None):
    if hay_reentrenamiento_activo_en_nube():
        print("[NUBE OCUPADA]  GitHub Actions ya tiene una ejecución activa. Omitiendo disparo duplicado.")
        return True

    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "iarajjjrl/Version2-ML-Tesis")
    if not token:
        print("[ERROR] No se encontró GITHUB_TOKEN en .env.")
        return False

    if not ruta_lote_causante:
        print("[ERROR] No se recibió la ruta del lote que causó el drift.")
        return False

    ruta_absoluta = os.path.abspath(ruta_lote_causante)
    ruta_relativa = os.path.relpath(ruta_absoluta, PROJECT_ROOT)
    if os.path.isabs(ruta_relativa) or ruta_relativa.startswith(".."):
        print("[ERROR] El lote debe estar dentro del repositorio para poder publicarlo.")
        return False

    try:
        subprocess.run(
            ["git", "add", "--", ruta_relativa],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Auto-upload lote drift"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"[ERROR] No se pudo publicar el lote en GitHub: {error}")
        return False
        
    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "event_type": "data_drift_detected",
        "client_payload": {"ruta_drift": ruta_relativa.replace(os.sep, "/")},
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        return response.status_code in (200, 204)
    except requests.RequestException as error:
        print(f"[ERROR] Falló el envío del evento a GitHub API: {error}")
        return False

def evaluar_lote_produccion(ruta_lote_csv, modelo_path="models/modelo_vigente.pkl", ruta_base_train="data/datos_entrenamiento.csv", umbral_f1=0.50, umbral_psi=0.25):
    inicio_ttd = time.time()

    # Si existe el manifiesto, usar el artefacto inmutable que declaró como activo.
    ruta_manifiesto = "models/active_model.json"
    if modelo_path == "models/modelo_vigente.pkl" and os.path.exists(ruta_manifiesto):
        with open(ruta_manifiesto, "r", encoding="utf-8") as archivo:
            manifiesto = json.load(archivo)
        modelo_activo = manifiesto.get("modelo_activo")
        if modelo_activo:
            modelo_path = modelo_activo
    
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en '{modelo_path}'")
        
    hash_modelo, fecha_mod_modelo = obtener_huella_modelo(modelo_path)
    
    df_lote = pd.read_csv(ruta_lote_csv)
    X_test = df_lote.drop(columns=['Class'])
    y_test = df_lote['Class']
    
    modelo = joblib.load(modelo_path)
    
    # Observabilidad: Extraer metadata del modelo activo
    nombre_algoritmo = getattr(modelo, "name", modelo.__class__.__name__)
    umbral_calibrado = getattr(modelo, "threshold", 0.50)
    
    y_pred = modelo.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    variables_psi = [
        "TransactionAmt",
        "card1",
        "C13",
        "C_sum_velocity",
        "amt_to_mean_card1",
    ]
    psi_por_variable = {}

    if os.path.exists(ruta_base_train):
        df_base = pd.read_csv(ruta_base_train)

        for variable in variables_psi:
            if variable not in df_base.columns or variable not in X_test.columns:
                print(f"[PSI OMITIDO] Falta la variable '{variable}' en la base histórica o el lote.")
                continue

            psi_por_variable[variable] = calcular_psi(
                df_base[variable],
                X_test[variable],
            )

    psi_valor = max(psi_por_variable.values(), default=0.0)
    variables_con_drift = [
        variable
        for variable, psi in psi_por_variable.items()
        if psi > umbral_psi
    ]
            
    ttd_segundos = time.time() - inicio_ttd
    alerta_disparada = False
    
    print(f"\n[INFO] Evaluando telemetría sobre el lote: {ruta_lote_csv}")
    print(f"[MODELO ACTIVO]  {nombre_algoritmo} | Umbral Calibrado: {umbral_calibrado:.2f} | SHA256: [{hash_modelo}] | Compilado: {fecha_mod_modelo}")
    print(f"[RESULTADO] F1-Score (Fraude): {f1:.4f} | Umbral Mínimo: {umbral_f1}")
    print(f"[RESULTADO] Population Stability Index (PSI): {psi_valor:.4f} | Umbral PSI: {umbral_psi}")
    print(f"[RESULTADO] Drift Detection Time (TTD): {ttd_segundos:.6f} segundos")
    
    # El modelo requiere reentrenamiento si su F1-Score cae por debajo de la regla de calidad (0.50)
    for variable, psi in psi_por_variable.items():
        print(
            f"[RESULTADO] PSI {variable}: {psi:.4f} "
            f"| Umbral PSI: {umbral_psi}"
        )

    hay_drift_f1 = f1 < umbral_f1
    hay_drift_psi = bool(variables_con_drift)
    hay_drift = hay_drift_f1 or hay_drift_psi

    if hay_drift:
        if hay_drift_f1:
            print(f"[ALERTA] Degradación detectada (F1: {f1:.4f} < {umbral_f1}).")
        if hay_drift_psi:
            print(
                "[ALERTA] Drift estadístico detectado en: "
                + ", ".join(variables_con_drift)
            )

        en_cooldown = consultar_ultimo_cooldown_alerta()
        if en_cooldown:
            print("[COOLDOWN] Alerta previa enviada recientemente (< 180s). Omitiendo disparo duplicado a GitHub API.")
        else:
            exito = disparar_reentrenamiento_github(ruta_lote_csv)
            if exito:
                print("[INFO] Evento 'data_drift_detected' procesado correctamente.")
                alerta_disparada = True
            else:
                print("[ERROR] Falló el envío del evento a GitHub API.")
    else:
        print("[INFO] Estado del modelo estable. Rendimiento dentro de parámetros.")
        
    os.makedirs("logs", exist_ok=True)
    ruta_log = "logs/telemetria_experimento.csv"
    
    registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lote_evaluado": ruta_lote_csv,
        "algoritmo": nombre_algoritmo,
        "umbral_calibrado": round(umbral_calibrado, 2),
        "modelo_sha256": hash_modelo,
        "f1_score_fraude": round(f1, 4),
        "psi_valor": round(psi_valor, 4),
        "ttd_segundos": round(ttd_segundos, 6),
        "alerta_disparada": alerta_disparada,
        "psi_por_variable": json.dumps(psi_por_variable),
        "variables_con_drift": json.dumps(variables_con_drift),
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
