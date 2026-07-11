import os
import sys
import joblib
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# Asegurar la ruta raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitor import ModelWithThreshold

optuna.logging.set_verbosity(optuna.logging.WARNING)

def cargar_datos_entrenamiento_consolidados(ruta_base="data/datos_entrenamiento.csv"):
    if not os.path.exists(ruta_base):
        raise FileNotFoundError(f"No se encontró el archivo de entrenamiento en '{ruta_base}'")
        
    df_base = pd.read_csv(ruta_base)
    
    # FEEDBACK LOOP: Incorporar lotes de produccion adicionales si existen
    lotes_drift = ["data/prod_lote2_drift.csv"]
    lotes_incorporados = []
    
    for ruta in lotes_drift:
        if os.path.exists(ruta):
            df_drift = pd.read_csv(ruta)
            df_base = pd.concat([df_base, df_drift], ignore_index=True)
            lotes_incorporados.append(ruta)
            
    if lotes_incorporados:
        print(f"[FEEDBACK LOOP MLOPS] Incorporando {len(lotes_incorporados)} lote(s) real(es) de drift al reentrenamiento.")
        # Guardar dataset consolidado actualizado
        df_base.to_csv(ruta_base, index=False)
        
    return df_base

def encontrar_umbral_optimo(modelo, X_val, y_val):
    probs = modelo.predict_proba(X_val)[:, 1]
    umbrales = np.linspace(0.15, 0.85, 71)
    mejor_f1 = 0.0
    mejor_umbral = 0.50
    
    for th in umbrales:
        preds = (probs >= th).astype(int)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if score > mejor_f1:
            mejor_f1 = score
            mejor_umbral = th
            
    return mejor_f1, mejor_umbral

def evaluar_y_seleccionar_mejor_modelo(ruta_train="data/datos_entrenamiento.csv", umbral_guardrail=0.50, n_trials=5):
    print("="*60)
    print("  AUTOTUNE MLOPS: BÚSQUEDA BAYESIANA ÁGIL (FEEDBACK LOOP)")
    print("="*60)
    
    df = cargar_datos_entrenamiento_consolidados(ruta_train)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # OPTUNA TRIAL 1: HistGradientBoosting
    # ---------------------------------------------------------
    print(f"\n[OPTUNA 1/2] Optimizando HistGradientBoosting ({n_trials} trials)...")
    
    def objective_hgb(trial):
        lr = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 250)
        max_depth = trial.suggest_int("max_depth", 6, 14)
        
        model = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=max_iter, max_depth=max_depth,
            class_weight='balanced', random_state=42
        )
        model.fit(X_train, y_train)
        f1, _ = encontrar_umbral_optimo(model, X_val, y_val)
        return f1

    study_hgb = optuna.create_study(direction="maximize")
    study_hgb.optimize(objective_hgb, n_trials=n_trials)
    
    best_params_hgb = study_hgb.best_params
    best_hgb = HistGradientBoostingClassifier(**best_params_hgb, class_weight='balanced', random_state=42)
    best_hgb.fit(X_train, y_train)
    f1_hgb, th_hgb = encontrar_umbral_optimo(best_hgb, X_val, y_val)
    print(f"  -> Mejor F1-Score HistGradientBoosting: {f1_hgb:.4f} (Umbral: {th_hgb:.2f})")

    # ---------------------------------------------------------
    # OPTUNA TRIAL 2: RandomForest
    # ---------------------------------------------------------
    print(f"\n[OPTUNA 2/2] Optimizando RandomForest ({n_trials} trials)...")
    
    def objective_rf(trial):
        n_est = trial.suggest_int("n_estimators", 60, 120)
        max_depth = trial.suggest_int("max_depth", 8, 15)
        
        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            class_weight='balanced_subsample', n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        f1, _ = encontrar_umbral_optimo(model, X_val, y_val)
        return f1

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=n_trials)
    
    best_params_rf = study_rf.best_params
    best_rf = RandomForestClassifier(**best_params_rf, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    best_rf.fit(X_train, y_train)
    f1_rf, th_rf = encontrar_umbral_optimo(best_rf, X_val, y_val)
    print(f"  -> Mejor F1-Score RandomForest: {f1_rf:.4f} (Umbral: {th_rf:.2f})")

    # ---------------------------------------------------------
    # COMPARATIVA Y SELECCIÓN DEL MODELO GANADOR
    # ---------------------------------------------------------
    modelos = [
        ("HistGradientBoosting (Optuna)", best_hgb, f1_hgb, th_hgb),
        ("RandomForest (Optuna)", best_rf, f1_rf, th_rf)
    ]
    
    modelos.sort(key=lambda x: x[2], reverse=True)
    nombre_ganador, modelo_ganador, f1_ganador, th_ganador = modelos[0]
    
    print("\n" + "="*60)
    print("  TABLA COMPARATIVA DE RENDIMIENTO (FEEDBACK LOOP)")
    print("="*60)
    for nombre, _, f1_val, th_val in modelos:
        marca = "🏆 GANADOR" if nombre == nombre_ganador else "  "
        print(f"  [{marca}] {nombre:<30} | F1-Score: {f1_val:.4f} | Umbral: {th_val:.2f}")
    print("="*60)
    
    # ---------------------------------------------------------
    # EVALUACIÓN DEL GUARDRAIL DE GOBERNANZA
    # ---------------------------------------------------------
    print(f"\n[GUARDRAIL] Evaluando regla de calidad (F1 >= {umbral_guardrail})...")
    if f1_ganador >= umbral_guardrail:
        print(f"[GUARDRAIL APROBADO] ✅ El modelo '{nombre_ganador}' alcanzó un F1-Score de {f1_ganador:.4f}.")
        
        modelo_ganador.fit(X, y)
        artefacto_final = ModelWithThreshold(base_model=modelo_ganador, threshold=th_ganador)
        
        os.makedirs("models", exist_ok=True)
        ruta_modelo = "models/modelo_vigente.pkl"
        joblib.dump(artefacto_final, ruta_modelo, compress=3)
        print(f"[ARTEFACTO] Modelo óptimo guardado exitosamente en '{ruta_modelo}'.")
    else:
        print(f"[GUARDRAIL RECHAZADO] ❌ Ningún modelo superó el umbral de {umbral_guardrail}. F1 máximo: {f1_ganador:.4f}.")
        raise ValueError(f"Reentrenamiento abortado por Guardrail (F1 {f1_ganador:.4f} < {umbral_guardrail})")

if __name__ == "__main__":
    evaluar_y_seleccionar_mejor_modelo()