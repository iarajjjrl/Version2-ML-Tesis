import os
import joblib
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def ejecutar_autoajuste_y_gobernanza(ruta_train="data/datos_entrenamiento.csv", umbral_guardrail=0.50):
    print("[INFO] Cargando lote de datos en memoria...")
    if not os.path.exists(ruta_train):
        raise FileNotFoundError(f"No se encontró el archivo '{ruta_train}'")
        
    df = pd.read_csv(ruta_train)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    def funcion_objetivo(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 350, step=50),
            'max_depth': trial.suggest_int('max_depth', 6, 16),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 40, step=5),
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        modelo = HistGradientBoostingClassifier(**params)
        modelo.fit(X_train, y_train)
        
        # En lugar de usar predict() rígido a 0.50, evaluamos probabilidades
        y_probs = modelo.predict_proba(X_val)[:, 1]
        
        # Encontrar el mejor umbral de decisión
        mejores_f1s = [f1_score(y_val, (y_probs >= th).astype(int), pos_label=1) for th in np.arange(0.20, 0.80, 0.05)]
        return max(mejores_f1s)
        
    print("[INFO] Iniciando autoajuste bayesiano con Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(funcion_objetivo, n_trials=15)
    
    mejor_f1 = study.best_value
    mejores_params = study.best_params
    mejores_params['class_weight'] = 'balanced'
    mejores_params['random_state'] = 42
    
    print(f"\n[INFO] Mejor F1-Score (Fraude) alcanzado: {mejor_f1:.4f}")
    print(f"[INFO] Parámetros seleccionados: {mejores_params}")
    
    if mejor_f1 >= umbral_guardrail:
        print(f"[GUARDRAIL APROBADO] F1-Score ({mejor_f1:.4f}) >= Umbral Mínimo ({umbral_guardrail}).")
        
        modelo_final = HistGradientBoostingClassifier(**mejores_params)
        modelo_final.fit(X, y)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(modelo_final, "models/modelo_vigente.pkl")
        print("[INFO] Modelo actualizado y guardado exitosamente en 'models/modelo_vigente.pkl'.")
    else:
        print(f"[ERROR] Guardrail Rechazado: F1-Score ({mejor_f1:.4f}) < Umbral Mínimo ({umbral_guardrail}).")
        raise ValueError("El modelo generado no cumple con las políticas de riesgo.")

if __name__ == "__main__":
    ejecutar_autoajuste_y_gobernanza()