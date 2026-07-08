import pandas as pd
import numpy as np
import optuna
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.INFO)

print("[INFO] Cargando lote de datos en memoria...")
df = pd.read_csv("data/datos_entrenamiento.csv")
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def funcion_objetivo(trial):
    # Espacio de busqueda optimizado para alta precision en fraude
    max_iter = trial.suggest_int('max_iter', 100, 300, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 12)
    learning_rate = trial.suggest_float('learning_rate', 0.03, 0.30, log=True)
    l2_regularization = trial.suggest_float('l2_regularization', 1e-2, 5.0, log=True)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50, step=10)
    
    modelo = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42
    )
    
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_val)
    
    return f1_score(y_val, predicciones, pos_label=1)

def ejecutar_autoajuste_y_gobernanza():
    print("[INFO] Iniciando autoajuste bayesiano con Optuna...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(funcion_objetivo, n_trials=10)
    
    f1_optimizado = study.best_value
    print(f"\n[INFO] Mejor F1-Score (Fraude) alcanzado: {f1_optimizado:.4f}")
    print(f"[INFO] Parametros seleccionados: {study.best_params}")
    
    # Umbral regulatorio de la organizacion
    UMBRAL_REGULATORIO_F1 = 0.75
    os.makedirs("models", exist_ok=True)
    
    if f1_optimizado >= UMBRAL_REGULATORIO_F1:
        print("[INFO] Guardrail Aprobado: El modelo recupero la capacidad de detectar fraude.")
        
        modelo_final = HistGradientBoostingClassifier(
            **study.best_params, class_weight='balanced', random_state=42
        )
        modelo_final.fit(X_train, y_train)
        
        joblib.dump(modelo_final, "models/modelo_vigente.pkl")
        print("[INFO] Artefacto guardado exitosamente en 'models/modelo_vigente.pkl'")
    else:
        print("[ERROR] Guardrail Rechazado: El reentrenamiento no supero el umbral minimo.")
        raise ValueError("El modelo generado no cumple con las politicas de riesgo.")

if __name__ == "__main__":
    ejecutar_autoajuste_y_gobernanza()