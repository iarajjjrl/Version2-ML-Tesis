import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.monitor import ModelWithThreshold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cargar_datos_entrenamiento_consolidados(
    ruta_base="data/datos_entrenamiento.csv", ruta_lote_drift=None
):
    if not os.path.exists(ruta_base):
        raise FileNotFoundError(f"No se encontró el archivo de entrenamiento en '{ruta_base}'")

    df_base = pd.read_csv(ruta_base)
    if ruta_lote_drift and os.path.exists(ruta_lote_drift):
        print(f"[FEEDBACK LOOP MLOPS] Incorporando lote '{ruta_lote_drift}'...")
        df_drift = pd.read_csv(ruta_lote_drift)
        df_base = pd.concat([df_base, df_drift], ignore_index=True)
        df_base.to_csv(ruta_base, index=False)
    else:
        print(f"[ENTRENAMIENTO BASE] Dataset cargado ({len(df_base)} filas).")
    return df_base


def encontrar_umbral_optimo(modelo, X_val, y_val):
    probabilidades = modelo.predict_proba(X_val)[:, 1]
    mejor_f1, mejor_umbral = 0.0, 0.50
    for umbral in np.linspace(0.15, 0.85, 71):
        predicciones = (probabilidades >= umbral).astype(int)
        score = f1_score(y_val, predicciones, pos_label=1, zero_division=0)
        if score > mejor_f1:
            mejor_f1, mejor_umbral = score, float(umbral)
    return mejor_f1, mejor_umbral


def _hash_archivo(ruta, longitud=8):
    hasher = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            hasher.update(bloque)
    return hasher.hexdigest()[:longitud]


def _guardar_artefacto_versionado(artefacto, nombre_modelo, f1, umbral, directorio="models"):
    os.makedirs(directorio, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    slug = re.sub(r"[^a-z0-9]+", "_", nombre_modelo.lower()).strip("_")
    temporal = os.path.join(directorio, f".{slug}_{timestamp}.tmp.pkl")

    try:
        joblib.dump(artefacto, temporal, compress=3)
        hash_corto = _hash_archivo(temporal)
        nombre_archivo = f"modelo_{slug}_{timestamp}_{hash_corto}.pkl"
        ruta_final = os.path.join(directorio, nombre_archivo)
        if os.path.exists(ruta_final):
            raise FileExistsError(f"El artefacto versionado ya existe: '{ruta_final}'")
        os.replace(temporal, ruta_final)

        manifiesto = {
            "modelo_activo": ruta_final.replace(os.sep, "/"),
            "f1_calibrado": round(float(f1), 6),
            "umbral": round(float(umbral), 6),
            "despliegue": datetime.now(timezone.utc).isoformat(),
        }
        ruta_manifiesto = os.path.join(directorio, "active_model.json")
        temporal_json = ruta_manifiesto + ".tmp"
        with open(temporal_json, "w", encoding="utf-8") as archivo:
            json.dump(manifiesto, archivo, indent=2, ensure_ascii=False)
            archivo.write("\n")
        os.replace(temporal_json, ruta_manifiesto)
        return manifiesto
    finally:
        if os.path.exists(temporal):
            os.remove(temporal)


def evaluar_y_seleccionar_mejor_modelo(
    ruta_train="data/datos_entrenamiento.csv",
    ruta_lote_drift=None,
    umbral_guardrail=0.50,
    n_trials=5,
    directorio_modelos="models",
):
    print("=" * 72)
    print("AUTOML MLOPS: COMPETENCIA DE CINCO MODELOS CON OPTUNA")
    print("=" * 72)

    df = cargar_datos_entrenamiento_consolidados(ruta_train, ruta_lote_drift)
    if "Class" not in df.columns:
        raise ValueError("El dataset debe contener la columna objetivo 'Class'.")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    if y.nunique() != 2:
        raise ValueError("La columna 'Class' debe contener exactamente dos clases.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    positivos = int((y_train == 1).sum())
    negativos = int((y_train == 0).sum())
    if positivos == 0:
        raise ValueError("No hay ejemplos positivos en el conjunto de entrenamiento.")
    scale_pos_weight = negativos / positivos

    def objective_xgb(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
        }
        model = XGBClassifier(
            **params, scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="logloss", n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return encontrar_umbral_optimo(model, X_val, y_val)[0]

    def objective_lgbm(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
        }
        model = LGBMClassifier(
            **params, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1,
        )
        model.fit(X_train, y_train)
        return encontrar_umbral_optimo(model, X_val, y_val)[0]

    def objective_catboost(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "iterations": trial.suggest_int("iterations", 80, 300),
        }
        model = CatBoostClassifier(
            **params, auto_class_weights="Balanced", verbose=0, random_state=42,
        )
        model.fit(X_train, y_train)
        return encontrar_umbral_optimo(model, X_val, y_val)[0]

    def objective_rf(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 5, 24),
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
        }
        model = RandomForestClassifier(
            **params, class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
        model.fit(X_train, y_train)
        return encontrar_umbral_optimo(model, X_val, y_val)[0]

    def objective_hist(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
        }
        model = HistGradientBoostingClassifier(
            **params, class_weight="balanced", random_state=42,
        )
        model.fit(X_train, y_train)
        return encontrar_umbral_optimo(model, X_val, y_val)[0]

    competidores = [
        ("XGBoost", XGBClassifier, objective_xgb,
         lambda p: {**p, "scale_pos_weight": scale_pos_weight, "random_state": 42,
                    "eval_metric": "logloss", "n_jobs": -1}),
        ("LightGBM", LGBMClassifier, objective_lgbm,
         lambda p: {**p, "class_weight": "balanced", "random_state": 42,
                    "n_jobs": -1, "verbosity": -1}),
        ("CatBoost", CatBoostClassifier, objective_catboost,
         lambda p: {**p, "auto_class_weights": "Balanced", "verbose": 0, "random_state": 42}),
        ("RandomForest", RandomForestClassifier, objective_rf,
         lambda p: {**p, "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": 42}),
        ("HistGradientBoosting", HistGradientBoostingClassifier, objective_hist,
         lambda p: {**p, "class_weight": "balanced", "random_state": 42}),
    ]

    resultados = []
    for nombre, clase, objective, completar_params in competidores:
        print(f"\n[OPTUNA] Optimizando {nombre} ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        modelo = clase(**completar_params(study.best_params))
        modelo.fit(X_train, y_train)
        f1, umbral = encontrar_umbral_optimo(modelo, X_val, y_val)
        resultados.append((nombre, modelo, f1, umbral))
        print(f"  -> F1-Score: {f1:.4f} | Umbral: {umbral:.2f}")

    resultados.sort(key=lambda resultado: resultado[2], reverse=True)
    nombre_ganador, modelo_ganador, f1_ganador, umbral_ganador = resultados[0]

    print("\n" + "=" * 72)
    print("TABLA COMPARATIVA DE MODELOS")
    for nombre, _, f1, umbral in resultados:
        marca = "GANADOR" if nombre == nombre_ganador else ""
        print(f"[{marca:7}] {nombre:<22} F1={f1:.4f} | umbral={umbral:.2f}")
    print("=" * 72)

    print(f"\n[GUARDRAIL] F1 mínimo requerido: {umbral_guardrail:.4f}")
    if f1_ganador < umbral_guardrail:
        print(
            "[INTERVENCIÓN MANUAL REQUERIDA] - "
            "El modelo reentrenado no superó el Guardrail de calidad"
        )
        sys.exit(0)

    print(f"[GUARDRAIL APROBADO] {nombre_ganador}: F1={f1_ganador:.4f}")
    modelo_ganador.fit(X, y)
    artefacto = ModelWithThreshold(
        base_model=modelo_ganador,
        threshold=umbral_ganador,
        name=nombre_ganador,
    )
    manifiesto = _guardar_artefacto_versionado(
        artefacto, nombre_ganador, f1_ganador, umbral_ganador,
        directorio=directorio_modelos,
    )
    print(f"[DEPLOY] Artefacto versionado: {manifiesto['modelo_activo']}")
    print(f"[DEPLOY] Manifiesto activo actualizado en {directorio_modelos}/active_model.json")
    return manifiesto


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML MLOps de cinco modelos")
    parser.add_argument("--lote-drift", type=str, default=None)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--umbral-guardrail", type=float, default=0.50)
    args = parser.parse_args()
    lote_drift_env = os.getenv("LOTE_DRIFT_PATH", args.lote_drift)
    evaluar_y_seleccionar_mejor_modelo(
        ruta_lote_drift=lote_drift_env,
        umbral_guardrail=args.umbral_guardrail,
        n_trials=args.trials,
    )
