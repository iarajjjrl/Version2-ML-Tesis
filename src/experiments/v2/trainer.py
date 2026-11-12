"""AutoML temporal para Estudio A v2; NO modifica models/ ni el pipeline B.

Tuning: bloque tune, F1 con umbral fijo 0.50.
Calibración de umbral: bloque calibrate, sin usarlo en Optuna.
Evaluación de promoción: ventana cronológica POSTERIOR, fuera de este módulo.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.prepare_data import FraudFeaturePreprocessor
from src.experiments.v2.temporal import split_temporally, fit_transform_blocks

optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIES = ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "HistGradientBoosting")
THRESHOLDS = tuple(float(x) for x in np.linspace(0.15, 0.85, 71))


def classification_counts(y, predictions):
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
    }


@dataclass
class ModelBundle:
    family: str
    estimator: object
    preprocessor: object
    features: tuple
    threshold: float
    seed: int
    tuning_f1: float
    calibration_f1: float
    search_results: list = field(default_factory=list)

    def probabilities(self, frame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Faltan características: {sorted(missing)}")
        x = self.preprocessor.transform(frame.loc[:, list(self.features)].copy())
        return np.asarray(self.estimator.predict_proba(x))[:, 1]

    def predict(self, frame):
        return (self.probabilities(frame) >= self.threshold).astype(int)

    def evaluate(self, frame):
        if "Class" not in frame:
            raise ValueError("Falta Class en la evaluación")
        return classification_counts(frame["Class"].astype(int).to_numpy(), self.predict(frame))


def _specs(seed, weight):
    """Mismo rango de búsqueda por familia para TODAS las políticas."""
    return [
        ("XGBoost", XGBClassifier,
         lambda t: {"learning_rate": t.suggest_float("learning_rate", .01, .30, log=True),
                    "max_depth": t.suggest_int("max_depth", 3, 10),
                    "n_estimators": t.suggest_int("n_estimators", 80, 300)},
         {"scale_pos_weight": weight, "random_state": seed, "eval_metric": "logloss",
          "n_jobs": 1, "tree_method": "hist"}),
        ("LightGBM", LGBMClassifier,
         lambda t: {"learning_rate": t.suggest_float("learning_rate", .01, .30, log=True),
                    "max_depth": t.suggest_int("max_depth", 3, 12),
                    "n_estimators": t.suggest_int("n_estimators", 80, 300)},
         {"class_weight": "balanced", "random_state": seed, "n_jobs": 1, "verbosity": -1}),
        ("CatBoost", CatBoostClassifier,
         lambda t: {"learning_rate": t.suggest_float("learning_rate", .01, .30, log=True),
                    "depth": t.suggest_int("depth", 4, 10),
                    "iterations": t.suggest_int("iterations", 80, 300)},
         {"auto_class_weights": "Balanced", "verbose": False, "random_seed": seed,
          "thread_count": 1, "allow_writing_files": False}),
        ("RandomForest", RandomForestClassifier,
         lambda t: {"max_depth": t.suggest_int("max_depth", 5, 24),
                    "n_estimators": t.suggest_int("n_estimators", 80, 300)},
         {"class_weight": "balanced_subsample", "n_jobs": 1, "random_state": seed}),
        ("HistGradientBoosting", HistGradientBoostingClassifier,
         lambda t: {"learning_rate": t.suggest_float("learning_rate", .01, .30, log=True),
                    "max_depth": t.suggest_int("max_depth", 3, 15),
                    "max_iter": t.suggest_int("max_iter", 80, 300)},
         {"class_weight": "balanced", "random_state": seed}),
    ]


def train_candidate(raw_history, *, features, trials=3, seed=42,
                    adaptation_window=None):
    """Competencia sobre historia previa; adaptación opcional a una ventana nueva.

    Si se aporta adaptation_window, el 70% cronológico inicial del lote se
    incorpora al ajuste FINAL del clasificador y preprocesador; el 30% final
    queda exclusivamente para calibrar el umbral. La promoción ocurre sobre
    la siguiente ventana, FUERA de esta función.
    """
    if trials < 1:
        raise ValueError("trials debe ser positivo")
    features = tuple(features)
    blocks = split_temporally(raw_history)
    data = fit_transform_blocks(blocks, FraudFeaturePreprocessor, features)
    for name, y in (("train", data.train_y), ("tune", data.tune_y),
                    ("calibrate", data.calibrate_y)):
        if set(y.unique()) != {0, 1}:
            raise ValueError(f"El bloque {name} debe contener ambas clases")
    positives = int((data.train_y == 1).sum())
    negatives = int((data.train_y == 0).sum())
    weight = negatives / positives
    search_results = []
    winner = None
    for family_index, (family, cls, propose, defaults) in enumerate(_specs(seed, weight)):
        def objective(trial):
            model = cls(**{**propose(trial), **defaults})
            model.fit(data.train_x, data.train_y)
            # Umbral fijo: ninguna etiqueta calibrate se consulta durante Optuna.
            pred = (model.predict_proba(data.tune_x)[:, 1] >= .50).astype(int)
            return float(f1_score(data.tune_y, pred, zero_division=0))
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed + family_index))
        study.optimize(objective, n_trials=trials)
        best = cls(**{**study.best_params, **defaults})
        best.fit(data.train_x, data.train_y)
        tune_f1 = float(f1_score(data.tune_y,
            (best.predict_proba(data.tune_x)[:, 1] >= .50).astype(int),
            zero_division=0))
        search_results.append({"family": family, "tune_f1_at_0_50": tune_f1,
                               "trials": trials, "best_params": study.best_params})
        # Orden estable para empates: la primera familia conserva prioridad.
        if winner is None or tune_f1 > winner[2]:
            winner = (family, best, tune_f1, cls, study.best_params, defaults)

    family, best_model, tune_f1, winner_class, winner_params, winner_defaults = winner
    if adaptation_window is None:
        # Baseline: calibración interna temporal que no participó en Optuna.
        final_model = best_model
        final_preprocessor = data.preprocessor
        calibration_x = data.calibrate_x
        calibration_y = data.calibrate_y
    else:
        if not isinstance(adaptation_window, pd.DataFrame) or len(adaptation_window) < 10:
            raise ValueError("Se requiere un lote de adaptación cronológico no vacío")
        if set(adaptation_window["TransactionID"]) & set(raw_history["TransactionID"]):
            raise ValueError("La adaptación se solapa con la historia previa")
        last_old = tuple(raw_history[["TransactionDT", "TransactionID"]].iloc[-1])
        first_new = tuple(adaptation_window[["TransactionDT", "TransactionID"]].iloc[0])
        if first_new <= last_old:
            raise ValueError("El lote de adaptación no es temporalmente posterior")
        if not pd.MultiIndex.from_frame(adaptation_window[
                ["TransactionDT", "TransactionID"]]).is_monotonic_increasing:
            raise ValueError("Lote de adaptación desordenado")
        split_at = int(len(adaptation_window) * .70)
        new_train = adaptation_window.iloc[:split_at]
        new_calibrate = adaptation_window.iloc[split_at:]
        if set(new_calibrate["Class"].unique()) != {0, 1}:
            raise ValueError("La calibración de adaptación requiere ambas clases")
        # Refit final con historia ya conocida + parte inicial del lote k.
        # Tuning y elección de familia se realizaron antes, solo con historia.
        final_raw = pd.concat([raw_history, new_train], ignore_index=True)
        final_preprocessor = FraudFeaturePreprocessor()
        final_preprocessor.fit(final_raw.loc[:, list(features)].copy())
        final_x = final_preprocessor.transform(final_raw.loc[:, list(features)].copy())
        calibration_x = final_preprocessor.transform(
            new_calibrate.loc[:, list(features)].copy())
        calibration_y = new_calibrate["Class"].astype(int)
        final_model = winner_class(**{**winner_params, **winner_defaults})
        final_model.fit(final_x, final_raw["Class"].astype(int))

    probabilities = final_model.predict_proba(calibration_x)[:, 1]
    scores = [(float(f1_score(calibration_y,
              (probabilities >= threshold).astype(int), zero_division=0)), threshold)
              for threshold in THRESHOLDS]
    score, threshold = max(scores, key=lambda v: (v[0], -abs(v[1] - .50), -v[1]))
    return ModelBundle(family, final_model, final_preprocessor, features,
                       threshold, int(seed), float(tune_f1), float(score),
                       search_results)

def save_bundle(bundle, destination):
    """Persiste un candidato experimental SIN activar ni sobrescribir models/."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(f"El archivo experimental ya existe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"artifact": str(path), "sha256": digest, "family": bundle.family,
            "threshold": bundle.threshold, "seed": bundle.seed,
            "tuning_f1": bundle.tuning_f1,
            "calibration_f1": bundle.calibration_f1}
