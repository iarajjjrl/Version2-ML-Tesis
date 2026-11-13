"""Estudio B: preparación y evaluación LOCAL de prueba temporal sellada.

Esta herramienta no modifica src/prepare_data.py, models/, GitHub Actions,
watcher ni main. Distingue generalización fuera de muestra de hot-swap funcional.

Preparar => entrena SOLO un preprocesador con historia inicial; crea un lote
nominal, un lote causal para adaptación y FUTUROS sellados, sin solapamiento.
Local-run => entrena dos versiones con auto_tune.py sobre una copia desechable
de la historia, y las puntúa en las mismas transacciones futuras. NO despliega.
Evaluar => puntúa dos modelos serializados confiables sin disparar alertas.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.experiments.prepare_windows import FEATURES, read_source
from src.prepare_data import FraudFeaturePreprocessor


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _ids_hash(frame):
    ids = frame["TransactionID"].astype(str).tolist()
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _shift(frame, seed):
    """Covariate shift artificial: NO implica concept drift real."""
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for col in ("TransactionAmt", "C13"):
        if col in result:
            value = pd.to_numeric(result[col], errors="coerce")
            result[col] = value * rng.uniform(2.0, 3.0, len(result))
    return result


def prepare_from_dataframe(source, output, *, max_rows=120000, seed=42):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"No sobrescribir evidencia: {out}")
    required = {"TransactionDT", "TransactionID", "isFraud", "TransactionAmt", "card1"}
    if not required.issubset(source):
        raise ValueError(f"Fuente incompleta: {sorted(required - set(source))}")
    if len(source) < 1000 or max_rows < 1000:
        raise ValueError("Se necesitan al menos 1000 observaciones")
    ordered = pd.MultiIndex.from_frame(source[["TransactionDT", "TransactionID"]])
    if not ordered.is_monotonic_increasing:
        raise ValueError("Datos origen deben venir en orden cronológico")
    if source.TransactionID.duplicated().any():
        raise ValueError("TransactionID duplicado")
    if len(source) > max_rows:
        indices = np.linspace(0, len(source) - 1, max_rows, dtype=np.int64)
        source = source.iloc[indices].reset_index(drop=True)
    else:
        source = source.reset_index(drop=True)
    n = len(source)
    bounds = (0, int(n * .50), int(n * .60), int(n * .80), n)
    splits = {
        "baseline": source.iloc[bounds[0]:bounds[1]].copy(),
        "nominal": source.iloc[bounds[1]:bounds[2]].copy(),
        "adaptation": source.iloc[bounds[2]:bounds[3]].copy(),
        "future": source.iloc[bounds[3]:bounds[4]].copy(),
    }
    times = []
    visited = set()
    for name, frame in splits.items():
        if frame.empty or len(frame.isFraud.unique()) != 2:
            raise ValueError(f"Bloque {name} sin ambas clases")
        if visited & set(frame.TransactionID):
            raise ValueError(f"Solapamiento de identificadores en {name}")
        visited.update(frame.TransactionID)
        times.append((frame.iloc[0].TransactionDT, frame.iloc[0].TransactionID,
                      frame.iloc[-1].TransactionDT, frame.iloc[-1].TransactionID))
    for first, next_ in zip(times, times[1:]):
        if tuple(first[2:]) >= tuple(next_[:2]):
            raise ValueError("La siguiente partición no es cronológicamente posterior")

    features = [col for col in FEATURES if col in source.columns]
    pre = FraudFeaturePreprocessor().fit(splits["baseline"][features].copy())
    scenarios = {
        "baseline_train.csv": splits["baseline"],
        "nominal.csv": splits["nominal"],
        "adaptation_shifted.csv": _shift(splits["adaptation"], seed + 101),
        "future_natural.csv": splits["future"],
        "future_shifted.csv": _shift(splits["future"], seed + 202),
    }
    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    provenance = {}
    for name, raw in scenarios.items():
        exported = pre.transform(raw[features].copy())
        exported["Class"] = raw.isFraud.astype(int).to_numpy()
        file = out / name
        exported.to_csv(file, index=False)
        group = name.split("_")[0].split(".")[0]
        base = splits["future" if name.startswith("future") else
                      "adaptation" if name.startswith("adaptation") else
                      "baseline" if name.startswith("baseline") else "nominal"]
        provenance[name] = {
            "sha256": sha256(file),
            "n": len(raw),
            "positives": int(raw.isFraud.sum()),
            "start_key": [float(base.iloc[0].TransactionDT),
                          int(base.iloc[0].TransactionID)],
            "end_key": [float(base.iloc[-1].TransactionDT),
                        int(base.iloc[-1].TransactionID)],
            "transaction_ids_sha256": _ids_hash(base),
            "synthetic_covariate_shift": "shifted" in name,
        }
    joblib.dump(pre, out / "preprocessor_baseline_only.pkl", compress=3)
    manifest = {
        "protocol": "study-b-new-independent-chronological-split-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "IEEE-CIS Fraud Detection",
        "selected_records": n,
        "fractions": {"baseline": .50, "nominal": .10,
                      "adaptation": .20, "future_sealed": .20},
        "features": features,
        "seed": seed,
        "preprocessor_sha256": sha256(out / "preprocessor_baseline_only.pkl"),
        "files": provenance,
        "notes": [
            "Solo el baseline participa en el fit del preprocesador",
            "Los modelos pueden entrenarse con baseline y adaptation, nunca con future",
            "Los dos futuros comparten IDs y etiquetas; son escenarios emparejados, no dos muestras independientes",
            "La perturbación altera covariables sin cambiar Class: NO demuestra concept drift",
            "No colocar estos CSV futuros en data/ingesta_viva: el watcher puede disparar reentrenamiento"
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _verify_sealed(prepared):
    base = Path(prepared)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    for filename, detail in manifest["files"].items():
        location = base / filename
        if not location.is_file() or sha256(location) != detail["sha256"]:
            raise ValueError(f"Archivo alterado desde preparación: {filename}")
    if sha256(base / "preprocessor_baseline_only.pkl") != manifest["preprocessor_sha256"]:
        raise ValueError("Preprocesador modificado")
    return manifest


def evaluate_pair(prepared, model_before, model_after, *, output):
    base = Path(prepared)
    manifest = _verify_sealed(base)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Ya existe el informe: {destination}")
    model_before = Path(model_before)
    model_after = Path(model_after)
    before_hash = sha256(model_before)
    after_hash = sha256(model_after)
    if before_hash == after_hash:
        raise ValueError("Ambos modelos tienen la misma huella; no se puede atribuir cambio")
    # Solo cargar artefactos propios/de confianza: los .pkl pueden ejecutar código.
    incumbent = joblib.load(model_before)
    challenger = joblib.load(model_after)
    report = {
        "protocol": manifest["protocol"],
        "independence": ("El lote fue sellado antes del ajuste en el protocolo de preparación. "
                         "La independencia del modelo externo debe acreditarse mediante su "
                         "origen de entrenamiento; pasar modelos arbitrarios no la demuestra."),
        "preprocessing": "fitted_on_baseline_only",
        "model_before": {"path": str(model_before), "sha256": before_hash,
                         "family": getattr(incumbent, "name", type(incumbent).__name__)},
        "model_after": {"path": str(model_after), "sha256": after_hash,
                        "family": getattr(challenger, "name", type(challenger).__name__)},
        "future_results": {},
    }
    for filename in ("future_natural.csv", "future_shifted.csv"):
        frame = pd.read_csv(base / filename)
        labels = frame["Class"].astype(int).to_numpy()
        X = frame.drop(columns=["Class"])
        results = {}
        for label, model in (("before", incumbent), ("after", challenger)):
            prediction = np.asarray(model.predict(X)).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
            results[label] = {
                "f1": float(f1_score(labels, prediction, zero_division=0)),
                "precision": float(precision_score(labels, prediction, zero_division=0)),
                "recall": float(recall_score(labels, prediction, zero_division=0)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            }
        report["future_results"][filename] = {
            "dataset_sha256": manifest["files"][filename]["sha256"],
            "n": len(frame), "positives": int(labels.sum()), **results,
            "f1_delta_after_minus_before": (results["after"]["f1"] -
                                             results["before"]["f1"]),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return report


def local_run(prepared, *, trials=1, seed=42):
    """Reproduce selección B local sobre temporalidad sellada. NO es GitHub Actions."""
    if trials < 1:
        raise ValueError("trials debe ser >=1")
    base = Path(prepared)
    manifest = _verify_sealed(base)
    work = base / "local_validation"
    if work.exists():
        raise FileExistsError("local_validation ya existe: nunca sobrescribir el experimento")
    work.mkdir(parents=True)
    training_copy = work / "working_training.csv"
    shutil.copyfile(base / "baseline_train.csv", training_copy)
    from src.auto_tune import evaluar_y_seleccionar_mejor_modelo

    baseline = evaluar_y_seleccionar_mejor_modelo(
        ruta_train=str(training_copy),
        ruta_lote_drift=None, umbral_guardrail=0.,
        n_trials=trials, directorio_modelos=str(work / "model_before"), seed=seed)
    adapted = evaluar_y_seleccionar_mejor_modelo(
        ruta_train=str(training_copy),
        ruta_lote_drift=str(base / "adaptation_shifted.csv"),
        umbral_guardrail=0.,
        n_trials=trials, directorio_modelos=str(work / "model_after"), seed=seed)
    report = evaluate_pair(base, baseline["modelo_activo"], adapted["modelo_activo"],
                           output=work / "comparison_future.json")
    evidence = {
        "mode": "LOCAL_NOT_GITHUB_ACTIONS_OR_DOCKER",
        "seed": seed, "trials_per_family": trials,
        "baseline_training_sha256": manifest["files"]["baseline_train.csv"]["sha256"],
        "adaptation_sha256": manifest["files"]["adaptation_shifted.csv"]["sha256"],
        "future_natural_sha256": manifest["files"]["future_natural.csv"]["sha256"],
        "future_shifted_sha256": manifest["files"]["future_shifted.csv"]["sha256"],
        "initial_artifact": baseline,
        "adapted_artifact": adapted,
        "guardrail_used_for_LOCAL_comparison": 0.0,
        "notes": [
            "Guardrail=0.0 permite comparar incluso candidatos peores: NO demuestra que habrían sido promovidos en producción",
            "Validación interna de auto_tune.py sigue siendo estratificada, no temporal",
            "Futuro queda fuera de entrenamiento y selección; dos variantes futuras comparten transacciones",
            "El estudio B histórico de Docker sigue siendo prueba funcional separada"
        ],
        "report": str(work / "comparison_future.json"),
    }
    (work / "provenance.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Validación B temporal sin tocar watcher, GitHub o models/")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--transactions", default="train_transaction.csv")
    prep.add_argument("--identity", default="train_identity.csv")
    prep.add_argument("--output", required=True)
    prep.add_argument("--max-rows", type=int, default=120000)
    prep.add_argument("--seed", type=int, default=42)
    comparison = commands.add_parser("evaluate")
    comparison.add_argument("--prepared", required=True)
    comparison.add_argument("--before", required=True)
    comparison.add_argument("--after", required=True)
    comparison.add_argument("--output", required=True)
    local = commands.add_parser("local-run")
    local.add_argument("--prepared", required=True)
    local.add_argument("--trials", type=int, default=1)
    local.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        df = read_source(Path(args.transactions), Path(args.identity))
        result = prepare_from_dataframe(
            df, args.output, max_rows=args.max_rows, seed=args.seed)
        print(f"[OK] Estudio B preparado: {result['selected_records']} filas, "
              "lotes cronológicos disjuntos y futuro sellado")
    elif args.command == "evaluate":
        result = evaluate_pair(args.prepared, args.before, args.after,
                               output=args.output)
        print(json.dumps(result["future_results"], indent=2, ensure_ascii=False))
    else:
        result = local_run(args.prepared, trials=args.trials, seed=args.seed)
        print("[OK] Prueba B LOCAL, sin hot-swap de Docker ni GitHub Actions")
        print(json.dumps(result["future_results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
