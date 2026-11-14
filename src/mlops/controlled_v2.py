"""Prototipo MLOps controlado con el protocolo EXACTO del Estudio A v2.

No toca models/active_model.json, no despliega en main y no llama a GitHub.
Entrada: carpetas raw-v2 generadas con src.experiments.v2.raw_windows.
Salida: estado, candidatos y telemetría separados en data/experiments/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import joblib
import pandas as pd

from src.experiments.v2.controller import PolicySettings
from src.experiments.v2.run_comparison_v2 import _read_inputs
from src.experiments.v2.trainer import save_bundle, train_candidate
from src.mlops.controller_bridge import (
    atomic_json, complete_observation, create_state,
    exclusive_lock, load_state,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(store, name):
    # Las rutas no proceden de un usuario externo: son nombres controlados.
    allowed = "baseline" if name == "baseline" else name
    if allowed != "baseline" and not allowed.startswith("event_candidate_after_"):
        raise ValueError("Identificador de modelo no permitido")
    if "/" in allowed or "\\" in allowed or ".." in allowed:
        raise ValueError("Ruta de artefacto no permitida")
    return Path(store) / "models" / (allowed + ".pkl")


def _load_bundle(store, state, model_id):
    artifact = state["metadata"]["artifacts"].get(model_id)
    if not artifact:
        raise FileNotFoundError(f"Modelo no registrado: {model_id}")
    location = _artifact_path(store, model_id)
    if not location.is_file() or sha256_file(location) != artifact["sha256"]:
        raise ValueError(f"El artefacto cambió o desapareció: {model_id}")
    return joblib.load(location)  # Únicamente artefactos propios de la sesión.


def bootstrap(raw_dir, store, *, seed=42, trials=3, f1_trigger=.50,
              consecutive_low=2, cooldown_windows=1, min_delta=.01):
    """Entrena M0 una sola vez con historia inicial; no activa candidatos."""
    raw = Path(raw_dir).resolve()
    destination = Path(store).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("El directorio de estado debe estar vacío")
    baseline, holdout, windows, manifest, features = _read_inputs(raw)
    if len(windows) < 3:
        raise ValueError("Se necesitan al menos tres ventanas futuras")
    settings = PolicySettings(
        policy="event", n_windows=len(windows), f1_trigger=f1_trigger,
        consecutive_low=consecutive_low, cooldown_windows=cooldown_windows,
        min_delta=min_delta, trigger_mode="absolute")
    start = time.perf_counter()
    model = train_candidate(baseline, features=features, trials=trials, seed=seed)
    duration = time.perf_counter() - start
    diagnostic = model.evaluate(holdout)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = save_bundle(model, _artifact_path(destination, "baseline"))
    metadata = {
        "raw_dir": str(raw),
        "source_manifest_sha256": sha256_file(raw / "manifest.json"),
        "dataset": manifest.get("dataset"),
        "seed": seed,
        "trials_per_family": trials,
        "features": features,
        "inputs_sha256": {
            fname: sha256_file(raw / fname)
            for fname in (["baseline_train_raw.csv", "diagnostic_holdout_raw.csv"] +
                          [entry["file"] for entry in manifest["windows"]])
        },
        "artifacts": {"baseline": bundle},
        "initial_training_seconds": round(duration, 4),
        "diagnostic_holdout": diagnostic,
        "mode": "local_isolated_no_cloud_publication",
    }
    state = create_state(settings, "baseline", metadata)
    atomic_json(destination / "state.json", state)
    return {
        "store": str(destination),
        "n_windows": len(windows),
        "initial_family": model.family,
        "initial_f1_diagnostic": diagnostic["f1"],
        "initial_seconds": duration,
        "active_model_id": "baseline",
    }


def step(store, batch_file=None):
    """Consume la SIGUIENTE ventana del manifiesto en orden cronológico.

    k usa el modelo vigente; si dispara, entrena candidato con historia previa
    y la ventana k. k+1 evalúa candidato vs vigente SIN reentrenarlos usando
    sus etiquetas. Un candidato aprobado se usa para puntuación desde k+2.
    """
    directory = Path(store).resolve()
    with exclusive_lock(directory):
        state, controller = load_state(directory / "state.json")
        meta = state["metadata"]
        raw = Path(meta["raw_dir"])
        if sha256_file(raw / "manifest.json") != meta["source_manifest_sha256"]:
            raise ValueError("El manifiesto del escenario fue modificado")
        manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
        window = controller.last_window + 1
        if window > controller.settings.n_windows:
            raise ValueError("El escenario ya completó todas sus ventanas")
        entry = manifest["windows"][window - 1]
        if entry["window"] != window:
            raise ValueError("El manifiesto contiene ventanas fuera de orden")
        batch_path = raw / entry["file"]
        if sha256_file(batch_path) != meta["inputs_sha256"][entry["file"]]:
            raise ValueError("La ventana preparada cambió desde la inicialización")
        if batch_file is not None:
            supplied = Path(batch_file).resolve()
            if sha256_file(supplied) != sha256_file(batch_path):
                raise ValueError("El lote recibido no coincide con la ventana esperada")
            batch_path = supplied
        batch = pd.read_csv(batch_path)
        if len(batch) != entry["records"] or len(batch) < 2:
            raise ValueError("Tamaño de ventana distinto del manifiesto")
        pair = pd.MultiIndex.from_frame(batch[["TransactionDT", "TransactionID"]])
        if not pair.is_monotonic_increasing or batch["TransactionID"].duplicated().any():
            raise ValueError("La ventana no está ordenada o tiene IDs duplicados")
        if window > 1:
            previous = manifest["windows"][window - 2]
            prev_frame = pd.read_csv(raw / previous["file"], usecols=["TransactionDT", "TransactionID"])
            if tuple(batch[["TransactionDT", "TransactionID"]].iloc[0]) <= tuple(
                    prev_frame[["TransactionDT", "TransactionID"]].iloc[-1]):
                raise ValueError("La ventana se solapa con la anterior")
        incumbent = _load_bundle(directory, state, controller.active_model)
        metrics = incumbent.evaluate(batch)
        pending = controller.pending
        challenger_metrics = None
        if pending:
            challenger = _load_bundle(directory, state, pending[1])
            challenger_metrics = challenger.evaluate(batch)
        perf = {"training_seconds": 0.0, "training_cpu_seconds": 0.0}

        def train_callback(index):
            previous_files = ["baseline_train_raw.csv", "diagnostic_holdout_raw.csv"]
            previous_files.extend(item["file"] for item in manifest["windows"][:index - 1])
            for name in previous_files:
                if sha256_file(raw / name) != meta["inputs_sha256"][name]:
                    raise ValueError(f"Los datos históricos fueron modificados: {name}")
            history = pd.concat([pd.read_csv(raw / name) for name in previous_files],
                                ignore_index=True)
            t0, cpu0 = time.perf_counter(), time.process_time()
            candidate = train_candidate(
                history, features=meta["features"], trials=meta["trials_per_family"],
                seed=meta["seed"], adaptation_window=batch)
            perf["training_seconds"] = time.perf_counter() - t0
            perf["training_cpu_seconds"] = time.process_time() - cpu0
            candidate_id = f"event_candidate_after_{index:02d}"
            bundle = save_bundle(candidate, _artifact_path(directory, candidate_id))
            meta["artifacts"][candidate_id] = bundle
            return candidate_id

        event = complete_observation(
            state, controller, window=window,
            observed_f1=metrics["f1"],
            candidate_f1=(challenger_metrics["f1"] if challenger_metrics else None),
            incumbent_f1=(metrics["f1"] if challenger_metrics else None),
            train_callback=train_callback,
            batch_sha256=sha256_file(batch_path))
        event.update({
            "scenario": manifest.get("scenario", "no_declarado"),
            "epoch": manifest.get("epoch"),
            "source_file": entry["file"],
            "transactions": len(batch),
            "positives": int(batch["Class"].sum()),
            "family": incumbent.family,
            "precision": metrics["precision"], "recall": metrics["recall"],
            "tp": metrics["tp"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tn": metrics["tn"],
            "challenger_metrics": challenger_metrics,
            **perf,
        })
        # La metadata también se actualiza cuando el callback guarda candidato.
        state["processed"][-1] = event
        atomic_json(directory / "state.json", state)
        return event


def main():
    parser = argparse.ArgumentParser(description="MLOps controlado, separado del modelo publicado")
    commands = parser.add_subparsers(dest="command", required=True)
    boot = commands.add_parser("bootstrap", help="Crear estado y modelo inicial M0")
    boot.add_argument("--raw-dir", required=True)
    boot.add_argument("--store", required=True)
    boot.add_argument("--seed", type=int, default=42)
    boot.add_argument("--trials", type=int, default=3)
    boot.add_argument("--f1-trigger", type=float, default=.50)
    boot.add_argument("--consecutive-low", type=int, default=2)
    boot.add_argument("--cooldown-windows", type=int, default=1)
    boot.add_argument("--min-delta", type=float, default=.01)
    run = commands.add_parser("step", help="Procesar la siguiente ventana del escenario")
    run.add_argument("--store", required=True)
    run.add_argument("--batch", default=None,
                     help="CSV recibido, debe coincidir con la siguiente ventana preparada")
    status = commands.add_parser("status", help="Mostrar resumen del estado persistido")
    status.add_argument("--store", required=True)
    args = parser.parse_args()
    if args.command == "bootstrap":
        result = bootstrap(args.raw_dir, args.store, seed=args.seed, trials=args.trials,
                           f1_trigger=args.f1_trigger,
                           consecutive_low=args.consecutive_low,
                           cooldown_windows=args.cooldown_windows,
                           min_delta=args.min_delta)
    elif args.command == "step":
        result = step(args.store, batch_file=args.batch)
    else:
        data, ctrl = load_state(Path(args.store) / "state.json")
        result = {"last_window": ctrl.last_window, "active_model": ctrl.active_model,
                  "pending": ctrl.pending, "attempts": ctrl.attempts,
                  "promotions": ctrl.promotions, "processed": len(data["processed"])}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
