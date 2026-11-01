"""Comparación piloto: static vs periodic (ventanas de tiempo) vs event.

Ejecutar desde la raíz del repositorio (véase README_EXPERIMENTO.md).
Este runner es LOCAL y no dispara GitHub Actions: mide la política y el AutoML
compartido. El costo de CI/CD real debe cuantificarse en una fase separada.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.auto_tune import evaluar_y_seleccionar_mejor_modelo  # noqa: E402

POLICIES = ("static", "periodic", "event")


def scores(model, frame):
    x = frame.drop(columns=["Class"])
    y = frame["Class"].astype(int)
    predicted = model.predict(x)
    return {
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "false_negatives": int(((y == 1) & (predicted == 0)).sum()),
        "false_positives": int(((y == 0) & (predicted == 1)).sum()),
    }


def load_model(manifest):
    return joblib.load(ROOT / manifest["modelo_activo"] if not Path(manifest["modelo_activo"]).is_absolute()
                       else Path(manifest["modelo_activo"]))


def train(df_train, directory, trials):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_snapshot.csv"
    df_train.to_csv(path, index=False)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    manifest = evaluar_y_seleccionar_mejor_modelo(
        ruta_train=str(path),
        n_trials=trials,
        umbral_guardrail=0.0,
        directorio_modelos=str(directory),
    )
    if manifest is None:
        raise RuntimeError("AutoML no generó un manifiesto")
    # La ruta en el manifiesto es relativa al working dir; resolverla sin tocar models/.
    return manifest, time.perf_counter() - start_wall, time.process_time() - start_cpu


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/experiments/chronological")
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments/results")
    parser.add_argument("--trials", type=int, default=1, help="Trials por algoritmo; 1 para smoke test")
    parser.add_argument("--every", type=int, default=4,
                        help="Frecuencia periódica en ventanas temporales iguales")
    parser.add_argument("--f1-trigger", type=float, default=0.50)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    args = parser.parse_args()
    if args.trials < 1 or args.every < 1:
        parser.error("--trials y --every deben ser positivos")
    if not 0 <= args.f1_trigger <= 1 or not 0 <= args.min_improvement <= 1:
        parser.error("Los valores de F1 deben estar entre 0 y 1")
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    config = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    train_initial = pd.read_csv(input_dir / "baseline_train.csv")
    holdout = pd.read_csv(input_dir / "promotion_holdout.csv")
    windows = [(w["window"], pd.read_csv(input_dir / w["file"])) for w in config["windows"]]
    features = list(train_initial.columns)
    for name, frame in [("holdout", holdout)] + [(f"window {i}", f) for i, f in windows]:
        if list(frame.columns) != features:
            raise ValueError(f"Esquema incompatible en {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Salida no vacía: {output_dir}; usa --output distinto")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configuration.json").write_text(json.dumps({
        "input_manifest": config,
        "trials": args.trials,
        "every": args.every,
        "f1_trigger": args.f1_trigger,
        "min_improvement": args.min_improvement,
        "protocol": "prequential: evaluate future window before adding labels to training",
        "limitations": [
            "Labels immediately available after each window",
            "No actual GitHub Actions latency/cost measured",
            "Promotion holdout is historic, not independently refreshed over time",
            "AutoML internal train/validation uses stratified random split",
            "Single pilot run; no statistical inference",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[BASELINE] Entrenando UNA vez para las tres políticas...")
    common_manifest, base_wall, base_cpu = train(
        train_initial, output_dir / "initial", args.trials
    )
    base_model = load_model(common_manifest)
    base_score = scores(base_model, holdout)
    print(f"[BASELINE] F1 holdout histórico={base_score['f1']:.4f}; "
          f"tiempo={base_wall:.1f}s")
    rows = []
    for policy in POLICIES:
        print(f"\n[POLICY] {policy}", flush=True)
        history = train_initial.copy()
        model = base_model
        active_algorithm = common_manifest["modelo_activo"]
        cumulative_train_wall = 0.0
        cumulative_train_cpu = 0.0
        n_retrains = 0
        for window_index, window in windows:
            model_used_this_window = active_algorithm
            start_eval = time.perf_counter()
            metrics = scores(model, window)
            eval_wall = time.perf_counter() - start_eval
            event = policy == "event" and metrics["f1"] < args.f1_trigger
            periodic = policy == "periodic" and window_index % args.every == 0
            triggered = event or periodic
            # Se evalúa primero sobre datos no vistos; luego quedan disponibles sus etiquetas.
            history = pd.concat([history, window], ignore_index=True)
            train_wall, train_cpu, promoted = 0.0, 0.0, False
            candidate_algorithm, candidate_holdout_f1 = "", ""
            if triggered:
                n_retrains += 1
                candidate_manifest, train_wall, train_cpu = train(
                    history, output_dir / policy / f"window_{window_index:02d}",
                    args.trials,
                )
                candidate = load_model(candidate_manifest)
                candidate_algorithm = candidate_manifest["modelo_activo"]
                candidate_holdout_f1 = scores(candidate, holdout)["f1"]
                incumbent_holdout_f1 = scores(model, holdout)["f1"]
                # Guardrail experimental adicional: debe superar al activo en holdout.
                # El holdout histórico no se usa para medir desempeño futuro del sistema.
                promoted = candidate_holdout_f1 >= incumbent_holdout_f1 + args.min_improvement
                if promoted:
                    model = candidate
                    active_algorithm = candidate_algorithm
                cumulative_train_wall += train_wall
                cumulative_train_cpu += train_cpu
            row = {
                "policy": policy,
                "window": window_index,
                "records": len(window),
                "synthetic_shift": config["windows"][window_index - 1]["synthetic_shift"],
                **metrics,
                "eval_wall_seconds": round(eval_wall, 6),
                "retraining_triggered": triggered,
                "candidate_promoted": promoted,
                "candidate_model": candidate_algorithm,
                "candidate_holdout_f1": candidate_holdout_f1,
                "active_model_for_window": model_used_this_window,
                "active_model_after_window": active_algorithm,
                "training_wall_seconds": round(train_wall, 6),
                "training_process_cpu_seconds": round(train_cpu, 6),
                "cumulative_training_wall_seconds": round(cumulative_train_wall, 6),
                "cumulative_training_process_cpu_seconds": round(cumulative_train_cpu, 6),
                "retrainings_cumulative": n_retrains,
            }
            rows.append(row)
            print(f" {window_index}: F1={metrics['f1']:.4f}, "
                  f"trigger={triggered}, promote={promoted}, "
                  f"train={train_wall:.1f}s", flush=True)
        save_csv(output_dir / f"{policy}.csv", [r for r in rows if r["policy"] == policy])
    save_csv(output_dir / "comparison.csv", rows)
    print(f"\n[OK] Resultados piloto en: {output_dir}")
    print("[NOTA] No afirmar ahorro cloud ni superioridad estadística con este piloto.")


if __name__ == "__main__":
    main()
