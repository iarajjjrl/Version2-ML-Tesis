"""Piloto reproducible de mantenimiento de modelos: static, periodic y event.

Evalúa cada ventana futura ANTES de usar sus etiquetas para reentrenar.
Comparación local: NO mide despliegue ni costos de GitHub Actions.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.auto_tune import evaluar_y_seleccionar_mejor_modelo  # noqa: E402
from src.experiments.policies import (  # noqa: E402
    POLICIES, approve_promotion, decide_update,
)


def scores(model, frame):
    x = frame.drop(columns=["Class"])
    y = frame["Class"].astype(int)
    pred = model.predict(x)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "false_negatives": int(((y == 1) & (pred == 0)).sum()),
        "false_positives": int(((y == 0) & (pred == 1)).sum()),
        "positives": int((y == 1).sum()),
    }


def load_model(manifest):
    path = Path(manifest["modelo_activo"])
    if not path.is_absolute():
        path = ROOT / path
    return joblib.load(path)


def train(df_train, directory, trials, seed):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_snapshot.csv"
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    df_train.to_csv(path, index=False)
    manifest = evaluar_y_seleccionar_mejor_modelo(
        ruta_train=str(path),
        n_trials=trials,
        umbral_guardrail=0.0,  # El filtro comparativo se aplica en approve_promotion.
        directorio_modelos=str(directory),
        seed=seed,
        validation_mode="chronological",
    )
    if not manifest:
        raise RuntimeError("AutoML no generó manifiesto de candidato")
    return manifest, time.perf_counter() - start_wall, time.process_time() - start_cpu


def write_csv(target, rows):
    if not rows:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Comparación piloto de políticas MLOps")
    p.add_argument("--input", type=Path, default=ROOT / "data/experiments/chronological")
    p.add_argument("--output", type=Path, default=ROOT / "data/experiments/results-v2")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--every", type=int, default=4,
                   help="Periodicidad en ventanas de duración temporal equivalente")
    p.add_argument("--f1-trigger", type=float, default=0.50)
    p.add_argument("--consecutive-low", type=int, default=1,
                   help="Ventanas consecutivas con F1 bajo antes de disparar")
    p.add_argument("--cooldown-windows", type=int, default=0,
                   help="Ventanas de espera tras cada intento de reentrenamiento")
    p.add_argument("--promotion-fraction", type=float, default=0.30,
                   help="Última fracción de ventana para evaluar promoción independiente")
    p.add_argument("--min-improvement", type=float, default=0.01,
                   help="Mejora mínima de F1 en promoción frente al modelo vigente")
    p.add_argument("--min-promote-f1", type=float, default=0.0,
                   help="Suelo de F1 en promoción; 0=sin suelo en piloto")
    args = p.parse_args()
    if args.trials < 1 or args.every < 1 or args.consecutive_low < 1 or args.cooldown_windows < 0:
        p.error("Trials, frecuencia y consecutivas deben ser positivos; cooldown >= 0")
    if not all(0 <= x <= 1 for x in
               (args.f1_trigger, args.min_improvement, args.min_promote_f1)):
        p.error("F1 y mejora deben estar entre 0 y 1")
    if not 0.1 <= args.promotion_fraction <= 0.5:
        p.error("promotion-fraction debe ser entre 0.10 y 0.50")

    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Carpeta de salida no vacía: {output}")
    config = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(source / "baseline_train.csv")
    initial_holdout = pd.read_csv(source / "promotion_holdout.csv")
    windows = [(w["window"], pd.read_csv(source / w["file"]))
               for w in config["windows"]]
    columns = list(baseline.columns)
    for name, frame in [("holdout", initial_holdout)] + [
        (f"window {i}", data) for i, data in windows
    ]:
        if list(frame.columns) != columns:
            raise ValueError(f"Esquema de características inconsistente en {name}")
        if frame["Class"].nunique() != 2:
            raise ValueError(f"El lote {name} no contiene ambas clases")
    output.mkdir(parents=True, exist_ok=True)
    details = {
        "source_manifest": config,
        "parameters": {
            "seed": args.seed, "trials": args.trials, "every": args.every,
            "f1_trigger": args.f1_trigger,
            "consecutive_low": args.consecutive_low,
            "cooldown_windows": args.cooldown_windows,
            "promotion_fraction": args.promotion_fraction,
            "min_improvement": args.min_improvement,
            "min_promote_f1": args.min_promote_f1,
        },
        "evaluation": "test-then-train by chronological windows",
        "promotion": "chronologically last fraction of the JUST-OBSERVED window, excluded from candidate training",
        "timing": "local elapsed seconds and own-process CPU seconds; excludes CI/CD, cloud, monitoring overhead",
        "limitations": [
            "Class labels are assumed available at end of each window.",
            "No GitHub Actions or Docker deployment time measured.",
            "No full-system CPU/RAM/cloud billing measured.",
            "AutoML tuning reuses its own internal validation subset for threshold selection.",
            "Only one 8-window time series per run; further scenarios and statistical validation needed.",
            "Initial preprocessing statistics remain fixed even after model retraining.",
        ],
    }
    (output / "configuration.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("[BASELINE] Entrenando el mismo modelo inicial para las tres estrategias...", flush=True)
    initial_manifest, initial_wall, initial_cpu = train(
        baseline, output / "initial", args.trials, args.seed
    )
    initial_model = load_model(initial_manifest)
    initial_score = scores(initial_model, initial_holdout)
    print(f"[BASELINE] F1 holdout histórico={initial_score['f1']:.4f}; "
          f"costo inicial={initial_wall:.2f}s (común a todas)", flush=True)
    (output / "baseline_metrics.json").write_text(
        json.dumps({
            "initial_model": initial_manifest["modelo_activo"],
            "initial_holdout_metrics": initial_score,
            "initial_training_wall_seconds": initial_wall,
            "initial_training_process_cpu_seconds": initial_cpu,
            "common_to_all_policies": True,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows = []
    for policy in POLICIES:
        print(f"\n[POLICY] {policy}", flush=True)
        history = baseline.copy()
        model = initial_model
        active_name = initial_manifest["modelo_activo"]
        last_trigger = None
        low_streak = 0
        cumulative_wall = cumulative_cpu = 0.0
        cumulative_attempts = cumulative_promotions = 0
        for index, frame in windows:
            before = active_name
            t0 = time.perf_counter()
            metric = scores(model, frame)
            monitor_wall = time.perf_counter() - t0
            trigger, low_streak, reason = decide_update(
                policy, index, metric["f1"], args.f1_trigger,
                args.every, args.consecutive_low, args.cooldown_windows,
                low_streak, last_trigger,
            )
            attempt_wall = attempt_cpu = 0.0
            promoted = False
            candidate_name = ""
            candidate_f1 = incumbent_f1 = ""
            delta_f1 = ""
            if trigger:
                last_trigger = index
                cumulative_attempts += 1
                split = int(len(frame) * (1.0 - args.promotion_fraction))
                # Última sección CRONOLÓGICA de esta ventana no entra al ajuste
                # ni a Optuna del candidato, pero puede usarse en ciclos futuros.
                adaptation = frame.iloc[:split].copy()
                promotion = frame.iloc[split:].copy()
                if adaptation["Class"].nunique() != 2 or promotion["Class"].nunique() != 2:
                    raise ValueError(f"Ventana {index}: adaptación/promoción sin ambas clases")
                candidate_data = pd.concat([history, adaptation], ignore_index=True)
                new_manifest, attempt_wall, attempt_cpu = train(
                    candidate_data, output / policy / f"window_{index:02d}",
                    args.trials, args.seed
                )
                challenger = load_model(new_manifest)
                candidate_name = new_manifest["modelo_activo"]
                candidate_f1 = scores(challenger, promotion)["f1"]
                incumbent_f1 = scores(model, promotion)["f1"]
                delta_f1 = candidate_f1 - incumbent_f1
                promoted = approve_promotion(
                    candidate_f1, incumbent_f1,
                    args.min_improvement, args.min_promote_f1
                )
                if promoted:
                    model = challenger
                    active_name = candidate_name
                    cumulative_promotions += 1
                cumulative_wall += attempt_wall
                cumulative_cpu += attempt_cpu

            # Siempre después de evaluar y de decidir. En una ventana futura
            # todos los datos observados quedan disponibles para cada política.
            history = pd.concat([history, frame], ignore_index=True)
            row = {
                "policy": policy, "window": index,
                "records": len(frame),
                "synthetic_shift": config["windows"][index - 1]["synthetic_shift"],
                **metric,
                "monitor_wall_seconds": round(monitor_wall, 6),
                "trigger": trigger,
                "trigger_reason": reason,
                "candidate_promoted": promoted,
                "model_before_window": before,
                "model_after_window": active_name,
                "candidate_model": candidate_name,
                "incumbent_promotion_f1": incumbent_f1,
                "candidate_promotion_f1": candidate_f1,
                "promotion_delta_f1": delta_f1,
                "training_wall_seconds": round(attempt_wall, 6),
                "training_process_cpu_seconds": round(attempt_cpu, 6),
                "cumulative_training_wall_seconds": round(cumulative_wall, 6),
                "cumulative_training_process_cpu_seconds": round(cumulative_cpu, 6),
                "retrainings_cumulative": cumulative_attempts,
                "promotions_cumulative": cumulative_promotions,
            }
            rows.append(row)
            print(f"  ventana={index} F1={metric['f1']:.4f}, "
                  f"retraining={trigger}, promote={promoted}, "
                  f"reason={reason}, train={attempt_wall:.1f}s", flush=True)

        write_csv(output / f"{policy}.csv", [
            r for r in rows if r["policy"] == policy
        ])
    write_csv(output / "comparison.csv", rows)
    print(f"\n[OK] {output / 'comparison.csv'}")
    print("[AVISO] Piloto: sin evidencia todavía de ahorro cloud ni superioridad estadística.")


if __name__ == "__main__":
    main()
