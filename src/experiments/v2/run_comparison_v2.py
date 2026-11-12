"""Runner comparativo del Estudio A v2: ejecución LOCAL y prospectiva.

No dispara GitHub Actions, no modifica models/ y NO sustituye los resultados v1.
Ventana k: evaluar vigente ANTES de actualizar; candidato se entrena con datos
conocidos hasta k. Ventana k+1: evaluar candidato vs vigente y promocionar.
Ventana k+2: primeras predicciones del nuevo modelo ya promocionado.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.resources import ResourceSampler
from src.experiments.v2.controller import MaintenanceController, PolicySettings
from src.experiments.v2.trainer import train_candidate, save_bundle, classification_counts


def _read_inputs(input_dir):
    root = Path(input_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("preprocessed") is not False:
        raise ValueError("Estudio A v2 exige archivos CRUDOS, no CSV transformados v1")
    baseline = pd.read_csv(root / "baseline_train_raw.csv")
    holdout = pd.read_csv(root / "diagnostic_holdout_raw.csv")
    windows = [pd.read_csv(root / entry["file"]) for entry in manifest["windows"]]
    parts = [baseline, holdout, *windows]
    seen = set()
    previous = None
    for i, part in enumerate(parts):
        if part.empty or part[["TransactionDT", "TransactionID", "Class"]].isna().any().any():
            raise ValueError(f"Bloque {i} inválido o vacío")
        ordered = pd.MultiIndex.from_frame(part[["TransactionDT", "TransactionID"]])
        if not ordered.is_monotonic_increasing:
            raise ValueError(f"El bloque {i} no está ordenado cronológicamente")
        first = tuple(part[["TransactionDT", "TransactionID"]].iloc[0])
        last = tuple(part[["TransactionDT", "TransactionID"]].iloc[-1])
        if previous is not None and first <= previous:
            raise ValueError("Hay solapamiento o regresión entre bloques temporales")
        previous = last
        identifiers = set(part.TransactionID)
        if identifiers & seen or len(identifiers) != len(part):
            raise ValueError("TransactionID repetido entre bloques")
        seen.update(identifiers)
    features = manifest.get("features")
    if not features or any(col not in baseline for col in features):
        raise ValueError("Características del manifiesto incompletas")
    return baseline, holdout, windows, manifest, features


def _train(raw_history, features, trials, seed, *, profile):
    start = time.perf_counter()
    if profile:
        with ResourceSampler() as resources:
            model = train_candidate(raw_history, features=features, trials=trials, seed=seed)
        usage = resources.result()
    else:
        model = train_candidate(raw_history, features=features, trials=trials, seed=seed)
        usage = {}
    return model, time.perf_counter() - start, usage


def _pooled(counts):
    tp = int(sum(r["tp"] for r in counts))
    fp = int(sum(r["fp"] for r in counts))
    fn = int(sum(r["fn"] for r in counts))
    tn = int(sum(r["tn"] for r in counts))
    p = tp / (tp + fp) if tp + fp else 0.
    r = tp / (tp + fn) if tp + fn else 0.
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_pooled": p, "recall_pooled": r,
            "f1_pooled": (2 * p * r / (p + r)) if p + r else 0.}


def run_comparison(raw_dir, out_dir, *, seed=42, trials=3, every=4,
                   f1_trigger=.50, consecutive_low=2, cooldown_windows=1,
                   min_delta=.01, max_attempts=None, profile=False):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Prohibido sobrescribir evidencia existente: {out}")
    baseline_raw, holdout, windows, manifest, features = _read_inputs(raw_dir)
    if len(windows) < 3:
        raise ValueError("Se requieren al menos 3 ventanas futuras")
    # Entrenamiento inicial COMPARTIDO: ninguna política puede elegir su baseline.
    baseline, init_seconds, init_usage = _train(baseline_raw, features, trials, seed,
                                                profile=profile)
    diagnostic = baseline.evaluate(holdout)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "candidate_artifacts"
    base_artifact = save_bundle(baseline, models_dir / "baseline.pkl")
    rows, summaries, candidates = [], [], []
    for policy in ("static", "periodic", "event"):
        settings = PolicySettings(
            policy, n_windows=len(windows), every=every, f1_trigger=f1_trigger,
            consecutive_low=consecutive_low, cooldown_windows=cooldown_windows,
            min_delta=min_delta, max_attempts=max_attempts)
        controller = MaintenanceController(settings, "baseline")
        versions = {"baseline": baseline}
        observed = [baseline_raw, holdout]
        policy_rows = []
        total_train_seconds = 0.
        total_cpu_seconds = 0.
        for window_index, window in enumerate(windows, start=1):
            incumbent = versions[controller.active_model]
            metrics = incumbent.evaluate(window)  # Predicción anterior a adaptación.
            pending = controller.pending
            if pending:
                challenger = versions[pending[1]]
                challenger_metrics = challenger.evaluate(window)
                decision = controller.complete_window(
                    window_index, metrics["f1"],
                    candidate_f1=challenger_metrics["f1"],
                    incumbent_f1=metrics["f1"])
            else:
                challenger_metrics = None
                decision = controller.complete_window(window_index, metrics["f1"])
            row = {
                "policy": policy, "window": window_index,
                "transactions": len(window), "positives": int(window["Class"].sum()),
                "scoring_model_id": decision.scoring_model,
                "scoring_family": incumbent.family,
                "next_model_id": decision.model_for_next_window,
                "trigger": decision.requested, "train_attempt": decision.execute,
                "reason": decision.reason, "candidate_reviewed": decision.candidate_reviewed,
                "candidate_approved": decision.candidate_approved,
                "challenger_f1_on_promotion_window": (
                    challenger_metrics["f1"] if challenger_metrics else None),
                "attempts_to_date": decision.attempts,
                "promotions_to_date": decision.promotions,
                "training_seconds": 0., "training_cpu_seconds": 0.,
                "training_peak_rss_mib": None, "training_error": None,
                **metrics,
            }
            # Solo datos observados hasta k. k+1 NO entra en este candidato.
            if decision.execute:
                attempt_history = pd.concat([*observed, window], ignore_index=True)
                candidate_id = f"{policy}_candidate_after_{window_index:02d}"
                try:
                    challenger, elapsed, usage = _train(attempt_history, features,
                                                        trials, seed, profile=profile)
                    versions[candidate_id] = challenger
                    controller.register_candidate(window_index, candidate_id)
                    saved = save_bundle(challenger, models_dir / (candidate_id + ".pkl"))
                    candidates.append({"policy": policy, "trained_after": window_index,
                                       "candidate_id": candidate_id, **saved,
                                       "search_results": challenger.search_results})
                    row["training_seconds"] = float(elapsed)
                    row["training_cpu_seconds"] = usage.get("process_tree_cpu_seconds", 0.)
                    row["training_peak_rss_mib"] = usage.get("peak_tree_rss_mib")
                    total_train_seconds += elapsed
                    total_cpu_seconds += usage.get("process_tree_cpu_seconds", 0.)
                except Exception as exc:
                    controller.record_training_failure(window_index)
                    row["training_error"] = f"{type(exc).__name__}: {exc}"
            observed.append(window)
            policy_rows.append(row)
            rows.append(row)
        totals = _pooled(policy_rows)
        summaries.append({
            "policy": policy, "seed": seed, "n_windows": len(windows),
            "f1_mean_window": float(np.mean([r["f1"] for r in policy_rows])),
            "recall_mean_window": float(np.mean([r["recall"] for r in policy_rows])),
            "attempts": controller.attempts, "promotions": controller.promotions,
            "rejected_or_failed": controller.attempts - controller.promotions,
            "training_seconds_additional": total_train_seconds,
            "training_cpu_seconds_additional": total_cpu_seconds,
            **totals,
        })
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    (out / "candidate_registry.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip() or None
    config = {
        "protocol": "study-a-v2-temporal-70-15-15-promotion-next-window",
        "source": str(raw_dir), "source_manifest": manifest,
        "seed": seed, "trials_per_family": trials, "families": 5,
        "every": every, "f1_trigger": f1_trigger,
        "consecutive_low": consecutive_low, "cooldown_windows": cooldown_windows,
        "min_delta": min_delta, "max_attempts": max_attempts,
        "monitoring_labels": "assumed_available_at_window_end",
        "hardware": platform.platform(), "git_sha": git_sha,
        "initial_training_seconds_excluded": init_seconds,
        "initial_training_resources_excluded": init_usage,
        "initial_family": baseline.family,
        "baseline_artifact": base_artifact,
        "diagnostic_holdout_metrics": diagnostic,
        "notes": [
            "Datos crudos, ajuste de preprocesador solo con train, tuning y calibración separados",
            "La ventana k se predice antes de entrenar; k+1 sirve para comparar/promover",
            "Un candidato aprobado se usa por primera vez en k+2",
            "No se ejecuta actualización si no quedan dos ventanas para promoción y prueba",
            "Resultados locales: no representan cloud billing, ni validación externa",
            "Las semillas Optuna no crean trayectorias temporales independientes"
        ],
    }
    (out / "configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pd.DataFrame(summaries)


def main():
    p = argparse.ArgumentParser(description="Estudio A v2; NO ejecuta v1, GitHub Actions ni Docker")
    p.add_argument("--input", required=True, help="Directorio raw-v2 con manifest.json")
    p.add_argument("--output", required=True, help="Directorio nuevo para resultados")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--f1-trigger", type=float, default=.50)
    p.add_argument("--consecutive-low", type=int, default=2)
    p.add_argument("--cooldown-windows", type=int, default=1)
    p.add_argument("--min-delta", type=float, default=.01)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--profile-resources", action="store_true")
    args = p.parse_args()
    result = run_comparison(args.input, args.output, seed=args.seed,
                            trials=args.trials, every=args.every,
                            f1_trigger=args.f1_trigger,
                            consecutive_low=args.consecutive_low,
                            cooldown_windows=args.cooldown_windows,
                            min_delta=args.min_delta,
                            max_attempts=args.max_attempts,
                            profile=args.profile_resources)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
