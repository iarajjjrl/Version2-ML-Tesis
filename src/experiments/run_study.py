"""Orquesta un estudio predefinido sobre escenarios y semillas de Optuna.

Ejecutar desde la raíz del repositorio. Los datos originales se leen, no se alteran.
Cada corrida usa una carpeta independiente; se reanudan corridas completas.
"""
import argparse
import json
import platform
from importlib import metadata
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "src" / "experiments"


def package_versions():
    names = ("pandas", "numpy", "scikit-learn", "optuna",
             "xgboost", "lightgbm", "catboost", "psutil")
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_logged(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    with logfile.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            check=False, text=True,
        )
    if proc.returncode:
        tail = logfile.read_text(encoding="utf-8").splitlines()[-30:]
        raise RuntimeError(
            f"Error en {logfile}, código {proc.returncode}:\n" +
            "\n".join(tail)
        )
    print(f"[OK] Registro completo: {logfile}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Estudio de robustez de políticas MLOps")
    p.add_argument("--output", type=Path, default=ROOT / "data/experiments/study-a")
    p.add_argument("--max-rows", type=int, default=120000)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--scenarios", choices=("natural", "synthetic"), nargs="+",
                   default=["natural", "synthetic"])
    p.add_argument("--f1-trigger", type=float, default=0.50)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--consecutive-low", type=int, default=2)
    p.add_argument("--cooldown-windows", type=int, default=1)
    p.add_argument("--min-improvement", type=float, default=0.01)
    p.add_argument("--synthetic-start", type=int, default=3)
    p.add_argument("--profile-resources", action="store_true")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.scenarios)) != len(args.scenarios):
        p.error("No repitas semillas ni escenarios")
    if args.trials < 1 or args.max_rows < 1000 or args.windows < 3:
        p.error("trials >= 1, max-rows >= 1000, windows >= 3")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "max_rows": args.max_rows,
        "windows": args.windows,
        "trials": args.trials,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "f1_trigger": args.f1_trigger,
        "every": args.every,
        "consecutive_low": args.consecutive_low,
        "cooldown_windows": args.cooldown_windows,
        "min_improvement": args.min_improvement,
        "synthetic_start": args.synthetic_start,
        "profile_resources": args.profile_resources,
    }
    cfg = output / "study_config.json"
    if cfg.exists():
        old = json.loads(cfg.read_text(encoding="utf-8"))
        if old["configuration"] != config:
            raise ValueError(
                "Ya existe una configuración diferente en esta carpeta. "
                "Usa otra ruta --output para no mezclar estudios."
            )
    else:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_hash = None
        record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": git_hash,
            "package_versions": package_versions(),
            "notes": [
                "Seeds de AutoML distintas, mismo dataset/ventanas; NO son réplicas independientes de datos.",
                "Escenario sintético altera covariables, NO garantiza concept drift.",
                "Tiempo y recursos medidos localmente, NO es costo de GitHub Actions.",
                "No hay pruebas de significancia estadística en este script.",
            ],
        }
        cfg.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")

    rows = []
    for scenario in args.scenarios:
        input_dir = output / "input" / scenario
        manifest = input_dir / "manifest.json"
        if not manifest.exists():
            command = [
                sys.executable, str(EXPERIMENTS / "prepare_windows.py"),
                "--max-rows", str(args.max_rows), "--windows", str(args.windows),
                "--seed", "42", "--output", str(input_dir),
            ]
            if scenario == "synthetic":
                command += ["--inject-drift-from", str(args.synthetic_start)]
            run_logged(command, output / "logs" / f"prepare_{scenario}.log")
        observed = json.loads(manifest.read_text(encoding="utf-8"))
        if observed["sampled_records"] != args.max_rows and observed["original_records"] >= args.max_rows:
            raise ValueError(f"Muestra inesperada para {scenario}")
        if observed["n_windows"] != args.windows:
            raise ValueError(f"Número de ventanas inesperado para {scenario}")
        if observed["inject_drift_from"] != (
                args.synthetic_start if scenario == "synthetic" else 0):
            raise ValueError(f"Escenario de drift inesperado: {scenario}")

        for seed in args.seeds:
            run_dir = output / "runs" / scenario / f"seed_{seed}"
            summary = run_dir / "summary.csv"
            if not summary.exists():
                if not (run_dir / "comparison.csv").exists():
                    if run_dir.exists() and any(run_dir.iterdir()):
                        # Se conserva la ejecución interrumpida, sin borrarla.
                        attempt = 1
                        while (run_dir.parent /
                               f"seed_{seed}_incomplete_{attempt}").exists():
                            attempt += 1
                        backup = run_dir.parent / f"seed_{seed}_incomplete_{attempt}"
                        run_dir.rename(backup)
                        print(f"[RECOVERY] Corrida incompleta preservada: {backup}")
                    cmd = [
                        sys.executable, str(EXPERIMENTS / "run_comparison.py"),
                        "--input", str(input_dir), "--output", str(run_dir),
                        "--trials", str(args.trials), "--seed", str(seed),
                        "--every", str(args.every),
                        "--f1-trigger", str(args.f1_trigger),
                        "--consecutive-low", str(args.consecutive_low),
                        "--cooldown-windows", str(args.cooldown_windows),
                        "--min-improvement", str(args.min_improvement),
                    ]
                    if args.profile_resources:
                        cmd.append("--profile-resources")
                    run_logged(cmd, output / "logs" / f"fit_{scenario}_{seed}.log")
                run_logged(
                    [sys.executable, str(EXPERIMENTS / "summarize.py"),
                     "--results", str(run_dir)],
                    output / "logs" / f"summary_{scenario}_{seed}.log",
                )
            else:
                print(f"[RESUME] Ya completado: {scenario}, semilla {seed}", flush=True)
            data = pd.read_csv(summary)
            data["scenario"] = scenario
            data["seed"] = seed
            rows.append(data)

    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(output / "study_runs.csv", index=False)
    metrics = [
        "window_mean_f1", "pooled_f1", "window_mean_recall",
        "false_negatives_total", "false_positives_total", "retraining_attempts",
        "promotions", "training_wall_seconds_total",
    ]
    for extra in ("process_tree_cpu_seconds_total", "peak_tree_rss_mib_max"):
        if extra in all_results.columns:
            metrics.append(extra)
    aggregate = all_results.groupby(["scenario", "policy"], as_index=False)[
        metrics
    ].agg(["mean", "std"])
    aggregate.columns = [
        "_".join(str(k) for k in item if k).rstrip("_")
        for item in aggregate.columns.to_flat_index()
    ]
    aggregate.to_csv(output / "study_aggregate.csv", index=False)
    print("\n[OK] Estudio terminado:", output)
    print("[REVISAR] study_runs.csv y study_aggregate.csv")
    print("[AVISO] Repetir seeds no produce datasets independientes.")


if __name__ == "__main__":
    main()
