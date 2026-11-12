"""Orquestador descriptivo v2 (época × escenario × semilla × periodicidad).

NO ejecuta pruebas de significación; no confunde semillas con datos independientes.
NO abre ni modifica GitHub Actions, Docker, models/ o resultados v1.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.experiments.v2.run_comparison_v2 import run_comparison


def run_study(epochs_dir, output, *, seeds=(42, 43, 44), every_values=(4,),
              trials=3, f1_trigger=.50, consecutive_low=2,
              cooldown_windows=1, max_attempts=None, profile=False):
    epochs_root = Path(epochs_dir)
    target = Path(output)
    if target.exists() and (target / "study_aggregate.csv").exists():
        raise FileExistsError("El resumen definitivo ya existe; no sobrescribir resultados")
    rows = []
    for epoch in sorted(epochs_root.glob("epoch_*")):
        for scenario in ("natural", "synthetic"):
            raw = epoch / scenario
            if not (raw / "manifest.json").exists():
                raise FileNotFoundError(raw / "manifest.json")
            for seed in seeds:
                for every in every_values:
                    output_run = (target / epoch.name / scenario /
                                  f"seed_{seed}" / f"periodic_every_{every}")
                    summary = output_run / "summary.csv"
                    if summary.exists() and (output_run / "configuration.json").exists():
                        result = pd.read_csv(summary)
                    elif output_run.exists() and any(output_run.iterdir()):
                        raise RuntimeError(f"Corrida parcial que requiere auditoría: {output_run}")
                    else:
                        result = run_comparison(
                            raw, output_run, seed=seed, trials=trials,
                            every=every, f1_trigger=f1_trigger,
                            consecutive_low=consecutive_low,
                            cooldown_windows=cooldown_windows,
                            max_attempts=max_attempts, profile=profile)
                    for record in result.to_dict(orient="records"):
                        rows.append({"epoch": epoch.name, "scenario": scenario,
                                     "seed": seed, "periodic_every": every,
                                     "trials": trials, **record})
    target.mkdir(parents=True, exist_ok=True)
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(target / "study_runs.csv", index=False)
    metric_columns = ["f1_pooled", "f1_mean_window",
                      "training_seconds_additional", "attempts",
                      "promotions", "fp", "fn"]
    grouped = all_runs.groupby(["scenario", "policy", "periodic_every"], dropna=False)
    aggregated = grouped[metric_columns].agg(["mean", "std", "min", "max"])
    aggregated.columns = [f"{a}_{b}" for a, b in aggregated.columns]
    aggregated = aggregated.reset_index()
    aggregated["n_epoch_seed_combinations"] = grouped.size().to_numpy()
    aggregated.to_csv(target / "study_aggregate.csv", index=False)
    return all_runs, aggregated


def main():
    p = argparse.ArgumentParser(description="Estudio A v2 multiepoca (local y descriptivo)")
    p.add_argument("--epochs-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--every", nargs="+", type=int, default=[4])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--f1-trigger", type=float, default=.50)
    p.add_argument("--consecutive-low", type=int, default=2)
    p.add_argument("--cooldown-windows", type=int, default=1)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--profile-resources", action="store_true")
    args = p.parse_args()
    results, aggregate = run_study(
        args.epochs_dir, args.output, seeds=args.seeds,
        every_values=args.every, trials=args.trials,
        f1_trigger=args.f1_trigger, consecutive_low=args.consecutive_low,
        cooldown_windows=args.cooldown_windows,
        max_attempts=args.max_attempts, profile=args.profile_resources)
    print(f"[OK] {len(results)} resultados de política; "
          "descriptivos, no evidencia de significancia poblacional")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
