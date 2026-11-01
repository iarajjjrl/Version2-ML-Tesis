"""Resumen reproducible y gráficos del piloto de políticas.

La media simple de F1 por ventana NO equivale al F1 agregado de transacciones.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    folder = args.results.resolve()
    df = pd.read_csv(folder / "comparison.csv")
    expected = {"policy", "window", "f1", "recall", "positives",
                "false_negatives", "false_positives", "trigger",
                "candidate_promoted", "training_wall_seconds",
                "training_process_cpu_seconds", "monitor_wall_seconds"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en comparison.csv: {missing}")

    df = df.sort_values(["policy", "window"])
    summary = []
    for policy, group in df.groupby("policy", sort=True):
        tp = int((group["positives"] - group["false_negatives"]).sum())
        fp = int(group["false_positives"].sum())
        fn = int(group["false_negatives"].sum())
        pooled_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        summary.append({
            "policy": policy,
            "window_mean_f1": float(group["f1"].mean()),
            "window_median_f1": float(group["f1"].median()),
            "pooled_f1": pooled_f1,
            "window_mean_recall": float(group["recall"].mean()),
            "false_negatives_total": fn,
            "false_positives_total": fp,
            "retraining_attempts": int(group["trigger"].sum()),
            "promotions": int(group["candidate_promoted"].sum()),
            "training_wall_seconds_total": float(group["training_wall_seconds"].sum()),
            "training_process_cpu_seconds_total": float(
                group["training_process_cpu_seconds"].sum()),
            "monitor_wall_seconds_total": float(group["monitor_wall_seconds"].sum()),
        })
    # Datos opcionales: el usuario activa --profile-resources y tiene psutil.
    if {"peak_tree_rss_mib", "process_tree_cpu_seconds"}.issubset(df.columns):
        for record in summary:
            group = df[df["policy"] == record["policy"]]
            rss = pd.to_numeric(group["peak_tree_rss_mib"], errors="coerce")
            cpu = pd.to_numeric(group["process_tree_cpu_seconds"], errors="coerce")
            record["peak_tree_rss_mib_max"] = (
                float(rss.max()) if rss.notna().any() else None
            )
            record["process_tree_cpu_seconds_total"] = (
                float(cpu.sum()) if cpu.notna().any() else None
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(folder / "summary.csv", index=False)
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    for policy, grp in df.groupby("policy"):
        ax.plot(grp["window"], grp["f1"], marker="o", label=policy)
    ax.set(xlabel="Ventana temporal", ylabel="F1", title="Desempeño predictivo prequential")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "f1_by_window.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for policy, grp in df.groupby("policy"):
        ax.plot(grp["window"], grp["training_wall_seconds"].cumsum(),
                marker="o", label=policy)
    ax.set(xlabel="Ventana temporal", ylabel="Segundos acumulados",
           title="Tiempo local de entrenamiento acumulado (sin baseline común)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "training_time_by_window.png", dpi=180)
    plt.close(fig)
    print("[OK] summary.csv y gráficos PNG guardados en", folder)


if __name__ == "__main__":
    main()
