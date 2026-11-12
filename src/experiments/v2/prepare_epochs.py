"""Prepara BLOQUES TEMPORALES DISJUNTOS para robustez del Estudio A v2.

Cada bloque usa un modelo inicial propio y N ventanas cronológicas; evita
considerar semillas Optuna como replicas de datos independientes. Las distintas
épocas pertenecen al mismo dataset, y no equivalen a bancos independientes.
Nunca sobrescribe resultados existentes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.prepare_windows import read_source
from src.experiments.v2.raw_windows import build_raw_windows


def partition_epochs(source, count=3):
    if count < 2:
        raise ValueError("Se requieren al menos dos épocas temporales")
    if source.duplicated(subset=["TransactionID"]).any():
        raise ValueError("TransactionID duplicado en origen")
    pair = pd.MultiIndex.from_frame(source[["TransactionDT", "TransactionID"]])
    if not pair.is_monotonic_increasing:
        raise ValueError("La fuente debe estar ordenada por TransactionDT y TransactionID")
    pieces = [p.copy() for p in np.array_split(np.arange(len(source)), count)]
    epochs = [source.iloc[positions].reset_index(drop=True) for positions in pieces]
    if any(len(x) < 1000 for x in epochs):
        raise ValueError("Cada época requiere al menos 1000 transacciones")
    for left, right in zip(epochs, epochs[1:]):
        a = tuple(left[["TransactionDT", "TransactionID"]].iloc[-1])
        b = tuple(right[["TransactionDT", "TransactionID"]].iloc[0])
        if a >= b:
            raise ValueError("Épocas no son cronológicamente disjuntas")
    return epochs


def prepare_epochs(source, output, *, epochs=3, max_rows=120000,
                   windows=8, seed=42):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"No sobrescribir el directorio: {out}")
    blocks = partition_epochs(source, epochs)
    out.mkdir(parents=True, exist_ok=True)
    catalogue = {"method": "contiguous_disjoint_chronological_epochs",
                 "n_epochs": epochs, "rows_in_source": len(source),
                 "seed_injection": seed, "epochs": []}
    for number, part in enumerate(blocks, start=1):
        desc = {"epoch": number, "source_rows": len(part),
                "transactiondt_start": float(part.TransactionDT.min()),
                "transactiondt_end": float(part.TransactionDT.max()),
                "scenarios": {}}
        for scenario, drift_start in (("natural", 0), ("synthetic", 3)):
            baseline, holdout, batches, manifest = build_raw_windows(
                part, max_rows=max_rows, n_windows=windows,
                seed=seed, inject_drift_from=drift_start)
            target = out / f"epoch_{number:02d}" / scenario
            target.mkdir(parents=True, exist_ok=False)
            baseline.to_csv(target / "baseline_train_raw.csv", index=False)
            holdout.to_csv(target / "diagnostic_holdout_raw.csv", index=False)
            for batch, meta in zip(batches, manifest["windows"]):
                batch.to_csv(target / meta["file"], index=False)
            manifest["epoch"] = number
            manifest["scenario"] = scenario
            (target / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            desc["scenarios"][scenario] = str(target)
        catalogue["epochs"].append(desc)
    (out / "epochs_manifest.json").write_text(
        json.dumps(catalogue, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return catalogue


def main():
    p = argparse.ArgumentParser(description="Épocas disjuntas IEEE-CIS para Estudio A v2")
    p.add_argument("--transactions", type=Path, default=Path("train_transaction.csv"))
    p.add_argument("--identity", type=Path, default=Path("train_identity.csv"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-rows", type=int, default=120000)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    source = read_source(args.transactions, args.identity)
    catalogue = prepare_epochs(
        source, args.output, epochs=args.epochs,
        max_rows=args.max_rows, windows=args.windows, seed=args.seed)
    print(f"[OK] {catalogue['n_epochs']} épocas disjuntas; "
          "dos escenarios por época. Ningún modelo ha sido entrenado.")


if __name__ == "__main__":
    main()
