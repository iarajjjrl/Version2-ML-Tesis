"""Preparación de entradas crudas para el Estudio A v2.

A diferencia de v1, no ajusta ni aplica FraudFeaturePreprocessor.
Conserva TransactionDT/TransactionID para poder auditar la cronología al
reentrenar y el origen de cada variable. Nunca sobrescribe salidas v1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.prepare_windows import FEATURES, read_source


def build_raw_windows(df: pd.DataFrame, *, max_rows: int = 120000,
                      n_windows: int = 8, seed: int = 42,
                      inject_drift_from: int = 0):
    """Devuelve baseline, holdout, ventanas y manifiesto sin aplicar fit.

    df debe ser el resultado cronológicamente ordenado de read_source.
    No ejecuta ML, no usa etiquetas futuras para ajustar transformaciones.
    """
    required = {"TransactionDT", "TransactionID", "isFraud", "TransactionAmt", "card1"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas esenciales: {sorted(required - set(df.columns))}")
    if n_windows < 3 or max_rows < 1000:
        raise ValueError("Se requieren al menos 3 ventanas y 1000 observaciones")
    if not 0 <= inject_drift_from <= n_windows:
        raise ValueError("inject_drift_from fuera de rango")
    if df.duplicated(subset=["TransactionID"]).any():
        raise ValueError("TransactionID debe ser único")
    if not pd.MultiIndex.from_frame(df[["TransactionDT", "TransactionID"]]).is_monotonic_increasing:
        raise ValueError("La tabla origen debe estar ordenada por TransactionDT, TransactionID")
    total = len(df)
    if max_rows < total:
        keep = np.linspace(0, total - 1, max_rows, dtype=np.int64)
        df = df.iloc[keep].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    n = len(df)
    n_train, n_holdout = int(n * .5), int(n * .1)
    baseline = df.iloc[:n_train].copy()
    holdout = df.iloc[n_train:n_train + n_holdout].copy()
    future = df.iloc[n_train + n_holdout:].copy()
    if len(future) < n_windows:
        raise ValueError("Menos filas futuras que ventanas")
    tmin, tmax = float(future.TransactionDT.min()), float(future.TransactionDT.max())
    if not np.isfinite([tmin, tmax]).all() or tmax <= tmin:
        raise ValueError("TransactionDT debe contener un horizonte temporal utilizable")
    edges = np.linspace(tmin, tmax + 1, n_windows + 1)
    assigned = np.searchsorted(edges[1:-1], future.TransactionDT.to_numpy(), side="right") + 1
    features = [c for c in FEATURES if c in df.columns]
    cols = ["TransactionDT", "TransactionID", *features, "isFraud"]
    def export(raw):
        result = raw.loc[:, cols].copy()
        return result.rename(columns={"isFraud": "Class"})
    windows = []
    info = []
    for k in range(1, n_windows + 1):
        part = future.iloc[np.flatnonzero(assigned == k)].copy()
        if part.empty:
            raise ValueError(f"Ventana {k} vacía")
        shifted = bool(inject_drift_from and k >= inject_drift_from)
        if shifted:
            rng = np.random.default_rng(seed + k)
            for col in ("TransactionAmt", "C13"):
                if col in part.columns:
                    part[col] = pd.to_numeric(part[col], errors="coerce") * rng.uniform(2., 3., size=len(part))
        windows.append(export(part))
        info.append({"window": k, "file": f"window_{k:02d}_raw.csv",
                     "records": len(part), "positives": int(part.isFraud.sum()),
                     "transactiondt_min": float(part.TransactionDT.min()),
                     "transactiondt_max": float(part.TransactionDT.max()),
                     "synthetic_shift": shifted})
    manifest = {"dataset": "IEEE-CIS Fraud Detection", "preprocessed": False,
                "original_records": total, "sampled_records": n,
                "baseline_train_records": len(baseline), "diagnostic_holdout_records": len(holdout),
                "n_windows": n_windows, "seed": seed,
                "inject_drift_from": inject_drift_from, "features": features,
                "order": ["TransactionDT", "TransactionID"],
                "label_availability_assumption": "available_at_end_of_each_window",
                "windows": info}
    return export(baseline), export(holdout), windows, manifest


def main():
    p = argparse.ArgumentParser(description="Preparación v2 de ventanas crudas (sin fit global)")
    p.add_argument("--transactions", type=Path, default=Path("train_transaction.csv"))
    p.add_argument("--identity", type=Path, default=Path("train_identity.csv"))
    p.add_argument("--output", type=Path, default=Path("data/experiments/raw-v2"))
    p.add_argument("--max-rows", type=int, default=120000)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject-drift-from", type=int, default=0)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"No sobrescribir una salida existente: {output}")
    source = read_source(args.transactions, args.identity)
    base, holdout, windows, manifest = build_raw_windows(
        source, max_rows=args.max_rows, n_windows=args.windows,
        seed=args.seed, inject_drift_from=args.inject_drift_from)
    output.mkdir(parents=True, exist_ok=True)
    base.to_csv(output / "baseline_train_raw.csv", index=False)
    holdout.to_csv(output / "diagnostic_holdout_raw.csv", index=False)
    for win, info in zip(windows, manifest["windows"]):
        win.to_csv(output / info["file"], index=False)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Ventanas crudas v2 creadas en {output}; no se ajustó ningún preprocesador")


if __name__ == "__main__":
    main()
