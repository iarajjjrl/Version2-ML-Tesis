"""Prepara ventanas cronológicas IEEE-CIS para comparar políticas de mantenimiento.

No modifica data/datos_entrenamiento.csv ni models/ de la demostración original.
Los lotes son estrictamente posteriores al entrenamiento y a la validación.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.prepare_data import FraudFeaturePreprocessor  # noqa: E402

FEATURES = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card4", "card6",
    "P_emaildomain", "R_emaildomain", "addr1", "addr2", "dist1",
    "C1", "C2", "C5", "C6", "C11", "C13", "D1", "D4", "DeviceType",
]


def read_source(transaction_path: Path, identity_path: Path) -> pd.DataFrame:
    tx_required = set(FEATURES) | {"TransactionID", "TransactionDT", "isFraud"}
    tx_cols = pd.read_csv(transaction_path, nrows=0).columns
    usable_tx = [c for c in tx_cols if c in tx_required]
    for required in ("TransactionID", "TransactionDT", "isFraud"):
        if required not in usable_tx:
            raise ValueError(f"Falta la columna {required} en {transaction_path}")
    tx = pd.read_csv(transaction_path, usecols=usable_tx, low_memory=False)
    id_cols = pd.read_csv(identity_path, nrows=0).columns
    use_id = [c for c in ("TransactionID", "DeviceType") if c in id_cols]
    if "DeviceType" in use_id:
        identity = pd.read_csv(identity_path, usecols=use_id, low_memory=False)
        tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    tx = tx.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    return tx.reset_index(drop=True)


def save_labeled(features: pd.DataFrame, labels: pd.Series, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = features.reset_index(drop=True).copy()
    frame["Class"] = labels.to_numpy(dtype=int)
    frame.to_csv(target, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Crear ventanas IEEE-CIS sin mezcla temporal")
    p.add_argument("--transactions", type=Path, default=ROOT / "train_transaction.csv")
    p.add_argument("--identity", type=Path, default=ROOT / "train_identity.csv")
    p.add_argument("--output", type=Path, default=ROOT / "data/experiments/chronological")
    p.add_argument("--max-rows", type=int, default=120000)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject-drift-from", type=int, default=0,
                   help="Desde esta ventana (1-based) distorsionar dos variables; 0 = sin drift sintético")
    args = p.parse_args()
    if args.windows < 3 or args.max_rows < 1000:
        p.error("Usa al menos 3 ventanas y 1000 registros")
    if args.inject_drift_from < 0 or args.inject_drift_from > args.windows:
        p.error("inject-drift-from debe estar entre 0 y windows")

    print("[DATA] Leyendo CSV originales (no se modificarán)...", flush=True)
    df = read_source(args.transactions, args.identity)
    total = len(df)
    if args.max_rows < total:
        # Submuestreo sistemático SIN desordenar la cronología.
        keep = np.linspace(0, total - 1, args.max_rows, dtype=np.int64)
        df = df.iloc[keep].reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.50)
    n_holdout = int(n * 0.10)
    if min(n_train, n_holdout) < 2:
        raise ValueError("Insuficientes datos para separar entrenamiento y validación")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = df.iloc[:n_train]
    holdout = df.iloc[n_train:n_train + n_holdout]
    future = df.iloc[n_train + n_holdout:].copy()
    if len(future) < args.windows:
        raise ValueError("No hay suficientes registros para tantas ventanas")

    feature_cols = [x for x in FEATURES if x in df.columns]
    if "TransactionAmt" not in feature_cols or "card1" not in feature_cols:
        raise ValueError("IEEE-CIS debe contener TransactionAmt y card1")
    def raw_features(data):
        return data.loc[:, feature_cols].copy()

    preprocess = FraudFeaturePreprocessor().fit(raw_features(train))
    joblib.dump(preprocess, output / "preprocessor.pkl")
    save_labeled(preprocess.transform(raw_features(train)), train["isFraud"],
                 output / "baseline_train.csv")
    save_labeled(preprocess.transform(raw_features(holdout)), holdout["isFraud"],
                 output / "promotion_holdout.csv")

    # Igual duración de TransactionDT en cada ventana; NO igual número de filas.
    tmin = int(future["TransactionDT"].min())
    tmax = int(future["TransactionDT"].max())
    if tmax <= tmin:
        raise ValueError("TransactionDT no cubre un rango temporal utilizable")
    boundaries = np.linspace(tmin, tmax + 1, args.windows + 1)
    window_id = np.searchsorted(boundaries[1:-1], future["TransactionDT"].to_numpy(),
                                side="right") + 1

    manifest = {
        "dataset": "IEEE-CIS Fraud Detection",
        "mode": "chronological_with_optional_synthetic_covariate_shift",
        "seed": args.seed,
        "original_records": total,
        "sampled_records": n,
        "baseline_train_records": len(train),
        "promotion_holdout_records": len(holdout),
        "n_windows": args.windows,
        "inject_drift_from": args.inject_drift_from,
        "features": list(preprocess.transform(raw_features(train.iloc[:1])).columns),
        "label_availability_assumption": "available_at_end_of_each_window",
        "windows": [],
    }
    for idx in range(1, args.windows + 1):
        part = future.iloc[np.flatnonzero(window_id == idx)]
        if part.empty:
            raise ValueError(f"La ventana {idx} está vacía: prueba reducir --windows")
        raw = raw_features(part)
        injected = args.inject_drift_from > 0 and idx >= args.inject_drift_from
        if injected:
            rng = np.random.default_rng(args.seed + idx)
            # Stress test determinístico de Data Drift; no equivale a Concept Drift.
            for col in ("TransactionAmt", "C13"):
                if col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce") * rng.uniform(
                        2.0, 3.0, size=len(raw)
                    )
        path = output / f"window_{idx:02d}.csv"
        save_labeled(preprocess.transform(raw), part["isFraud"], path)
        info = {
            "window": idx,
            "file": path.name,
            "records": int(len(part)),
            "positives": int(part["isFraud"].sum()),
            "transactiondt_min": int(part["TransactionDT"].min()),
            "transactiondt_max": int(part["TransactionDT"].max()),
            "synthetic_shift": bool(injected),
        }
        manifest["windows"].append(info)
        print(f"[DATA] {path.name}: {info['records']} registros; "
              f"positivos={info['positives']}; drift_sintetico={injected}", flush=True)

    with (output / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[OK] Datos generados en {output}; archivos originales intactos.")


if __name__ == "__main__":
    main()
