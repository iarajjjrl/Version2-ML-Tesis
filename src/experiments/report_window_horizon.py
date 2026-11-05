"""Audita el periodo de las ventanas IEEE-CIS sin modificar ni reentrenar.

PowerShell:
  .\venv\Scripts\python.exe src\experiments\report_window_horizon.py --manifest data\experiments\study-a\input\natural\manifest.json
"""
import argparse
import json
from pathlib import Path

SECONDS_PER_DAY = 86400.0


def main():
    parser = argparse.ArgumentParser(description="Auditoría temporal de IEEE-CIS")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    obj = json.loads(args.manifest.read_text(encoding="utf-8"))
    windows = sorted(obj.get("windows", []), key=lambda w: w["window"])
    if not windows:
        parser.error("manifest.json no contiene ventanas")

    first = windows[0]["transactiondt_min"]
    last = windows[-1]["transactiondt_max"]
    duration = (last - first) / SECONDS_PER_DAY
    print("HORIZONTE IEEE-CIS: TransactionDT en segundos relativos, NO fechas calendario")
    print(f"Registros seleccionados: {obj.get('sampled_records', 'ND')}")
    print(f"Entrenamiento inicial: {obj.get('baseline_train_records', 'ND')}")
    print(f"Holdout histórico: {obj.get('promotion_holdout_records', 'ND')}")
    print(f"Horizonte de evaluación observado: {duration:.3f} días")
    print(f"Número de ventanas: {len(windows)}")
    print(f"Duración nominal aproximada por ventana: {duration / len(windows):.3f} días")
    print("NOTA: las ventanas son intervalos de igual duración; el rango observado dentro de cada una puede ser menor.")
    print("Ventana | Registros | Fraudes | Prevalencia (%) | Rango observado (días)")
    previous_max = None
    for w in windows:
        low, high = w["transactiondt_min"], w["transactiondt_max"]
        n, positive = w["records"], w["positives"]
        if n <= 0 or high < low or (previous_max is not None and low < previous_max):
            raise ValueError("Ventanas inconsistentes o vacías")
        previous_max = high
        print(
            f"{w['window']:>7} | {n:>9} | {positive:>7} | "
            f"{positive / n * 100:>13.2f} | {(high - low)/SECONDS_PER_DAY:>22.3f}"
        )
    print("No atribuir fechas calendario ni afirmar un año sin evidencia.")


if __name__ == "__main__":
    main()
