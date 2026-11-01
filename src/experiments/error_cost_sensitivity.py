"""Análisis de sensibilidad para costos de errores (unidades relativas, NO dinero).

No presupone costos bancarios reales: explora cómo cambia el resultado según
cuánto más caro consideramos un falso negativo frente a un falso positivo.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fn-ratios", type=float, nargs="+",
                        default=[1, 2, 5, 10, 20],
                        help="Costo relativo de FN cuando FP cuesta 1 unidad")
    args = parser.parse_args()
    root = args.results.resolve()
    data = pd.read_csv(root / "comparison.csv")
    required = {"policy", "false_negatives", "false_positives"}
    if required - set(data.columns):
        raise ValueError("comparison.csv no tiene columnas de errores")

    totals = data.groupby("policy", as_index=False)[
        ["false_negatives", "false_positives"]
    ].sum()
    output = []
    for weight in args.fn_ratios:
        if weight <= 0:
            parser.error("Todos los ratios de costo deben ser positivos")
        for row in totals.itertuples(index=False):
            output.append({
                "fn_cost_over_fp": weight,
                "policy": row.policy,
                "false_negatives": row.false_negatives,
                "false_positives": row.false_positives,
                "error_cost_relative_units":
                    weight * row.false_negatives + row.false_positives,
            })
    report = pd.DataFrame(output)
    report["rank_per_scenario"] = report.groupby(
        "fn_cost_over_fp")["error_cost_relative_units"].rank(method="min")
    result_path = root / "error_cost_sensitivity.csv"
    report.to_csv(result_path, index=False)
    print(report.to_string(index=False))
    print(f"\n[OK] {result_path}")
    print("[ADVERTENCIA] Unidades arbitrarias para escenarios, NO S/ ni costo real de fraude.")


if __name__ == "__main__":
    main()
