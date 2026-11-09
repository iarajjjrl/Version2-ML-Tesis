"""Separación cronológica y transformación sin información futura (estudio A, v2).

Este módulo NO ejecuta el benchmark, y no modifica el preparador v1. El
preprocesador se ajusta EXCLUSIVAMENTE sobre el bloque de entrenamiento.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalBlocks:
    train: pd.DataFrame
    tune: pd.DataFrame
    calibrate: pd.DataFrame


@dataclass
class TransformedBlocks:
    train_x: pd.DataFrame
    tune_x: pd.DataFrame
    calibrate_x: pd.DataFrame
    train_y: pd.Series
    tune_y: pd.Series
    calibrate_y: pd.Series
    preprocessor: object


def split_temporally(frame: pd.DataFrame, train_share: float = .70,
                     tune_share: float = .15, time_col: str = "TransactionDT",
                     id_col: str = "TransactionID") -> TemporalBlocks:
    """Parte un DataFrame *ya ordenado*, sin mezclar filas ni ajustar estimadores.

    Exige claves cronológicas para poder verificar el orden real. Los empates
    de TransactionDT se ordenan por TransactionID. No acepta dataframes sin
    claves de tiempo: hacerlo impediría demostrar ausencia de fuga temporal.
    """
    if not 0 < train_share < 1 or not 0 < tune_share < 1 or train_share + tune_share >= 1:
        raise ValueError("Proporciones inválidas para train/tune/calibrate")
    if frame.empty or frame[[time_col, id_col]].isna().any().any():
        raise ValueError("Se requieren tiempos e identificadores no nulos")
    n = len(frame)
    a = int(n * train_share)
    b = a + int(n * tune_share)
    if min(a, b - a, n - b) < 2:
        raise ValueError("Al menos dos registros en cada bloque")
    if frame.duplicated(subset=[id_col]).any():
        raise ValueError("Hay TransactionID duplicados")
    pair = pd.MultiIndex.from_frame(frame[[time_col, id_col]])
    if not pair.is_monotonic_increasing:
        raise ValueError("El historial debe estar ordenado por tiempo e identificador")
    return TemporalBlocks(frame.iloc[:a].copy(), frame.iloc[a:b].copy(),
                          frame.iloc[b:].copy())


def fit_transform_blocks(blocks: TemporalBlocks, preprocessor_factory,
                         feature_cols, label_col: str = "Class") -> TransformedBlocks:
    """Ajusta el preprocesador SOLO en train; aplica transform a tune y calibrate.

    No incluye identificadores/tiempos como predictores: solo feature_cols.
    preprocessor_factory debe crear una instancia nueva por intento de ajuste.
    """
    cols = list(feature_cols)
    if not cols or len(set(cols)) != len(cols):
        raise ValueError("feature_cols debe ser una selección no vacía y única")
    if label_col in cols or "TransactionDT" in cols or "TransactionID" in cols:
        raise ValueError("No incluir etiqueta, tiempo ni identificador en features")
    for part in (blocks.train, blocks.tune, blocks.calibrate):
        missing = set(cols + [label_col]) - set(part.columns)
        if missing:
            raise ValueError(f"Columnas requeridas ausentes: {sorted(missing)}")
    p = preprocessor_factory()
    train_x = blocks.train.loc[:, cols].copy()
    p.fit(train_x)  # Único fit; tune/calibrate no afectan imputación, codificación ni estadísticas
    result = TransformedBlocks(
        train_x=p.transform(train_x),
        tune_x=p.transform(blocks.tune.loc[:, cols].copy()),
        calibrate_x=p.transform(blocks.calibrate.loc[:, cols].copy()),
        train_y=blocks.train[label_col].astype(int).copy(),
        tune_y=blocks.tune[label_col].astype(int).copy(),
        calibrate_y=blocks.calibrate[label_col].astype(int).copy(),
        preprocessor=p,
    )
    expected = result.train_x.columns.tolist()
    if result.tune_x.columns.tolist() != expected or result.calibrate_x.columns.tolist() != expected:
        raise ValueError("El preprocesamiento cambió el esquema entre particiones")
    return result
