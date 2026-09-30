import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Codifica categorías y crea features usando estadísticas aprendidas en fit."""

    def __init__(self, amount_col="TransactionAmt", card_col="card1"):
        self.amount_col = amount_col
        self.card_col = card_col

    def fit(self, X, y=None):
        X = X.copy()
        self.categorical_maps_ = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                values = X[col].fillna("UNKNOWN").astype(str)
                self.categorical_maps_[col] = {
                    value: idx for idx, value in enumerate(pd.unique(values))
                }

        encoded = self._encode_and_fill(X)
        if self.card_col in encoded.columns:
            self.card1_counts_ = encoded[self.card_col].value_counts().to_dict()
            if self.amount_col in encoded.columns:
                self.card1_amount_means_ = encoded.groupby(self.card_col)[self.amount_col].mean().to_dict()
            else:
                self.card1_amount_means_ = {}
        else:
            self.card1_counts_ = {}
            self.card1_amount_means_ = {}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _encode_and_fill(self, X):
        result = X.copy()
        for col in result.columns:
            if col in getattr(self, "categorical_maps_", {}):
                result[col] = (result[col].fillna("UNKNOWN").astype(str)
                               .map(self.categorical_maps_[col]).fillna(-1))
            else:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(-999)
        return result

    def transform(self, X):
        if not hasattr(self, "categorical_maps_"):
            raise RuntimeError("El preprocesador debe ajustarse con fit antes de transform.")
        result = self._encode_and_fill(X)
        if self.amount_col in result.columns:
            result[f"{self.amount_col}_log"] = np.log1p(result[self.amount_col].clip(lower=0))
        if self.card_col in result.columns:
            result["card1_count"] = result[self.card_col].map(self.card1_counts_).fillna(1)
            means = result[self.card_col].map(self.card1_amount_means_)
            if self.amount_col in result.columns:
                result["amt_to_mean_card1"] = (result[self.amount_col] / (means + 1e-5)).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0).round(4)
        c_cols = [c for c in ["C1", "C2", "C5", "C6", "C11", "C13"] if c in result.columns]
        if c_cols:
            result["C_sum_velocity"] = result[c_cols].sum(axis=1)
        return result


def transformar_lote_produccion(df, ruta_preprocessor="models/preprocessor.pkl"):
    """Transforma un lote nuevo con el estado aprendido durante el entrenamiento."""
    preprocessor = joblib.load(ruta_preprocessor)
    return preprocessor.transform(df)

def aplicar_feature_engineering_industrial(df, preprocessor=None):
    """
    Aplica técnicas de Feature Engineering de nivel industrial para detección de fraude:
    1. Frecuencia de uso de tarjetas (Velocity Encoding).
    2. Ratios transaccionales respecto a la media de la tarjeta.
    3. Transformaciones logarítmicas de montos.
    4. Agregados de contadores de actividad (C-scores).
    """
    if preprocessor is None:
        raise ValueError("Pasa un FraudFeaturePreprocessor ajustado para evitar fuga de datos.")
    return preprocessor.transform(df)

def preparar_ieee_fraud(ruta_trans="train_transaction.csv", ruta_id="train_identity.csv"):
    print("[INFO] Cargando datasets originales de IEEE-CIS...")
    df_trans = pd.read_csv(ruta_trans)
    df_id = pd.read_csv(ruta_id)
    
    print("[INFO] Unificando archivos mediante TransactionID...")
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    if 'TransactionDT' in df.columns:
        print("[INFO] Ordenando cronológicamente por TransactionDT...")
        df = df.sort_values(by='TransactionDT').reset_index(drop=True)
    
    cols_negocio = [
        'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card4', 'card6',
        'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2', 'dist1',
        'C1', 'C2', 'C5', 'C6', 'C11', 'C13', 'D1', 'D4', 'DeviceType'
    ]
    
    cols_existentes = [c for c in cols_negocio if c in df.columns]
    df_filtrado = df[cols_existentes + ['isFraud']].copy()
    
    df_filtrado = df_filtrado.rename(columns={'isFraud': 'Class'})
    X = df_filtrado.drop(columns=['Class'])
    y = df_filtrado['Class']
    
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=0.30, random_state=42, stratify=y
    )
    
    X_train_raw, X_prod_raw, y_train, y_prod = train_test_split(
        X_sample, y_sample, test_size=0.40, random_state=42, stratify=y_sample
    )

    # Fit solo con entrenamiento; toda estadística derivada queda congelada para producción.
    preprocessor = FraudFeaturePreprocessor().fit(X_train_raw)
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    X_train = preprocessor.transform(X_train_raw)
    X_prod = preprocessor.transform(X_prod_raw)
    
    # Crear Lotes Reales de Producción
    X_lote1_raw, X_lote2_raw, y_lote1, y_lote2 = train_test_split(
        X_prod, y_prod, test_size=0.50, random_state=42, stratify=y_prod
    )
    X_lote1, X_lote2 = X_lote1_raw.copy(), X_lote2_raw.copy()
    
    # Inyectar Data Drift en Lote 2 (Simulación de Ataque Masivo)
    X_lote2_drift = X_lote2.copy()
    cols_drift = ['TransactionAmt', 'card1', 'C1', 'C13', 'amt_to_mean_card1']
    for col in cols_drift:
        if col in X_lote2_drift.columns:
            X_lote2_drift[col] = (X_lote2_drift[col] * np.random.uniform(2.5, 4.0, size=len(X_lote2_drift))).astype(X_lote2_drift[col].dtype)
            
    os.makedirs("data", exist_ok=True)
    X_train.assign(Class=y_train).to_csv("data/datos_entrenamiento.csv", index=False)
    X_lote1.assign(Class=y_lote1).to_csv("data/prod_lote1_normal.csv", index=False)
    X_lote2_drift.assign(Class=y_lote2).to_csv("data/prod_lote2_drift.csv", index=False)
    X_lote1.assign(Class=y_lote1).to_csv("data/prod_lote3_post_retrain.csv", index=False)
    
    print("[INFO]  Lotes enriquecidos generados exitosamente en 'data/'.")

if __name__ == "__main__":
    preparar_ieee_fraud()
