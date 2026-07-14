import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def aplicar_feature_engineering_industrial(df):
    """
    Aplica técnicas de Feature Engineering de nivel industrial para detección de fraude:
    1. Frecuencia de uso de tarjetas (Velocity Encoding).
    2. Ratios transaccionales respecto a la media de la tarjeta.
    3. Transformaciones logarítmicas de montos.
    4. Agregados de contadores de actividad (C-scores).
    """
    print("[FEATURE ENGINEERING] Generando variables derivadas de alta discriminación...")
    df_feat = df.copy()
    
    # 1. Transformación Logarítmica del Monto (Suaviza outliers extremos)
    if 'TransactionAmt' in df_feat.columns:
        df_feat['TransactionAmt_log'] = np.log1p(df_feat['TransactionAmt'])
    
    # 2. Velocity Encoding: Frecuencia de aparición de la tarjeta (card1)
    if 'card1' in df_feat.columns:
        card1_counts = df_feat['card1'].value_counts()
        df_feat['card1_count'] = df_feat['card1'].map(card1_counts).fillna(1)
        
        # 3. Ratio del Monto vs Promedio de esa Tarjeta
        card1_amt_mean = df_feat.groupby('card1')['TransactionAmt'].transform('mean')
        df_feat['amt_to_mean_card1'] = (df_feat['TransactionAmt'] / (card1_amt_mean + 1e-5)).round(4)
    
    # 4. Agregado de Velocidad Transaccional (Suma de contadores C1-C13)
    c_cols = [c for c in ['C1', 'C2', 'C5', 'C6', 'C11', 'C13'] if c in df_feat.columns]
    if c_cols:
        df_feat['C_sum_velocity'] = df_feat[c_cols].sum(axis=1)
        
    return df_feat

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
    
    print("[INFO] Codificando variables categóricas...")
    for col in df_filtrado.columns:
        if col == 'isFraud':
            continue
        if not pd.api.types.is_numeric_dtype(df_filtrado[col]):
            df_filtrado[col] = df_filtrado[col].fillna('UNKNOWN').astype(str)
            le = LabelEncoder()
            df_filtrado[col] = le.fit_transform(df_filtrado[col])
        else:
            df_filtrado[col] = df_filtrado[col].fillna(-999)
            
    df_filtrado = df_filtrado.rename(columns={'isFraud': 'Class'})
    
    # 🚀 APLICACIÓN DE FEATURE ENGINEERING
    df_enriquecido = aplicar_feature_engineering_industrial(df_filtrado)
    
    # Muestreo estratificado del 30%
    X = df_enriquecido.drop(columns=['Class'])
    y = df_enriquecido['Class']
    
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=0.30, random_state=42, stratify=y
    )
    
    X_train, X_prod, y_train, y_prod = train_test_split(
        X_sample, y_sample, test_size=0.40, random_state=42, stratify=y_sample
    )
    
    # Crear Lotes Reales de Producción
    X_lote1, X_lote2, y_lote1, y_lote2 = train_test_split(
        X_prod, y_prod, test_size=0.50, random_state=42, stratify=y_prod
    )
    
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