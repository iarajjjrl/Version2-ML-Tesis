import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preparar_ieee_fraud(ruta_trans="train_transaction.csv", ruta_id="train_identity.csv"):
    print("[INFO] Cargando datasets originales de IEEE-CIS...")
    df_trans = pd.read_csv(ruta_trans)
    df_id = pd.read_csv(ruta_id)
    
    print("[INFO] Unificando archivos mediante TransactionID...")
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    # 1. ORDENAMIENTO TEMPORAL ESTRICTO (Metodología 4.2: Evita Data Leakage)
    if 'TransactionDT' in df.columns:
        print("[INFO] Ordenando cronológicamente el dataset por TransactionDT...")
        df = df.sort_values(by='TransactionDT').reset_index(drop=True)
    
    cols_negocio = [
        'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card4', 'card6',
        'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2', 'dist1',
        'C1', 'C2', 'C5', 'C6', 'C11', 'C13', 'D1', 'D4', 'DeviceType'
    ]
    
    cols_existentes = [c for c in cols_negocio if c in df.columns]
    df_filtrado = df[cols_existentes + ['isFraud']].copy()
    
    print("[INFO] Codificando variables categóricas a valores numéricos...")
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
    
    # Muestreo del 30% para MLOps
    print("[INFO] Generando muestreo estratificado (30% para MLOps)...")
    X = df_filtrado.drop(columns=['Class'])
    y = df_filtrado['Class']
    
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=0.30, random_state=42, stratify=y
    )
    
    X_train, X_prod, y_train, y_prod = train_test_split(
        X_sample, y_sample, test_size=0.30, random_state=42, stratify=y_sample
    )
    
    X_p1, X_p2, y_p1, y_p2 = train_test_split(
        X_prod, y_prod, test_size=0.50, random_state=42, stratify=y_prod
    )
    
    X_p2_drift = X_p2.copy()
    cols_drift = ['TransactionAmt', 'card1', 'C1', 'C13']
    for col in cols_drift:
        if col in X_p2_drift.columns:
            X_p2_drift[col] = X_p2_drift[col] * np.random.uniform(2.5, 4.0, size=len(X_p2_drift))
            
    os.makedirs("data", exist_ok=True)
    X_train.assign(Class=y_train).to_csv("data/datos_entrenamiento.csv", index=False)
    X_p1.assign(Class=y_p1).to_csv("data/prod_dia1_normal.csv", index=False)
    X_p2_drift.assign(Class=y_p2).to_csv("data/prod_dia2_drift.csv", index=False)
    print("[INFO] ✅ Archivos de datos 100% numéricos ordenados cronológicamente en 'data/'.")

if __name__ == "__main__":
    preparar_ieee_fraud()