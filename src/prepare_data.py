import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preparar_y_generar_lotes_fraude(ruta_csv_input):
    print("[INFO] Cargando dataset de tarjetas de credito...")
    df = pd.read_csv(ruta_csv_input)
    
    if 'Class' not in df.columns:
        raise ValueError("El archivo CSV debe contener la columna objetivo 'Class'")
        
    os.makedirs("data", exist_ok=True)
    
    # 1. Escalamiento de Amount y Time
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Amount', 'Time'])
    
    # 2. Muestreo estratificado del 25% para optimizar peso en Git y FinOps
    X_full = df.drop(columns=['Class'])
    y_full = df['Class']
    
    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=0.25, random_state=42, stratify=y_full
    )
    
    # 3. Separar 60% Entrenamiento y 40% Producción
    X_train, X_prod, y_train, y_prod = train_test_split(
        X_sample, y_sample, test_size=0.40, random_state=42, stratify=y_sample
    )
    
    # 4. Dividir Producción en Día 1 (Normal) y Día 2 (Drift)
    X_p1, X_p2, y_p1, y_p2 = train_test_split(
        X_prod, y_prod, test_size=0.50, random_state=42, stratify=y_prod
    )
    
    # 5. Inyección de Data Drift en Día 2
    X_p2_drift = X_p2.copy()
    cols_alterar = ['V1', 'V2', 'V3', 'V4', 'V5']
    for col in cols_alterar:
        X_p2_drift[col] = X_p2_drift[col] * np.random.uniform(2.5, 4.0, size=len(X_p2_drift))
        
    # Guardar los archivos de salida
    X_train.assign(Class=y_train).to_csv("data/datos_entrenamiento.csv", index=False)
    X_p1.assign(Class=y_p1).to_csv("data/prod_dia1_normal.csv", index=False)
    X_p2_drift.assign(Class=y_p2).to_csv("data/prod_dia2_drift.csv", index=False)
    
    print("[INFO] Lotes optimizados generados correctamente en la carpeta data/")

if __name__ == "__main__":
    preparar_y_generar_lotes_fraude("creditcard.csv")