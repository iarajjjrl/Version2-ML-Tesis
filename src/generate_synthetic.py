import pandas as pd
import numpy as np
import os

def generar_lote_sintetico(ruta_base_csv, num_muestras=5000, proporcion_fraude=0.02, inyectar_drift=False):
    """
    Genera un lote de transacciones sinteticas basadas en la distribucion
    estadistica multivariable del dataset real.
    """
    print(f"[SINTETICO] Cargando datos base para modelar distribucion...")
    df_base = pd.read_csv(ruta_base_csv)
    
    # Separar clases reales para modelar cada distribucion independientemente
    df_legitimas = df_base[df_base['Class'] == 0].drop(columns=['Class'])
    df_fraudes = df_base[df_base['Class'] == 1].drop(columns=['Class'])
    
    num_fraudes = int(num_muestras * proporcion_fraude)
    num_legitimas = num_muestras - num_fraudes
    
    print(f"[SINTETICO] Sintetizando {num_legitimas} transacciones legitimas y {num_fraudes} fraudes...")
    
    # 1. Muestreo Multivariable Gaussiano para Transacciones Legitimas
    mean_leg = df_legitimas.mean()
    cov_leg = df_legitimas.cov()
    sintetico_leg = np.random.multivariate_normal(mean_leg, cov_leg, size=num_legitimas)
    df_sint_leg = pd.DataFrame(sintetico_leg, columns=df_legitimas.columns)
    df_sint_leg['Class'] = 0
    
    # 2. Muestreo Multivariable Gaussiano para Transacciones Fraudulentas
    mean_frd = df_fraudes.mean()
    cov_frd = df_fraudes.cov()
    sintetico_frd = np.random.multivariate_normal(mean_frd, cov_frd, size=num_fraudes)
    df_sint_frd = pd.DataFrame(sintetico_frd, columns=df_fraudes.columns)
    df_sint_frd['Class'] = 1
    
    # 3. Concatenar lote sintético unificado
    df_sintetico = pd.concat([df_sint_leg, df_sint_frd]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 4. Inyección Opcional de Data Drift en variables reales de IEEE-CIS
    if inyectar_drift:
        print("[SINTETICO] ⚠️ Inyectando Data Drift en variables criticas (TransactionAmt, card1, C1, C13)...")
        df_sintetico['TransactionAmt'] = df_sintetico['TransactionAmt'] * np.random.uniform(2.5, 4.0, size=len(df_sintetico))
        df_sintetico['card1'] = df_sintetico['card1'] * np.random.uniform(1.5, 3.0, size=len(df_sintetico))
        df_sintetico['C1'] = df_sintetico['C1'] * np.random.uniform(2.0, 3.5, size=len(df_sintetico))
        df_sintetico['C13'] = df_sintetico['C13'] * np.random.uniform(2.0, 3.5, size=len(df_sintetico))
        
    os.makedirs("data", exist_ok=True)
    ruta_salida = "data/lote_sintetico_actual.csv"
    df_sintetico.to_csv(ruta_salida, index=False)
    
    print(f"[SINTETICO] Lote sintetico generado exitosamente en '{ruta_salida}' ({len(df_sintetico)} filas).\n")
    return ruta_salida

if __name__ == "__main__":
    generar_lote_sintetico("data/datos_entrenamiento.csv", num_muestras=3000, inyectar_drift=False)