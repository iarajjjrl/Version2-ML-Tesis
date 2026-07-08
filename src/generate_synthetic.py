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
    
    # Concatenar lote sintético
    df_sintetico = pd.concat([df_sint_leg, df_sint_frd]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 3. Inyección Opcional de Data Drift (Simulación de un patrón de fraude evasivo o cambio de mercado)
    if inyectar_drift:
        print("[SINTETICO] ⚠️ Inyectando Data Drift en variables criticas (V14, V10, V12, scaled_amount)...")
        # Alteramos patrones que desestabilizan las fronteras de decision del modelo
        df_sintetico['V14'] = df_sintetico['V14'] * np.random.uniform(2.0, 3.5, size=len(df_sintetico))
        df_sintetico['V10'] = df_sintetico['V10'] - np.random.uniform(1.5, 3.0, size=len(df_sintetico))
        df_sintetico['scaled_amount'] = df_sintetico['scaled_amount'] * np.random.uniform(3.0, 5.0, size=len(df_sintetico))
        
    os.makedirs("data", exist_ok=True)
    ruta_salida = "data/lote_sintetico_actual.csv"
    df_sintetico.to_csv(ruta_salida, index=False)
    
    print(f"[SINTETICO] Lote sintetico generado exitosamente en '{ruta_salida}' ({len(df_sintetico)} filas).\n")
    return ruta_salida

if __name__ == "__main__":
    # Prueba individual: Generar lote normal y lote con drift
    generar_lote_sintetico("data/datos_entrenamiento.csv", num_muestras=3000, inyectar_drift=False)