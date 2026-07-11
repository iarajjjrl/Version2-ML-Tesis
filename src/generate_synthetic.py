import os

import pandas as pd

import numpy as np



def generar_lote_sintetico(

    ruta_base_csv="data/datos_entrenamiento.csv",

    ruta_salida="data/lote_sintetico_actual.csv",

    n_legitimas=3900,

    n_fraudes=100,

    inyectar_drift=False,

    factor_drift=3.0

):

    """

    Genera transacciones sintéticas aplicando Muestreo Gaussiano Multivariable

    y saneando los tipos de datos para preservar el esquema entero/categórico de IEEE-CIS.

    """

    print(f"[SINTETICO] Cargando datos base para modelar distribución desde '{ruta_base_csv}'...")

    if not os.path.exists(ruta_base_csv):

        raise FileNotFoundError(f"No se encontró el archivo base de datos en '{ruta_base_csv}'. Ejecuta prepare_data.py primero.")

       

    df_base = pd.read_csv(ruta_base_csv)

   

    # Separar clases reales para modelar cada distribución independientemente

    df_legitimas = df_base[df_base['Class'] == 0].drop(columns=['Class'])

    df_fraudes = df_base[df_base['Class'] == 1].drop(columns=['Class'])

   

    print(f"[SINTETICO] Sintetizando {n_legitimas} transacciones legítimas y {n_fraudes} fraudes...")

   

    # 1. Muestreo Multivariable Gaussiano para Transacciones Legítimas

    mean_leg = df_legitimas.mean()

    cov_leg = df_legitimas.cov()

    sintetico_leg = np.random.multivariate_normal(mean_leg, cov_leg, size=n_legitimas)

    df_sint_leg = pd.DataFrame(sintetico_leg, columns=df_legitimas.columns)

    df_sint_leg['Class'] = 0

   

    # 2. Muestreo Multivariable Gaussiano para Transacciones Fraudulentas

    mean_frd = df_fraudes.mean()

    cov_frd = df_fraudes.cov()

    sintetico_frd = np.random.multivariate_normal(mean_frd, cov_frd, size=n_fraudes)

    df_sint_frd = pd.DataFrame(sintetico_frd, columns=df_fraudes.columns)

    df_sint_frd['Class'] = 1

   

    # 3. Concatenar y reordenar aleatoriamente

    df_sintetico = pd.concat([df_sint_leg, df_sint_frd]).sample(frac=1.0, random_state=42).reset_index(drop=True)

   

    # 4. SANEAMIENTO DE TIPOS DE DATOS (Alineación con esquema IEEE-CIS)

    cols_enteras = ['card1', 'card2', 'card4', 'card6', 'ProductCD', 'addr1', 'addr2',

                    'dist1', 'C1', 'C2', 'C5', 'C6', 'C11', 'C13', 'D1', 'D4', 'DeviceType']

   

    for col in cols_enteras:

        if col in df_sintetico.columns:

            df_sintetico[col] = np.clip(np.round(df_sintetico[col]), a_min=0, a_max=None).astype(int)

           

    if 'TransactionAmt' in df_sintetico.columns:

        df_sintetico['TransactionAmt'] = np.clip(df_sintetico['TransactionAmt'], a_min=0.01, a_max=None)

   

    # 5. Inyección Opcional de Data Drift en variables de negocio

    if inyectar_drift:

        print(f"[SINTETICO] ⚠️ Inyectando Data Drift (factor {factor_drift}) en variables críticas (TransactionAmt, card1, C1, C13)...")

        df_sintetico['TransactionAmt'] = df_sintetico['TransactionAmt'] * np.random.uniform(2.5, 4.0, size=len(df_sintetico))

        df_sintetico['card1'] = (df_sintetico['card1'] * np.random.uniform(1.5, 3.0, size=len(df_sintetico))).astype(int)

        df_sintetico['C1'] = (df_sintetico['C1'] * np.random.uniform(2.0, 3.5, size=len(df_sintetico))).astype(int)

        df_sintetico['C13'] = (df_sintetico['C13'] * np.random.uniform(2.0, 3.5, size=len(df_sintetico))).astype(int)

       

    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)

    df_sintetico.to_csv(ruta_salida, index=False)

    print(f"[SINTETICO] Lote sintético saneado generado en '{ruta_salida}' ({len(df_sintetico)} filas).\n")

    return ruta_salida



if __name__ == "__main__":

    generar_lote_sintetico() 

