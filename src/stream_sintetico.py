import time
from generate_synthetic import generar_lote_sintetico
from monitor import evaluar_lote_produccion

def ejecutar_streaming_sintetico():
    print("==========================================================")
    print("  SIMULADOR DE INGESTA TRANSACCIONAL SINTÉTICA (MLOps)    ")
    print("==========================================================\n")
    
    pasos_simulacion = [
        {"paso": 1, "drift": False, "descripcion": "Lote Sintético Normal - Operación Habitual"},
        {"paso": 2, "drift": False, "descripcion": "Lote Sintético Normal - Operación Habitual"},
        {"paso": 3, "drift": True,  "descripcion": "Lote Sintético CON DRIFT - Ataque de Fraude Evasivo"},
    ]
    
    for estado in pasos_simulacion:
        print(f"\n[STREAM] --- SIMULANDO VENTANA #{estado['paso']}: {estado['descripcion']} ---")
        
        # 1. Generar la data sintética en tiempo real
        ruta_lote = generar_lote_sintetico(
            ruta_base_csv="data/datos_entrenamiento.csv",
            num_muestras=4000,
            proporcion_fraude=0.025,
            inyectar_drift=estado['drift']
        )
        
        # 2. Evaluar el modelo vigente contra la data recién generada
        evaluar_lote_produccion(ruta_lote, "models/modelo_vigente.pkl")
        
        print("[STREAM] Pausa de 10 segundos antes de la siguiente ingesta...")
        time.sleep(10)

if __name__ == "__main__":
    ejecutar_streaming_sintetico()