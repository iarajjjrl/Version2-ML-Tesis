import time
import subprocess
from generate_synthetic import generar_lote_sintetico
from monitor import evaluar_lote_produccion

def ejecutar_git_pull_automatico():
    """Ejecuta git pull mediante Python para actualizar el modelo local."""
    print("\n[AUTO-PULL] Descargando el nuevo modelo reentrenado desde GitHub...")
    try:
        resultado = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[AUTO-PULL] ✅ Repositorio actualizado con éxito:")
        print(f"            {resultado.stdout.strip()}")
    except Exception as e:
        print(f"[AUTO-PULL] ❌ Error al ejecutar git pull: {str(e)}")

def ejecutar_streaming_sintetico():
    print("==========================================================")
    print("  SIMULADOR DE INGESTA TRANSACCIONAL SINTÉTICA (MLOps)    ")
    print("==========================================================\n")
    
    pasos_simulacion = [
        {"paso": 1, "drift": False, "descripcion": "Lote Sintético Normal - Operación Habitual"},
        {"paso": 2, "drift": True,  "descripcion": "Lote Sintético CON DRIFT - Ataque de Fraude Evasivo"},
        {"paso": 3, "drift": False, "descripcion": "Lote Sintético Post-Reentrenamiento (Validación)"},
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
        # NOTA: evaluamos y capturamos si la alerta de drift fue disparada
        # Para esto, evaluamos el lote sintético generado
        evaluar_lote_produccion(ruta_lote, "models/modelo_vigente.pkl")
        
        # Si acabamos de simular el lote con Drift (Paso 2), esperamos a la nube y traemos el nuevo modelo
        if estado['drift']:
            print("\n[STREAM] ⏳ Pausa de 60 segundos: Esperando que GitHub Actions termine el reentrenamiento en la nube...")
            time.sleep(60) # Tiempo de espera para que Optuna corra en Actions
            
            # 3. Traer el nuevo modelo actualizado desde GitHub de forma automática
            ejecutar_git_pull_automatico()
        else:
            print("[STREAM] Pausa de 10 segundos antes de la siguiente ingesta...")
            time.sleep(10)

if __name__ == "__main__":
    ejecutar_streaming_sintetico()