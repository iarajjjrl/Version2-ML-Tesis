"""Watcher experimental AISLADO para lotes raw-v2 preparados y verificados.

No utiliza src/folder_watcher.py, no realiza git push ni dispara el flujo
historico de GitHub Actions. Ejecuta el mismo controlador del Estudio A.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from src.mlops.controlled_v2 import step
from src.mlops.controller_bridge import load_state


def expected_name(store):
    state, controller = load_state(Path(store) / "state.json")
    raw = Path(state["metadata"]["raw_dir"])
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    if controller.last_window >= controller.settings.n_windows:
        return None
    return manifest["windows"][controller.last_window]["file"]


class ControlledIngestHandler:
    def __init__(self, store, incoming, processed):
        self.store = Path(store)
        self.incoming = Path(incoming)
        self.processed = Path(processed)

    def process(self, path):
        file = Path(path)
        if not file.is_file() or file.suffix.lower() != ".csv":
            return
        name = expected_name(self.store)
        if name is None:
            print("[CONTROL] Todas las ventanas fueron procesadas", flush=True)
            return
        if file.name != name:
            print(f"[CONTROL] Esperando {name}. Se ignoró {file.name}", flush=True)
            return
        # No usar un CSV cuya escritura todavía no terminó.
        size = -1
        stable = 0
        for _ in range(25):
            try:
                next_size = file.stat().st_size
            except FileNotFoundError:
                return
            stable = stable + 1 if next_size == size and next_size > 0 else 0
            if stable >= 2:
                break
            size = next_size
            time.sleep(.25)
        if stable < 2:
            print(f"[CONTROL] Archivo todavía en escritura: {name}", flush=True)
            return
        try:
            self.processed.mkdir(parents=True, exist_ok=True)
            target = self.processed / file.name
            if target.exists():
                raise FileExistsError(f"El archivo procesado ya existe: {target}")
            outcome = step(self.store, batch_file=file)
            shutil.move(str(file), str(target))
            print(f"[CONTROL] Ventana {outcome['window']}, F1 {outcome['observed_f1']:.4f}, "
                  f"entrenar={outcome['train_attempt']}, "
                  f"promocion={outcome['candidate_approved']}", flush=True)
        except Exception as exc:
            # Conservar el lote, no omitir una ventana por un error.
            print(f"[CONTROL][ERROR] {type(exc).__name__}: {exc}", flush=True)

def main():
    p = argparse.ArgumentParser(description="Ingesta v2 sin publicar modelos")
    p.add_argument("--store", required=True)
    p.add_argument("--incoming", required=True)
    p.add_argument("--processed", required=True)
    args = p.parse_args()
    ingest = Path(args.incoming).resolve()
    ingest.mkdir(parents=True, exist_ok=True)
    handler = ControlledIngestHandler(args.store, ingest, args.processed)
    print(f"[CONTROL] Vigilando {ingest}, ventana esperada: {expected_name(args.store)}",
          flush=True)
    try:
        while True:
            next_name = expected_name(args.store)
            if next_name is not None and (ingest / next_name).is_file():
                handler.process(ingest / next_name)
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
