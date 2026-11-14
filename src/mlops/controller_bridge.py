"""Persistencia segura del mismo controlador usado en el Estudio A v2.

Esta capa no altera la lógica de decisión del controlador. Solo conserva el
estado entre lotes, evita procesarlos fuera de orden y registra las decisiones.
"""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path

from src.experiments.v2.controller import MaintenanceController, PolicySettings

STATE_VERSION = 1
FIELDS = (
    "active_model", "last_window", "last_attempt", "low_streak",
    "attempts", "promotions", "pending", "awaiting_registration",
    "reference_f1", "reference_pending",
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@contextmanager
def exclusive_lock(directory):
    """El bloqueo falla cerrado ante procesos simultáneos o bloqueos huérfanos."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".controller.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def controller_to_dict(controller):
    return {name: getattr(controller, name) for name in FIELDS}


def controller_from_dict(settings, state):
    ctrl = MaintenanceController(settings, state["active_model"])
    for name in FIELDS:
        if name not in state:
            raise ValueError(f"Falta campo de controlador: {name}")
        value = state[name]
        if name == "pending" and value is not None:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("Candidato pendiente inválido")
            value = tuple(value)
        setattr(ctrl, name, value)
    if ctrl.awaiting_registration is not None:
        raise ValueError("Transacción interrumpida: se necesita auditoría manual")
    if not 0 <= ctrl.last_window <= settings.n_windows:
        raise ValueError("Índice temporal inválido")
    return ctrl


def create_state(settings, model_id, metadata=None):
    ctrl = MaintenanceController(settings, model_id)
    return {
        "schema": STATE_VERSION,
        "settings": asdict(settings),
        "controller": controller_to_dict(ctrl),
        "processed": [],
        "metadata": metadata or {},
    }


def load_state(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != STATE_VERSION:
        raise ValueError("Versión de estado no soportada")
    settings = PolicySettings(**data["settings"])
    ctrl = controller_from_dict(settings, data["controller"])
    if len(data["processed"]) != ctrl.last_window:
        raise ValueError("Registro temporal y controlador no coinciden")
    return data, ctrl


def complete_observation(state, controller, *, window, observed_f1,
                         candidate_f1=None, incumbent_f1=None,
                         train_callback=None, batch_sha256=None):
    """Una ventana, una decisión y, si corresponde, un candidato.

    El callback se ejecuta DESPUÉS de puntuar el vigente en k y ANTES de
    persistir el estado. La promoción se decide en k+1 y afecta el uso en k+2.
    """
    if window != controller.last_window + 1:
        raise ValueError("Se exige la siguiente ventana sin saltos ni repeticiones")
    scoring_model = controller.active_model
    pending = controller.pending
    if pending and (candidate_f1 is None or incumbent_f1 is None):
        raise ValueError("Falta puntuar el candidato pendiente en la nueva ventana")
    if not pending and (candidate_f1 is not None or incumbent_f1 is not None):
        raise ValueError("No se pueden pasar métricas de un candidato inexistente")
    decision = controller.complete_window(
        window, observed_f1, candidate_f1=candidate_f1, incumbent_f1=incumbent_f1)
    trained_model = None
    training_error = None
    if decision.execute:
        if train_callback is None:
            controller.record_training_failure(window)
            training_error = "No se configuró entrenador"
        else:
            try:
                trained_model = train_callback(window)
                controller.register_candidate(window, trained_model)
            except Exception as exc:
                controller.record_training_failure(window)
                training_error = f"{type(exc).__name__}: {exc}"
    event = {
        "window": window,
        "scoring_model": scoring_model,
        "active_after_window": controller.active_model,
        "requested": decision.requested,
        "train_attempt": decision.execute,
        "reason": decision.reason,
        "candidate_reviewed": decision.candidate_reviewed,
        "candidate_approved": decision.candidate_approved,
        "candidate_created": trained_model,
        "training_error": training_error,
        "attempts": controller.attempts,
        "promotions": controller.promotions,
        "observed_f1": observed_f1,
        "candidate_f1": candidate_f1,
        "incumbent_f1": incumbent_f1,
        "batch_sha256": batch_sha256,
    }
    state["controller"] = controller_to_dict(controller)
    state["processed"].append(event)
    return event
