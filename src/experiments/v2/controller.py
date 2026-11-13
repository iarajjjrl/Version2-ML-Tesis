"""Control temporal de mantenimiento: solicitud k, promoción k+1, uso k+2.

No entrena modelos ni consume las etiquetas de una ventana futura. Este
controlador implementa únicamente la secuencia de decisión especificada en
el protocolo v2. Las métricas se calculan fuera de este módulo.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PolicySettings:
    policy: str
    n_windows: int = 8
    every: int = 4
    f1_trigger: float = .50
    consecutive_low: int = 2
    cooldown_windows: int = 1
    min_delta: float = .01
    min_promotion_f1: float = 0.
    max_attempts: int | None = None
    # 'absolute' conserva el protocolo original (F1 < f1_trigger).
    # 'relative' exige un F1 inicial de holdout NO usado para selección.
    trigger_mode: str = "absolute"
    relative_drop: float = .15
    reference_f1: float | None = None

    def __post_init__(self):
        if self.policy not in {"static", "periodic", "event"}:
            raise ValueError("Política no soportada")
        if (self.n_windows < 3 or self.every < 1 or self.consecutive_low < 1
                or self.cooldown_windows < 0):
            raise ValueError("Parámetros de ventana inválidos")
        if any(not 0 <= x <= 1 for x in
               (self.f1_trigger, self.min_delta, self.min_promotion_f1)):
            raise ValueError("Umbral fuera de [0,1]")
        if self.max_attempts is not None and self.max_attempts < 0:
            raise ValueError("max_attempts no puede ser negativo")
        if self.trigger_mode not in {"absolute", "relative"}:
            raise ValueError("Modo de disparo no soportado")
        if not 0 < self.relative_drop < 1:
            raise ValueError("relative_drop debe encontrarse estrictamente entre 0 y 1")
        if self.reference_f1 is not None and not (
                math.isfinite(self.reference_f1) and 0 < self.reference_f1 <= 1):
            raise ValueError("F1 de referencia debe estar en (0, 1]")
        if (self.policy == "event" and self.trigger_mode == "relative"
                and self.reference_f1 is None):
            raise ValueError("El disparador relativo requiere F1 de referencia inicial")


@dataclass(frozen=True)
class WindowDecision:
    window: int
    scoring_model: str
    model_for_next_window: str
    requested: bool
    execute: bool
    reason: str
    candidate_reviewed: bool
    candidate_approved: bool | None
    attempts: int
    promotions: int
    reference_f1: float | None = None
    trigger_threshold: float | None = None


class MaintenanceController:
    """El evaluador llama complete_window tras puntuar el modelo activo.

    Si complete_window pide entrenamiento, el evaluador debe llamar a
    register_candidate o record_training_failure antes de la ventana siguiente.
    Cuando hay candidato pendiente, complete_window(k+1, ...) exige los F1
    del candidato y del incumbente en k+1. El candidato aprobado solo se
    utilizará para puntuar a partir de la ventana k+2.
    """

    def __init__(self, settings: PolicySettings, initial_model: str):
        if not initial_model:
            raise ValueError("Falta modelo inicial")
        self.settings = settings
        self.active_model = initial_model
        self.last_window = 0
        self.last_attempt = None
        self.low_streak = 0
        self.attempts = 0
        self.promotions = 0
        self.pending = None  # (created_window, model_id)
        self.awaiting_registration = None
        self.reference_f1 = settings.reference_f1
        # Una referencia de modelo NUEVO solo puede construirse cuando este
        # recibe su primera ventana k+2 sin haberla usado para promoción.
        self.reference_pending = False

    def complete_window(self, window: int, observed_f1: float,
                        candidate_f1: float | None = None,
                        incumbent_f1: float | None = None) -> WindowDecision:
        s = self.settings
        if window != self.last_window + 1 or window > s.n_windows:
            raise ValueError("Ventanas deben procesarse de forma cronológica y consecutiva")
        if self.awaiting_registration is not None:
            raise RuntimeError("Registrar candidato o fallo del intento anterior")
        if not math.isfinite(observed_f1) or not 0 <= observed_f1 <= 1:
            raise ValueError("F1 observado no válido")

        scoring_model = self.active_model
        had_pending = self.pending is not None
        reviewed = False
        approved = None
        if had_pending:
            created, candidate_id = self.pending
            if window != created + 1:
                raise RuntimeError("La promoción solo se evalúa en la ventana siguiente")
            if candidate_f1 is None or incumbent_f1 is None:
                raise ValueError("Faltan métricas de candidato e incumbente para promoción")
            if not all(math.isfinite(v) and 0 <= v <= 1
                       for v in (candidate_f1, incumbent_f1)):
                raise ValueError("F1 de promoción no válido")
            approved = (candidate_f1 - incumbent_f1 > 0 and
                        candidate_f1 - incumbent_f1 + 1e-12 >= s.min_delta and
                        candidate_f1 >= s.min_promotion_f1)
            reviewed = True
            if approved:
                self.active_model = candidate_id
                self.promotions += 1
                if s.policy == "event" and s.trigger_mode == "relative":
                    self.reference_pending = True
            self.pending = None
        elif candidate_f1 is not None or incumbent_f1 is not None:
            raise ValueError("Métricas de promoción sin candidato pendiente")

        if s.policy == "static":
            due, reason = False, "static_no_updates"
        elif s.policy == "periodic":
            due = window % s.every == 0
            reason = "periodic_due" if due else "periodic_not_due"
        elif reviewed and approved:
            # El F1 de k+1 pertenece al incumbente; nunca fija la referencia nueva.
            self.low_streak = 0
            due, reason = False, "new_model_not_evaluated_until_next_window"
        elif s.trigger_mode == "relative" and self.reference_pending:
            # Primera puntuación real del nuevo modelo (k+2), no usada para
            # entrenamiento, ajuste de umbral ni selección del challenger.
            self.reference_f1 = observed_f1 if observed_f1 > 0 else None
            self.reference_pending = False
            self.low_streak = 0
            due, reason = False, "reference_established_from_first_scoring_window"
        elif s.trigger_mode == "relative" and self.reference_f1 is None:
            # Si la primera ventana independiente da F1=0, no hay escala
            # relativa positiva. Registrar la limitación, sin inventar umbral.
            self.low_streak = 0
            due, reason = False, "reference_unavailable_zero_f1"
        else:
            threshold = (s.f1_trigger if s.trigger_mode == "absolute" else
                         self.reference_f1 * (1. - s.relative_drop))
            self.low_streak = self.low_streak + 1 if observed_f1 < threshold else 0
            due = self.low_streak >= s.consecutive_low
            reason = ("event_due" if due else
                      "waiting_for_persistent_degradation" if self.low_streak else "event_healthy")

        execute = False
        if due and had_pending:
            reason = "deferred_candidate_reviewed_this_window"
        elif due and self.last_attempt is not None and window - self.last_attempt <= s.cooldown_windows:
            reason = "cooldown"
        elif due and window + 2 > s.n_windows:
            reason = "end_of_horizon_no_independent_evaluation"
        elif due and s.max_attempts is not None and self.attempts >= s.max_attempts:
            reason = "budget_exhausted"
        elif due:
            execute = True
            self.attempts += 1
            self.last_attempt = window
            self.awaiting_registration = window
            if s.policy == "event":
                self.low_streak = 0

        self.last_window = window
        active_threshold = None
        if s.policy == "event":
            if s.trigger_mode == "absolute":
                active_threshold = s.f1_trigger
            elif self.reference_f1 is not None:
                active_threshold = self.reference_f1 * (1. - s.relative_drop)
        return WindowDecision(window, scoring_model, self.active_model,
                              due, execute, reason, reviewed, approved,
                              self.attempts, self.promotions, self.reference_f1,
                              active_threshold)

    def register_candidate(self, window: int, candidate_model_id: str) -> None:
        if self.awaiting_registration != window or self.pending is not None:
            raise RuntimeError("No hay intento de entrenamiento reservado")
        if not candidate_model_id or candidate_model_id == self.active_model:
            raise ValueError("El candidato debe ser una versión diferente")
        self.pending = (window, candidate_model_id)
        self.awaiting_registration = None

    def record_training_failure(self, window: int) -> None:
        if self.awaiting_registration != window:
            raise RuntimeError("No hay intento reservado para marcar como fallido")
        self.awaiting_registration = None
