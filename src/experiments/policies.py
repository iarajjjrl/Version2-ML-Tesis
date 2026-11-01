"""Políticas experimentales puras; sin dependencias de GitHub ni de ML."""

POLICIES = ("static", "periodic", "event")


def decide_update(policy, window, f1, f1_trigger, every,
                  consecutive_low, cooldown_windows, low_streak,
                  last_trigger_window):
    """Devuelve (activar, nuevo_streak, motivo). La evaluación precede al entrenamiento."""
    if policy not in POLICIES:
        raise ValueError(f"Política desconocida: {policy}")
    if policy == "static":
        return False, 0, "no_updates"
    if policy == "periodic":
        trigger = window % every == 0
        return trigger, 0, "periodic_due" if trigger else "not_due"

    streak = low_streak + 1 if f1 < f1_trigger else 0
    if streak < consecutive_low:
        return False, streak, "waiting_for_persistent_degradation" if streak else "healthy"
    if (last_trigger_window is not None
            and window - last_trigger_window <= cooldown_windows):
        return False, streak, "cooldown"
    return True, 0, "f1_degradation"


def approve_promotion(candidate_f1, incumbent_f1, min_delta, min_f1):
    """Empates no promueven. Ambos F1 se evalúan sobre los mismos datos."""
    if candidate_f1 < min_f1:
        return False
    delta = candidate_f1 - incumbent_f1
    return delta > 0 and delta + 1e-12 >= min_delta
