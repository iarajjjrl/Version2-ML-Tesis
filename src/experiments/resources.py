"""Sampler opcional del proceso de AutoML para costo local.

Incluye proceso Python y procesos hijos visibles durante el muestreo.
No mide GPU, energía, cloud billing ni el sistema operativo completo.
"""
import os
import threading

try:
    import psutil
except ImportError:
    psutil = None


class ResourceSampler:
    def __init__(self, interval_seconds=0.10):
        self.interval_seconds = interval_seconds
        self.peak_tree_rss_bytes = 0
        self.process_tree_cpu_seconds = 0.0
        self._last_cpu_seconds = 0.0
        self._initial_cpu_seconds = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _sample(self):
        tree_rss = 0
        tree_cpu = 0.0
        try:
            root = psutil.Process(os.getpid())
            processes = [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = []
        for p in processes:
            try:
                tree_rss += p.memory_info().rss
                usage = p.cpu_times()
                tree_cpu += usage.user + usage.system
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        self.peak_tree_rss_bytes = max(self.peak_tree_rss_bytes, tree_rss)
        self._last_cpu_seconds = max(self._last_cpu_seconds, tree_cpu)

    def _poll(self):
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        if psutil is None:
            raise RuntimeError(
                "Instala psutil en tu venv para medir recursos: "
                r".\venv\Scripts\python.exe -m pip install psutil"
            )
        self._sample()
        self._initial_cpu_seconds = self._last_cpu_seconds
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        self._sample()
        self.process_tree_cpu_seconds = max(
            0.0, self._last_cpu_seconds - self._initial_cpu_seconds
        )

    def result(self):
        return {
            "peak_tree_rss_mib": round(
                self.peak_tree_rss_bytes / (1024 ** 2), 3
            ),
            "process_tree_cpu_seconds": round(
                self.process_tree_cpu_seconds, 4
            ),
            "sampling_interval_seconds": self.interval_seconds,
        }
