"""Prueba ligera del monitor opcional sin ejecutar modelos."""
import unittest

from src.experiments.resources import ResourceSampler, psutil


@unittest.skipIf(psutil is None, "psutil no instalado (profiler opcional)")
class ResourceSamplerTests(unittest.TestCase):
    def test_peak_rss_and_cpu_are_nonnegative(self):
        with ResourceSampler(interval_seconds=0.01) as sampler:
            _ = sum(i * i for i in range(100000))
        result = sampler.result()
        self.assertGreater(result["peak_tree_rss_mib"], 0)
        self.assertGreaterEqual(result["process_tree_cpu_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
