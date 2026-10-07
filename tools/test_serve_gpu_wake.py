"""Wake allocation must follow engine cache sizing, and failures must release clocks."""
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from serve_gpu_wake import run_server
from serve import server


class WakeLifecycle(unittest.TestCase):
    def run_case(self, fail_probe=False, fail_generate=False):
        events = []

        class Power:
            nvml_initialized = True
            cuda_kick_ready = False

            def init_nvml(self): events.append("nvml")
            def init_cuda_kickstarter(self):
                events.append("probe")
                self.cuda_kick_ready = not fail_probe
            def _boost_sync(self): events.append("boost")
            def _relax_sync(self): events.append("relax")
            def shutdown(self): events.append("shutdown")

        class Engine:
            def __init__(self): events.append("ready")
            def alive(self): return True
            def generate(self):
                events.append("generate")
                if fail_generate: raise ValueError("failed generation")
                yield 123

        def main():
            e = server.StrataEngine()
            list(e.generate())
            return 0

        module = SimpleNamespace(GpuPowerManager=Power)
        timer = mock.MagicMock()
        with mock.patch.dict(sys.modules, gpu_monitor=module), mock.patch.dict(os.environ), \
             mock.patch.object(sys, "path", list(sys.path)), mock.patch.object(server, "StrataEngine", Engine), \
             mock.patch.object(server, "main", main), mock.patch("serve_gpu_wake.threading.Timer", return_value=timer):
            old_argv = sys.argv
            if fail_probe or fail_generate:
                with self.assertRaises((RuntimeError, ValueError)):
                    run_server([], ".", "GPU-wake")
            else:
                self.assertEqual(run_server([], ".", "GPU-wake"), 0)
            self.assertIs(server.StrataEngine, Engine)
            self.assertIs(sys.argv, old_argv)
        self.assertEqual(events[-1], "shutdown")
        self.assertLess(events.index("ready"), events.index("probe"))
        if not fail_probe:
            self.assertEqual(events[:5], ["nvml", "boost", "ready", "probe", "boost"])
            timer.start.assert_called_once()
            timer.cancel.assert_called_once()

    def test_ready_before_probe_and_shutdown(self): self.run_case()
    def test_failed_probe_releases_clocks(self): self.run_case(fail_probe=True)
    def test_failed_generation_releases_clocks(self): self.run_case(fail_generate=True)


if __name__ == "__main__":
    unittest.main()
