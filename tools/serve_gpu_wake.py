"""Local Strata launcher using the existing proxy's GpuPowerManager (without running the proxy).

The wake process sees only --wake-device; engine/vision visibility comes from their
config. The retained CUDA probe starts after engine READY, outside cache sizing.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_server(server_args, module_dir, wake_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = wake_device
    sys.path.insert(0, str(Path(module_dir).resolve()))
    from gpu_monitor import GpuPowerManager
    from serve import server

    power = GpuPowerManager()
    power.init_nvml()
    if not power.nvml_initialized:
        raise RuntimeError("GPU wake: NVML initialization failed")
    lock = threading.RLock()
    timer = None
    active = 0

    def relax():
        with lock:
            if not active:
                power._relax_sync()

    def initialize():
        with lock:
            if not power.cuda_kick_ready:
                power.init_cuda_kickstarter()
                if not power.cuda_kick_ready:
                    raise RuntimeError("GPU wake: retained CUDA probe failed")

    base = server.StrataEngine

    class AwakeEngine(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.alive():
                initialize()

        def generate(self, *args, **kwargs):
            nonlocal timer, active
            with lock:
                if timer:
                    timer.cancel()
                initialize()
                power._boost_sync()
                active += 1
            try:
                yield from super().generate(*args, **kwargs)
            finally:
                with lock:
                    active -= 1
                    if not active:
                        timer = threading.Timer(120.0, relax)
                        timer.daemon = True
                        timer.start()

    original_argv = sys.argv
    try:
        # No retained Torch allocation during model loading/cache budgeting.
        power._boost_sync()
        server.StrataEngine = AwakeEngine
        sys.argv = ["serve.server", *server_args]
        return server.main()
    finally:
        with lock:
            if timer:
                timer.cancel()
            power.shutdown()
        server.StrataEngine = base
        sys.argv = original_argv


def main():
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--power-module-dir", required=True, help="directory containing existing gpu_monitor.py")
    ap.add_argument("--wake-device", required=True, help="CUDA UUID of the GPU needing the wake pulse")
    a, rest = ap.parse_known_args()
    return run_server(rest, a.power_module_dir, a.wake_device)


if __name__ == "__main__":
    raise SystemExit(main())
