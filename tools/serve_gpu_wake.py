"""Local Strata launcher using the existing proxy's GpuPowerManager (without running the proxy).

The wake process sees only --wake-device; engine/vision visibility comes from their
config. The retained CUDA probe starts after engine READY, outside cache sizing.
Clock locks expire after 120 idle seconds, including a start with no requests.
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
    stopping = False

    # These helpers run under lock, including callbacks racing with cancellation.
    def cancel_idle():
        nonlocal timer
        if timer is not None:
            timer.cancel()
            timer = None

    def schedule_idle():
        nonlocal timer
        cancel_idle()
        if active or stopping:
            return

        def relax():
            nonlocal timer
            with lock:
                # Timer.cancel() cannot stop a callback already waiting for lock.
                # An obsolete callback must not shorten a newer idle interval.
                if timer is pending and not active:
                    timer = None
                    power._relax_sync()

        pending = threading.Timer(120.0, relax)
        pending.daemon = True
        timer = pending
        pending.start()

    def initialize():
        with lock:
            if not power.cuda_kick_ready:
                power.init_cuda_kickstarter()
                if not power.cuda_kick_ready:
                    raise RuntimeError("GPU wake: retained CUDA probe failed")

    base = server.StrataEngine

    class AwakeEngine(base):
        def __init__(self, *args, **kwargs):
            with lock:
                cancel_idle()
                # Apply on reload too; no retained Torch allocation on first load.
                power._boost_sync()
            try:
                super().__init__(*args, **kwargs)
                if self.alive():
                    initialize()
            finally:
                with lock:
                    # READY (or a lazy/failed start) may never get a generation.
                    schedule_idle()

        def close(self):
            super().close()
            with lock:
                cancel_idle()
                if not active:
                    # /unload may be followed by Windows process termination,
                    # which does not execute run_server's finally block.
                    power._relax_sync()

        def generate(self, *args, **kwargs):
            nonlocal active
            with lock:
                cancel_idle()
                active += 1
            try:
                with lock:
                    initialize()
                    power._boost_sync()
                yield from super().generate(*args, **kwargs)
            finally:
                with lock:
                    active -= 1
                    schedule_idle()

    original_argv = sys.argv
    try:
        server.StrataEngine = AwakeEngine
        sys.argv = ["serve.server", *server_args]
        return server.main()
    finally:
        with lock:
            stopping = True
            cancel_idle()
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
