"""Clock locks must cover work and expire even when startup gets no requests."""
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from serve_gpu_wake import run_server
from serve import server


class WakeLifecycle(unittest.TestCase):
    def run_case(self, scenario=None, *, fail_probe=False, fail_generate=False,
                 fail_load=False, lazy=False, after_shutdown=None):
        events, timers = [], []

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
            def __init__(self, lazy=False):
                events.append("lazy" if lazy else "ready")
                self.running = not lazy
                if fail_load:
                    raise ValueError("failed loading")
            def alive(self): return self.running
            def close(self):
                events.append("closed")
                self.running = False
            def generate(self):
                events.append("generate")
                if fail_generate:
                    raise ValueError("failed generation")
                yield 123

        class Timer:
            def __init__(self, seconds, callback):
                self.seconds, self.callback = seconds, callback
                self.started = self.cancelled = False
                timers.append(self)
            def start(self): self.started = True
            def cancel(self): self.cancelled = True
            def fire(self):
                # Also simulate a cancelled callback that was already running
                # and waiting for the wrapper's lock when cancel() was called.
                self.callback()

        def main():
            e = server.StrataEngine(lazy=lazy)
            self.assertEqual(events[:3], ["nvml", "boost", "lazy" if lazy else "ready"])
            if not lazy:
                self.assertLess(events.index("ready"), events.index("probe"))
            if scenario:
                scenario(e, events, timers)
            else:
                self.assertEqual(list(e.generate()), [123])
            return 0

        module = SimpleNamespace(GpuPowerManager=Power)
        with mock.patch.dict(sys.modules, gpu_monitor=module), mock.patch.dict(os.environ), \
             mock.patch.object(sys, "path", list(sys.path)), mock.patch.object(server, "StrataEngine", Engine), \
             mock.patch.object(server, "main", main), mock.patch("serve_gpu_wake.threading.Timer", Timer):
            old_argv = sys.argv
            if fail_probe or fail_generate or fail_load:
                with self.assertRaises((RuntimeError, ValueError)):
                    run_server([], ".", "GPU-wake")
            else:
                self.assertEqual(run_server([], ".", "GPU-wake"), 0)
            self.assertIs(server.StrataEngine, Engine)
            self.assertIs(sys.argv, old_argv)
        self.assertEqual(events[-1], "shutdown")
        if after_shutdown:
            after_shutdown(events, timers)
        # No callback may touch NVML after shutdown, even one already in flight.
        after = list(events)
        for timer in timers:
            self.assertTrue(timer.started)
            self.assertTrue(timer.daemon)
            self.assertEqual(timer.seconds, 120.0)
            timer.fire()
        self.assertEqual(events, after)

    def test_ready_before_probe_and_shutdown(self):
        self.run_case()

    def test_failed_probe_releases_clocks(self):
        self.run_case(fail_probe=True)

    def test_failed_generation_releases_clocks(self):
        self.run_case(fail_generate=True)

    def test_failed_load_releases_clocks(self):
        self.run_case(fail_load=True)

    def test_startup_without_requests_relaxes(self):
        def scenario(e, events, timers):
            self.assertEqual(len(timers), 1)
            timers[0].fire()
            self.assertEqual(events[-1], "relax")
        self.run_case(scenario)

    def test_lazy_start_without_requests_relaxes(self):
        def scenario(e, events, timers):
            self.assertNotIn("probe", events)
            self.assertEqual(len(timers), 1)
            timers[0].fire()
            self.assertEqual(events[-1], "relax")
        self.run_case(scenario, lazy=True)

    def test_request_cancels_idle_and_cancelled_stream_rearms_it(self):
        def scenario(e, events, timers):
            startup = timers[-1]
            g = e.generate()
            self.assertEqual(next(g), 123)
            self.assertTrue(startup.cancelled)
            startup.fire()
            self.assertNotIn("relax", events)
            g.close()
            self.assertEqual(len(timers), 2)
            # The old callback also cannot shorten the new request's idle period.
            startup.fire()
            self.assertNotIn("relax", events)
            timers[-1].fire()
            self.assertEqual(events[-1], "relax")
        self.run_case(scenario)

    def test_overlapping_requests_relax_only_after_both_finish(self):
        def scenario(e, events, timers):
            first, second = e.generate(), e.generate()
            next(first)
            next(second)
            first.close()
            self.assertEqual(len(timers), 1)
            timers[0].fire()
            self.assertNotIn("relax", events)
            second.close()
            self.assertEqual(len(timers), 2)
            timers[-1].fire()
            self.assertEqual(events[-1], "relax")
        self.run_case(scenario)

    def test_unload_releases_and_reload_starts_a_new_idle_period(self):
        def scenario(e, events, timers):
            startup = timers[-1]
            e.close()
            self.assertEqual(events[-2:], ["closed", "relax"])
            self.assertTrue(startup.cancelled)
            before = list(events)
            startup.fire()
            self.assertEqual(events, before)
            e.__init__()
            self.assertEqual(events[-2:], ["boost", "ready"])
            self.assertEqual(events.count("probe"), 1)
            startup.fire()
            self.assertEqual(events.count("relax"), 1)
            timers[-1].fire()
            self.assertEqual(events.count("relax"), 2)
        self.run_case(scenario)

    def test_stream_finishing_after_shutdown_does_not_arm_a_timer(self):
        streams = []

        def scenario(e, events, timers):
            g = e.generate()
            next(g)
            streams.append(g)

        def after_shutdown(events, timers):
            before = list(events)
            count = len(timers)
            streams[0].close()
            self.assertEqual(events, before)
            self.assertEqual(len(timers), count)

        self.run_case(scenario, after_shutdown=after_shutdown)


if __name__ == "__main__":
    unittest.main()
