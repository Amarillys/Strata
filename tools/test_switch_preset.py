"""Ownership and failure behavior of the local preset switcher; no live processes."""
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import switch_preset as S


class SwitchPresetTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[1]
        self.launcher = self.base / 'tools/serve_gpu_wake.py'
        self.cfg = self.base / 'strata-test.json'
        self.allowed = {self.cfg: 'test'}

    def test_only_exact_workspace_launcher_and_config_are_owned(self):
        args = ['python', '-B', 'tools/serve_gpu_wake.py', '--config', 'strata-test.json', '--port', '8080']
        self.assertEqual(S.identify(args, self.base, self.allowed, self.launcher), ('test', 8080))
        for cwd, argv in [(self.base.parent, args),
                          (self.base, [*args[:2], 'tools/serve_gpu_wake.py.bak', *args[3:]]),
                          (self.base, ['python', str(self.launcher), '--config', str(self.base.parent / self.cfg.name)])]:
            with self.subTest(cwd=cwd, argv=argv):
                self.assertIsNone(S.identify(argv, cwd, self.allowed, self.launcher))

    def manager(self, old):
        p = S.Presets.__new__(S.Presets)
        p.port = 8080
        p.lock = Mock(return_value=nullcontext())
        p.running = Mock(return_value=[old] if old else [])
        for name in ('validate', 'check_port', 'api', 'await_ready', 'record', 'drain', 'launch'):
            setattr(p, name, Mock())
        return p

    def test_unrelated_listener_is_never_stopped(self):
        old = SimpleNamespace(name='v2', port=8080)
        p = self.manager(old)
        p.check_port.side_effect = RuntimeError('unrelated listener')
        with self.assertRaisesRegex(RuntimeError, 'unrelated'):
            p.switch('v4')
        p.drain.assert_not_called()
        p.launch.assert_not_called()

    def test_missing_new_files_leave_the_current_model_running(self):
        p = self.manager(SimpleNamespace(name='v2', port=8080))
        p.validate.side_effect = RuntimeError('missing pack')
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            p.switch('v4')
        p.drain.assert_not_called()
        p.launch.assert_not_called()

    def test_busy_current_model_does_not_overlap_a_new_load(self):
        p = self.manager(SimpleNamespace(name='v2', port=8080))
        p.drain.side_effect = RuntimeError('still busy')
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            p.switch('v4')
        p.launch.assert_not_called()

    def test_start_failure_restores_the_previous_model(self):
        old = SimpleNamespace(name='v2', port=8080)
        p = self.manager(old)
        p.launch.side_effect = [RuntimeError('new load failed'), old]
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'new load failed'):
            p.switch('v4')
        self.assertEqual(p.launch.call_args_list[0].args, ('v4',))
        p.launch.assert_called_with('v2', previous=old)
        p.record.assert_called_once_with(old)

    def test_clicking_an_already_loaded_model_does_not_restart_or_load_it(self):
        old = SimpleNamespace(name='v4', port=8080)
        p = self.manager(old)
        p.api.return_value = {'loaded': True}
        with redirect_stdout(io.StringIO()):
            p.switch('v4')
        p.api.assert_called_once_with(8080, 'health')
        p.launch.assert_not_called()
        p.drain.assert_not_called()
        p.record.assert_called_once_with(old)


if __name__ == '__main__':
    unittest.main()
