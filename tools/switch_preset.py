"""Start one local Strata preset, draining the previous preset on the same port.

The local, untracked strata-models.json catalog names allowed configs. Only this
workspace's serve_gpu_wake.py processes with those configs are managed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[1]


def resolved(base, value):
    return (Path(base) / value).resolve()


def argument(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def identify(args, cwd, configs, launcher):
    """Reject similar filenames and configs from other workspaces."""
    if not any(Path(a).name == launcher.name and resolved(cwd, a) == launcher for a in args[1:]):
        return None
    path = argument(args, '--config')
    if not path:
        return None
    path = resolved(cwd, path)
    if path not in configs:
        return None
    return configs[path], int(argument(args, '--port', '8080'))


@dataclass
class Running:
    name: str
    process: psutil.Process
    argv: list[str]
    cwd: str
    port: int


class Presets:
    def __init__(self, catalog, port=None):
        self.catalog_path = Path(catalog).resolve()
        self.base = self.catalog_path.parent
        self.catalog = json.loads(self.catalog_path.read_text(encoding='utf-8-sig'))
        self.port = int(self.catalog.get('port', 8080)) if port is None else port
        if not 1 <= self.port <= 65535:
            raise ValueError('Port must be in 1..65535')
        self.launcher = ROOT / 'tools/serve_gpu_wake.py'
        self.paths = {name: resolved(self.base, row['config']) for name, row in self.catalog['presets'].items()}
        self.configs = {name: json.loads(path.read_text(encoding='utf-8-sig')) for name, path in self.paths.items()}
        keys = {cfg.get('api_key') for cfg in self.configs.values()}
        if len(keys) != 1 or not all(isinstance(k, str) and k.strip() for k in keys):
            raise ValueError('All presets must have the same nonempty API key; no server was changed')
        self.key = next(iter(keys))
        self.logs = self.base / 'logs/presets'
        self.logs.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        # One transition per catalog, including launches onto a different port.
        path = self.catalog_path.with_suffix('.lock')
        with path.open('a+b') as f:
            f.seek(0, 2)
            if not f.tell():
                f.write(b'0'); f.flush()
            deadline = time.monotonic() + 900
            announced = False
            while True:
                try:
                    f.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Another preset switch is still running') from None
                    if not announced:
                        print('Waiting for the other preset switch to finish...', flush=True)
                        announced = True
                    time.sleep(0.5)
            try:
                yield
            finally:
                f.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def running(self):
        allowed = {path: name for name, path in self.paths.items()}
        found = []
        for p in psutil.process_iter(['name', 'cmdline']):
            try:
                if not (p.info['name'] or '').lower().startswith('python'):
                    continue
                args = p.info['cmdline'] or []
                cwd = p.cwd()
                identity = identify(args, cwd, allowed, self.launcher)
                if identity:
                    name, port = identity
                    found.append(Running(name, p, args, cwd, port))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return found

    def check_port(self, running):
        allowed = {r.process.pid for r in running if r.port == self.port}
        for c in psutil.net_connections(kind='tcp'):
            if c.status == psutil.CONN_LISTEN and c.laddr.port == self.port and c.pid not in allowed:
                raise RuntimeError(f'Port {self.port} belongs to another process (PID {c.pid}); it was not stopped')

    def api(self, port, endpoint, body=None):
        req = urllib.request.Request(f'http://127.0.0.1:{port}/{endpoint}',
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        # Ignore proxy environment variables for local model controls.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=120) as r:
            return json.load(r)

    def validate(self, name):
        cfg = self.configs[name]
        cwd = resolved(self.base, cfg.get('cwd', '.'))
        files = [resolved(cwd, cfg['exe']), resolved(cwd, cfg['tokenizer']) / 'vocab.json']
        args = cfg.get('args', [])
        pack = argument(args, '--pack')
        if pack:
            files += [resolved(cwd, pack) / f for f in ('dense.bin', 'index.txt', 'native_experts.txt')]
        native = argument(args, '--native')
        if native:
            files.append(resolved(cwd, native))
        vision = cfg.get('vision', {})
        files += [resolved(cwd, vision[k]) for k in ('exe', 'model', 'mmproj') if k in vision]
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise RuntimeError('Missing preset files: ' + ', '.join(missing))

    @staticmethod
    def terminate_owned(r):
        try:
            children = r.process.children(recursive=True)
            r.process.terminate()
            r.process.wait(timeout=30)
            for child in reversed(children):
                try:
                    if child.is_running():
                        child.terminate()
                        child.wait(timeout=30)
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass

    def drain(self, r):
        print(f'Unloading {r.name}; waiting for active requests to finish...', flush=True)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                state = self.api(r.port, 'status')
                if not state.get('busy') and not state.get('queued'):
                    result = self.api(r.port, 'unload', {})
                    if result.get('status') in ('unloaded', 'not loaded'):
                        self.terminate_owned(r)
                        return
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    raise RuntimeError(f'Cannot unload the current server (HTTP {e.code}); it was not stopped') from None
            except (urllib.error.URLError, TimeoutError):
                if not r.process.is_running():
                    raise RuntimeError('The current server exited while being unloaded') from None
            time.sleep(1)
        raise RuntimeError('The current server stayed busy or unavailable; no new model was started')

    def await_ready(self, r):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if not r.process.is_running() or r.process.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError(f'{r.name} exited while loading; see {self.logs}')
            try:
                h = self.api(r.port, 'health')
                if h.get('status') == 'ok' and h.get('loaded'):
                    expected = argument(self.configs[r.name].get('args', []), '--max-context')
                    if expected and h.get('max_context') != int(expected):
                        raise RuntimeError('Loaded context does not match the requested preset')
                    return h
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        raise RuntimeError(f'{r.name} did not become ready; see {self.logs}')

    def launch(self, name, previous=None):
        cfg = self.configs[name]
        cwd = str(resolved(self.base, cfg.get('cwd', '.')))
        port = previous.port if previous else self.port
        argv = previous.argv if previous else [sys.executable, '-B', str(self.launcher),
            '--power-module-dir', str(resolved(self.base, self.catalog['power_module_dir'])),
            '--wake-device', self.catalog['wake_device'], '--engine', 'strata', '--config', str(self.paths[name]),
            '--host', self.catalog.get('host', '0.0.0.0'), '--port', str(port)]
        if previous:
            cwd = previous.cwd
        env = dict(os.environ, PYTHONUTF8='1')
        env.pop('GGML_VK_VISIBLE_DEVICES', None)
        options = {}
        if os.name == 'nt':
            options['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
            options['startupinfo'] = si
        else:
            options['start_new_session'] = True
        print(f'Loading {name} on port {port}...', flush=True)
        with (self.logs / f'{name}.stdout.log').open('ab') as out, (self.logs / f'{name}.stderr.log').open('ab') as err:
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, **options)
        r = Running(name, psutil.Process(p.pid), list(argv), cwd, port)
        try:
            self.await_ready(r)
        except BaseException:
            self.terminate_owned(r)
            raise
        return r

    def record(self, r):
        state = {'preset': r.name, 'pid': r.process.pid, 'port': r.port,
                 'config': str(self.paths[r.name]), 'updated_at': time.time()} if r else {'preset': None, 'updated_at': time.time()}
        path = self.base / 'strata-active.json'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        tmp.replace(path)

    def switch(self, name):
        with self.lock():
            running = self.running()
            if len(running) > 1:
                raise RuntimeError('Several managed servers are running; no processes were stopped')
            if name == 'status':
                print(json.dumps([{'preset': r.name, 'pid': r.process.pid, 'port': r.port} for r in running]))
                return
            if name != 'stop':
                self.validate(name)   # finish checks before unloading the current model
                self.check_port(running)
            old = running[0] if running else None
            if name == 'stop':
                if old:
                    self.drain(old)
                self.record(None)
                print('Strata stopped.', flush=True)
                return
            if old and old.name == name and old.port == self.port:
                if not self.api(old.port, 'health').get('loaded'):
                    self.api(old.port, 'load', {})
                self.await_ready(old)
                self.record(old)
                print(f'{name} is already running: http://127.0.0.1:{old.port}', flush=True)
                return
            if old:
                self.drain(old)
            try:
                new = self.launch(name)
            except BaseException:
                if old:
                    print(f'New preset failed; restoring {old.name}...', flush=True)
                    self.record(self.launch(old.name, previous=old))
                raise
            self.record(new)
            print(f'READY {name}: http://127.0.0.1:{new.port} (same API key)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('preset', help='catalog preset name, status, or stop')
    ap.add_argument('--catalog', type=Path, default=ROOT / 'strata-models.json')
    ap.add_argument('--port', type=int)
    args = ap.parse_args()
    presets = Presets(args.catalog, args.port)
    if args.preset not in {*presets.paths, 'stop', 'status'}:
        ap.error('unknown preset')
    try:
        presets.switch(args.preset)
    except (RuntimeError, ValueError, OSError, psutil.Error) as e:
        print(f'Strata: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
