"""Loopback-only HTTP transport and one asynchronous scientific worker."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

from .model import CATALOG, ConfigurationError, examples, loads, parse_json, validate


class BusyError(RuntimeError):
    pass


class RunManager:
    def __init__(self, runner=None):
        if runner is None:
            from .engine import execute
            runner = execute
        self.runner = runner
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fiatlux')
        self.lock = threading.Lock()
        self.future = None
        self.run_id = None

    def start(self, config):
        validate(config)
        snapshot = deepcopy(config)
        with self.lock:
            if self.future is not None and not self.future.done():
                raise BusyError('Une simulation est déjà en cours')
            self.run_id = uuid.uuid4().hex
            self.future = self.executor.submit(self.runner, snapshot)
            return self.run_id

    def status(self, run_id):
        with self.lock:
            if run_id != self.run_id or self.future is None:
                raise ConfigurationError('Simulation inconnue ou remplacée')
            if not self.future.done():
                return dict(state='running', run_id=run_id)
            try:
                result = self.future.result()
            except Exception as error:
                return dict(state='error', run_id=run_id, error=str(error))
            return dict(state='complete', run_id=run_id, result=result.metadata())

    def image(self, run_id, **kwargs):
        with self.lock:
            if run_id != self.run_id or self.future is None or not self.future.done():
                raise ConfigurationError('Résultat non disponible')
            return self.future.result().image(**kwargs)

    def close(self):
        self.executor.shutdown(wait=True)


def make_server(port=8765, manager=None):
    manager = manager or RunManager()
    from .revisions import RevisionManager
    from .graph import GRAPH_CATALOG, graph_examples, compile_graph, check_connections, ports, upgrade
    canvas_manager = RevisionManager()
    static = Path(__file__).parent / 'static'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type='application/json; charset=utf-8'):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self):
            host = self.headers.get('Host')
            valid = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return host in valid and (origin is None or origin in {f'http://{h}' for h in valid})

        def do_GET(self):
            if not self.allowed():
                self.send(403, dict(error='Accès local uniquement'))
                return
            url = urlsplit(self.path)
            try:
                if url.path == '/api/canvas/catalog':
                    self.send(200, dict(components=GRAPH_CATALOG, examples=graph_examples(),
                                       ports={kind: ports(kind) for kind in GRAPH_CATALOG}))
                elif url.path == '/api/canvas/status':
                    self.send(200, canvas_manager.status(parse_qs(url.query).get('session', [''])[0]))
                elif url.path == '/api/canvas/image':
                    p = parse_qs(url.query)
                    self.send(200, canvas_manager.image(p.get('session', [''])[0], int(p.get('revision', ['-1'])[0]),
                              node=p.get('node', [''])[0], port=p.get('port', ['field'])[0],
                              quantity=p.get('quantity', ['intensity'])[0], channel=p.get('channel', ['sum'])[0],
                              opd_unit=p.get('opd_unit', ['nm'])[0], latent=json.loads(p.get('latent', ['{}'])[0])))
                elif url.path == '/api/catalog':
                    self.send(200, dict(components=CATALOG, examples=examples()))
                elif url.path == '/api/status':
                    params = parse_qs(url.query)
                    self.send(200, manager.status(params.get('run', [''])[0]))
                elif url.path == '/api/image':
                    params = parse_qs(url.query)
                    self.send(200, manager.image(params.get('run', [''])[0],
                              step=int(params.get('step', ['0'])[0]),
                              quantity=params.get('quantity', ['intensity'])[0],
                              channel=params.get('channel', ['sum'])[0],
                              opd_unit=params.get('opd_unit', ['nm'])[0]))
                elif url.path in ('/', '/index.html', '/linear', '/app.js', '/style.css', '/canvas.js', '/canvas.css'):
                    name = 'linear.html' if url.path == '/linear' else ('index.html' if url.path == '/' else url.path[1:])
                    mime = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[name.rsplit('.', 1)[1]]
                    self.send(200, (static / name).read_bytes(), mime+'; charset=utf-8')
                else:
                    self.send(404, dict(error='Ressource inconnue'))
            except (ValueError, TypeError, KeyError) as error:
                self.send(400, dict(error=str(error)))
            except Exception as error:
                self.send(500, dict(error=str(error)))

        def do_POST(self):
            if not self.allowed():
                self.send(403, dict(error='Accès local uniquement'))
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ConfigurationError('Configuration vide ou supérieure à 1 Mo')
                text = self.rfile.read(length).decode('utf-8')
                if self.path.startswith('/api/canvas/'):
                    data = parse_json(text)
                    if self.path == '/api/canvas/submit':
                        self.send(202, canvas_manager.submit(data['session'], data['revision'], data['configuration']))
                    elif self.path == '/api/canvas/validate':
                        config = upgrade(data) if data.get('version') == 1 else data
                        compile_graph(config)
                        self.send(200, dict(valid=True, configuration=config))
                    elif self.path == '/api/canvas/connect':
                        check_connections(data)
                        self.send(200, dict(valid=True))
                    else:
                        self.send(404, dict(error='Ressource inconnue'))
                    return
                config = loads(text)
                if self.path == '/api/run':
                    self.send(202, dict(run_id=manager.start(config)))
                elif self.path == '/api/validate':
                    self.send(200, dict(valid=True))
                else:
                    self.send(404, dict(error='Ressource inconnue'))
            except BusyError as error:
                self.send(409, dict(error=str(error)))
            except (ValueError, TypeError, UnicodeError, KeyError) as error:
                self.send(400, dict(error=str(error)))
            except Exception as error:
                self.send(500, dict(error=str(error)))

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.manager = manager
    server.canvas_manager = canvas_manager
    return server
