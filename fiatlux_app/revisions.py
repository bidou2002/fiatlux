"""Single worker, per-session latest-only queue and guarded publication."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import threading

from .graph import compile_graph
from .model import ConfigurationError


@dataclass
class Session:
    revision: int = -1
    published: int = -1
    state: str = 'idle'
    error: str | None = None
    result: object = None
    affected: list | None = None


class RevisionManager:
    def __init__(self, engine_factory=None):
        if engine_factory is None:
            from .incremental import IncrementalEngine
            engine_factory = IncrementalEngine
        self.factory = engine_factory
        self.engines, self.sessions, self.pending = {}, {}, {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fiatlux-canvas')
        self.worker = None
        self.closed = False

    def submit(self, session, revision, config):
        if not isinstance(session, str) or not 1 <= len(session) <= 80 or type(revision) is not int or revision < 0:
            raise ConfigurationError('Session et révision entière positive requises')
        with self.lock:
            if session not in self.sessions and len(self.sessions) >= 8:
                raise ConfigurationError('Huit sessions maximum ; redémarrer le serveur pour les libérer')
            state = self.sessions.setdefault(session, Session())
            if revision <= state.revision:
                return dict(accepted=False, revision=state.revision)
            state.revision, state.state, state.error = revision, 'pending', None
            self.pending.pop(session, None)
        snapshot = deepcopy(config)
        try:
            compile_graph(snapshot)
        except (ValueError, TypeError, KeyError) as error:
            with self.lock:
                if state.revision == revision:
                    state.state, state.error = 'error', str(error)
            return dict(accepted=True, revision=revision, valid=False, error=str(error))
        with self.lock:
            if state.revision != revision:
                return dict(accepted=False, revision=state.revision)
            if session not in self.engines:
                self.engines[session] = self.factory()
            state.affected = self.engines[session].affected(snapshot)
            self.pending[session] = (revision, snapshot)
            if self.worker is None or self.worker.done():
                self.worker = self.executor.submit(self._drain)
        return dict(accepted=True, revision=revision, valid=True)

    def _drain(self):
        while True:
            with self.lock:
                if self.closed or not self.pending:
                    self.worker = None
                    return
                session = next(iter(self.pending))
                revision, config = self.pending.pop(session)
                state = self.sessions[session]
                if state.revision != revision:
                    continue
                state.state = 'running'
            try:
                result = self.engines[session].execute(config)
                error = None
            except Exception as failure:
                result, error = None, str(failure)
            with self.lock:
                # Outdated work may warm valid caches, but must never publish.
                if state.revision == revision:
                    if error is None:
                        state.result, state.published, state.state, state.error = result, revision, 'current', None
                        state.affected = []
                    else:
                        state.state, state.error = 'error', error

    def status(self, session):
        with self.lock:
            if session not in self.sessions:
                raise ConfigurationError('Session inconnue')
            state = self.sessions[session]
            return dict(revision=state.revision, published_revision=state.published, state=state.state,
                        error=state.error, affected=state.affected or [],
                        result=None if state.result is None else state.result.metadata())

    def image(self, session, revision, **kwargs):
        with self.lock:
            state = self.sessions.get(session)
            if state is None or state.result is None or state.published != revision:
                raise ConfigurationError('Révision de résultat indisponible')
            result = state.result
        return dict(result.image(**kwargs), revision=revision)

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
        self.executor.shutdown(wait=True)
