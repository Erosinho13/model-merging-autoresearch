"""Research runtime budget that excludes time waiting for human review."""
from contextlib import contextmanager
import math
from time import monotonic

import anyio


class PausableTimer:
    def __init__(self, limit, clock=monotonic):
        self.limit = limit
        self.clock = clock
        self.started_at = None
        self.finished_at = None
        self.paused_at = None
        self.human_wait_seconds = 0.0
        self.scope = None
        self.expired = False

    def start(self):
        self.started_at = self.clock()

    @property
    def elapsed(self):
        if self.started_at is None:
            return 0.0
        now = self.finished_at if self.finished_at is not None else self.clock()
        waiting = now - self.paused_at if self.paused_at is not None else 0.0
        return max(0.0, now - self.started_at - self.human_wait_seconds - waiting)

    @property
    def remaining(self):
        return max(0.0, self.limit - self.elapsed)

    def attach(self, scope):
        self.scope = scope
        self._refresh_deadline()

    def _refresh_deadline(self):
        if self.scope is not None:
            self.scope.deadline = (math.inf if self.paused_at is not None
                                   else anyio.current_time() + self.remaining)

    @contextmanager
    def pause_for_human(self):
        if self.paused_at is not None:
            raise RuntimeError("Human review is already pending")
        self.paused_at = self.clock()
        self._refresh_deadline()
        try:
            yield
        finally:
            self.human_wait_seconds += self.clock() - self.paused_at
            self.paused_at = None
            self._refresh_deadline()

    def finish(self):
        self.finished_at = self.clock()
        self.scope = None
