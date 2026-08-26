"""
Base crypto-worker primitives.

Each subclass of :class:`BaseCryptoWorker` implements two methods:

  * :meth:`init_state` — called once per worker process; loads SEAL context
    and any worker-specific secrets.
  * :meth:`handle_request` — called for every request; returns a Python
    dict (the response payload).

Workers run inside a ``multiprocessing.Pool`` (forked from the
``HeterogeneousProtocol`` driver). They communicate with the driver over
``multiprocessing.Queue`` pairs — but in practice, since ``Pool.apply_async``
hides the queue plumbing, the driver-side API is just
``CryptoWorkerPool.submit(payload)``.

Why not multiprocessing.Manager / shared memory? Because the per-request
payload (a list of s3pir_responses or a list of ciphertexts) is already
pickled by the pool — going through a manager would just add overhead.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  Worker-side state cache (one entry per worker process)
# -----------------------------------------------------------------------------
# This module-level dict caches per-worker state (SEAL context, etc.) so that
# repeated calls don't re-init. Keyed by worker PID.
_WORKER_STATE: Dict[str, Any] = {}


def _get_worker_state() -> Dict[str, Any]:
    pid = mp.current_process().pid
    state = _WORKER_STATE.get(pid)
    if state is None:
        state = {}
        _WORKER_STATE[pid] = state
    return state


# -----------------------------------------------------------------------------
#  Pool initializer
# -----------------------------------------------------------------------------
def init_pool_worker(
    worker_cls_path: str,
    init_kwargs: Dict[str, Any],
) -> None:
    """Initializer called in each forked worker process.

    Args:
        worker_cls_path: dotted path of the worker class (e.g.
            ``"src.parties.crypto_workers.crypto_u.CryptoUWorker"``).
            We import lazily inside the worker to avoid circular imports.
        init_kwargs: kwargs forwarded to ``worker_cls.init_state``.
    """
    import importlib

    module_path, _, attr = worker_cls_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    state = cls.init_state(**init_kwargs)
    pid = mp.current_process().pid
    _WORKER_STATE[pid] = state
    logger.info("[%s pid=%d] initialized", cls.__name__, pid)


# -----------------------------------------------------------------------------
#  Pool task function
# -----------------------------------------------------------------------------
def run_worker_request(
    worker_cls_path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Task function submitted to ``Pool.apply_async``.

    Args:
        worker_cls_path: same dotted path as in ``init_pool_worker``.
        payload: opaque dict consumed by ``handle_request``.
    """
    import importlib

    module_path, _, attr = worker_cls_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    state = _get_worker_state()
    return cls.handle_request(state, payload)
