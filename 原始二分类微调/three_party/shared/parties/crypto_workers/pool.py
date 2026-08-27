"""
CryptoWorkerPool — long-lived fork pool for CPU crypto workers.

Replaces both the old ``ChunkedPool`` (in ``src/core/chunk_pool.py``) and the
fork pattern hidden inside ``add_mask_to_ct_batch_par`` /
``decrypt_only_batch_par``. The pool is:

  * Initialized **once** per process lifetime (avoids the ~50ms × N_workers
    startup cost that per-batch pools paid).
  * Forks from the ``HeterogeneousProtocol`` driver (which already holds the
    GPU model + pk_M).
  * Independent of CUDA: workers never touch torch tensors or
    ``torch.cuda``; only ``numpy`` / ``seal``.

Worker selection
----------------
The pool is **typed** — instantiated for one specific worker class (e.g.
``CryptoUWorker``). Use three separate pools for U/M/S, each with its own
fork:

  * ``CryptoWorkerPool(CryptoUWorker, n_workers=8, init_kwargs={"bfv_pk_pem": ...})``
  * ``CryptoWorkerPool(CryptoMWorker, n_workers=8, init_kwargs={"bfv_sk_pem": ..., ...})``
  * ``CryptoWorkerPool(CryptoSWorker, n_workers=1, init_kwargs={"prg_seed": ...})``
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Dict, Type

from . import base

logger = logging.getLogger(__name__)


class CryptoWorkerPool:
    """Long-lived multiprocessing pool for one type of crypto worker."""

    def __init__(
        self,
        worker_cls: Type,
        n_workers: int,
        init_kwargs: Dict[str, Any],
    ) -> None:
        """Create the pool and fork workers.

        Args:
            worker_cls: subclass of crypto worker protocol — must expose
                ``init_state(**kwargs) -> dict`` and
                ``handle_request(state: dict, payload: dict) -> dict``.
            n_workers: number of fork processes.
            init_kwargs: passed to ``worker_cls.init_state`` exactly once per
                worker.
        """
        self.worker_cls = worker_cls
        self.worker_cls_path = f"{worker_cls.__module__}.{worker_cls.__name__}"
        self.n_workers = n_workers

        # Workers use ``spawn`` so that the forked CUDA caching allocator
        # doesn't propagate. With ``fork``, every worker would inherit a
        # ~6 GB snapshot of the parent's CUDA allocator, leading to OOM
        # when 8+ workers run concurrently.
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=n_workers,
            initializer=base.init_pool_worker,
            initargs=(self.worker_cls_path, init_kwargs),
        )
        logger.info(
            "[CryptoWorkerPool] %s ready: n_workers=%d", self.worker_cls_path, n_workers
        )

    def submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a single request, block until result is ready.

        ``Pool.map`` is used internally by ``apply_async``; we use ``map``
        here too — the pool worker count is small enough that map's
        auto-chunking keeps things simple.
        """
        # We use ``apply_async`` so callers get a single AsyncResult back,
        # then immediately ``.get()`` to block. This is the recommended
        # pattern when the result is needed synchronously by the caller.
        async_result = self._pool.apply_async(
            base.run_worker_request, (self.worker_cls_path, payload)
        )
        return async_result.get()

    def submit_async(self, payload: Dict[str, Any]) -> "mp.pool.AsyncResult":
        """Submit a request asynchronously (don't block).

        Returns the ``AsyncResult`` so the caller can ``.get()`` later or
        compose a pipeline.
        """
        return self._pool.apply_async(
            base.run_worker_request, (self.worker_cls_path, payload)
        )

    def close(self) -> None:
        """Tear down the pool. Idempotent."""
        try:
            self._pool.close()
            self._pool.join()
        except Exception as e:
            logger.warning("[CryptoWorkerPool] close raised: %s", e)

    def __enter__(self) -> "CryptoWorkerPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
