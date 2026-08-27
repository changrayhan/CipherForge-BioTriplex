"""
Wire types and the StepProfiler shared by the heterogeneous runtime and
the legacy three-process IPC stub.

Kept in its own module so that :mod:`heterogeneous_protocol`,
:mod:`legacy_ipc_stub`, and any test scripts can share them without pulling
in the worker-process machinery.
"""

from __future__ import annotations

import logging
import os
import resource
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

__all__ = ["StepResult", "StepProfiler"]


@dataclass
class StepResult:
    """Result of one training step."""
    step: int
    loss: float
    gpu_mem_mb: float
    step_time_ms: float
    attack_dumps: dict = field(default_factory=dict)
    n_chunks: int = 1  # 1 for non-chunked flat step
    dp_audit: dict = field(default_factory=dict)  # dχ-privacy per-step audit
    # Optional true cross-entropy monitor computed by PartyS at the gold
    # answer positions (for training telemetry only; the actual update uses
    # the encrypted gradient ``a_t - V_gold``).
    loss_ce: Optional[float] = None


class StepProfiler:
    """Per-step phase timer + rolling-window JSONL writer."""

    def __init__(self, log_dir: Optional[str] = None, max_in_memory: int = 100):
        self.phases: Dict[str, float] = {}
        self.order: List[str] = []
        self.t_phase_start: Dict[str, float] = {}
        self.recent_steps: Deque[Dict[str, Any]] = deque(maxlen=max_in_memory)
        self.is_chunked: bool = False
        self._chunk_u_times: List[float] = []
        self._chunk_m_times: List[float] = []
        self.cumulative_steps: int = 0
        self.log_path: Optional[Path] = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.log_path = Path(log_dir) / "step_profiles.jsonl"

    def begin_phase(self, name: str) -> None:
        self.t_phase_start[name] = time.time()

    def end_phase(self, name: str) -> None:
        if name not in self.t_phase_start:
            return
        dt_ms = (time.time() - self.t_phase_start.pop(name)) * 1000
        if name not in self.phases:
            self.order.append(name)
        self.phases[name] = self.phases.get(name, 0.0) + dt_ms

    def record_chunk(self, party: str, dt_ms: float) -> None:
        if party == "U":
            self._chunk_u_times.append(dt_ms)
        elif party == "M":
            self._chunk_m_times.append(dt_ms)

    def end_step(
        self,
        step: int,
        n_tokens: int,
        n_chunks: int,
        step_time_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "step": step,
            "n_tokens": n_tokens,
            "n_chunks": n_chunks,
            "step_time_ms": step_time_ms,
            "phase_ms": dict(self.phases),
            "phase_order": list(self.order),
            "chunk_u_times_ms": list(self._chunk_u_times),
            "chunk_m_times_ms": list(self._chunk_m_times),
            "rss_mb": _rss_mb(),
            "ts": time.time(),
        }
        if extra:
            record.update(extra)
        self.recent_steps.append(record)
        self.cumulative_steps += 1

        if self.log_path is not None:
            try:
                with open(self.log_path, "a") as f:
                    import json
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass
        self.phases.clear()
        self.order.clear()
        self._chunk_u_times.clear()
        self._chunk_m_times.clear()


def _rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in KB on Linux.
        return usage / 1024.0
    except Exception:
        return 0.0
