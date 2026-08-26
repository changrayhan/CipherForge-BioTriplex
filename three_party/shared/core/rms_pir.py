"""RMS-PIR — Ren-Mughees-Sun 2024 (CCS'24) two-server stateful PIR,
adapted to the CipherForge three-party pipeline as a backup PIR mode.

Structure (paper §3, "two-server scheme"):

  Offline phase (hint construction)
    The database of N rows is divided into n equal partitions of size p
    (n·p >= N; the last partition may be padded with zero rows).  Each hint
    j selects an exactly-half subset of the partitions (by ranking PRF
    values), picks one row per selected partition via a PRF offset, and adds
    one extra index e_j from an unselected partition.  The hint's parity is
    the sum (over the encrypted rows) of the whole subset.  In this codebase
    U derives the subsets from its private seed and asks S to compute the
    parities (S is a parity oracle; it sees the subsets, see docs caveats).

  Online phase
    For target y: take a hint whose subset contains y, remove y to form the
    real query subset, and build a dummy subset covering the remaining
    partitions with one random row each.  Send both subsets in random order;
    S returns both parities.  U recovers Enc(-V_y) = hint_parity - real_parity
    (ciphertext subtraction), adds the PRG mask r_t, and sends the result to
    M — exactly the same wire format as the block-PIR path.

  Hint replenishment
    The consumed hint is replaced by a fresh hint containing y: S computes
    the parities of both halves of the next hint ID; U picks the half that
    does not select y's partition and adds y as the extra index
    (new parity = picked-half parity + recovered Enc(-V_y)).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RMSHintParams:
    """Partition parameters for RMS-PIR over a database of n_entries rows."""

    def __init__(self, n_entries: int, partition_size: int, lam: int):
        self.n_entries = int(n_entries)
        self.p = int(partition_size)
        self.n = (self.n_entries + self.p - 1) // self.p
        if self.n % 2 != 0:
            # Paper assumes an even number of partitions for the half split.
            self.n += 1
        self.lam = int(lam)
        self.M = self.lam * self.n          # initial hint pool size
        self.padded_entries = self.n * self.p

    def to_dict(self) -> Dict[str, int]:
        return {
            "n_entries": self.n_entries,
            "partition_size": self.p,
            "n_partitions": self.n,
            "lam": self.lam,
            "M": self.M,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "RMSHintParams":
        return cls(d["n_entries"], d["partition_size"], d["lam"])


def _prf32(seed: bytes, label: bytes, *parts: int) -> int:
    """32-bit pseudorandom value PRF(seed, label, parts...)."""
    h = hashlib.sha256()
    h.update(seed)
    h.update(label)
    for x in parts:
        h.update(struct.pack("!Q", int(x)))
    return int.from_bytes(h.digest()[:8], "big") & 0xFFFFFFFF


def hint_half_rows(
    seed: bytes,
    params: RMSHintParams,
    j: int,
) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
    """Derive the rank-selected half of hint ``j``.

    Returns ``(rows_a, rows_b, extra, extra_partition)`` where
    ``rows_a`` maps partition -> row for the rank-selected half and
    ``rows_b`` maps partition -> row for the complement half.  ``extra`` is
    the extra index (a random row of a random unselected partition) and
    ``extra_partition`` is its partition id.
    """
    n = params.n
    p = params.p
    vals = sorted(
        (_prf32(seed, b"select", j, k), k) for k in range(n)
    )
    half = n // 2
    selected_ks = [k for _, k in vals[:half]]
    selected_set = set(selected_ks)
    rows_a: Dict[int, int] = {}
    rows_b: Dict[int, int] = {}
    for k in range(n):
        off = _prf32(seed, b"offset", j, k) % p
        row = min(k * p + off, params.padded_entries - 1)
        if k in selected_set:
            rows_a[k] = row
        else:
            rows_b[k] = row
    unselected = [k for k in range(n) if k not in selected_set]
    ek = unselected[_prf32(seed, b"extra_part", j) % len(unselected)]
    eoff = _prf32(seed, b"extra_off", j) % p
    extra = min(ek * p + eoff, params.padded_entries - 1)
    return rows_a, rows_b, extra, ek


class RMSHintStore:
    """U-side hint bookkeeping: disk-backed parity ciphertexts + subsets.

    Hints are indexed by hint ID ``j``.  Each query for label ``y`` pops a
    hint from ``label_hints[y]`` (precomputed at construction); the
    replenished hint (which contains ``y`` by construction) is appended back
    to the same list, so each label always has a non-empty supply.
    """

    def __init__(self, seed: bytes, params: RMSHintParams, cache_dir: str):
        self.seed = seed
        self.params = params
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hints: Dict[int, Dict] = {}          # j -> hint record
        self.label_hints: Dict[int, List[int]] = {}  # y -> list of j
        self.next_j = params.M
        self._known_labels: List[int] = []
        self.debug_consumed: Dict[int, int] = {}
        self.debug_replenished: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  Persistence helpers
    # ------------------------------------------------------------------ #
    def _parity_path(self, j: int) -> Path:
        return self.cache_dir / f"hint_{j:06d}.ct"

    def add_hint(
        self,
        j: int,
        rows: Dict[int, int],
        extra: int,
        parity: bytes,
    ) -> None:
        """Register a hint (subset ``rows`` + extra index) with its parity."""
        self._parity_path(j).write_bytes(parity)
        self.hints[j] = {
            "rows": dict(rows),
            "extra": int(extra),
            "parity_path": str(self._parity_path(j)),
        }
        self.label_hints.setdefault(int(extra), []).append(j)
        # A hint also contains y for every row in ``rows``.
        for k, row in rows.items():
            self.label_hints.setdefault(int(row), []).append(j)

    def build_initial_pool(
        self,
        known_labels: List[int],
        min_coverage: int = 8,
    ) -> Tuple[Dict[int, List[int]], Dict[int, Dict]]:
        """Derive the initial M hint subsets; return ``{j: [rows...]}`` so the
        caller can request parities from S in chunks.

        The caller must call :meth:`add_hint` for each j with the parity bytes
        returned by S.  Returns ``(req, topups)``:

        - ``req``: dict ``j -> full row list`` for the parity request
          (selected-half rows + extra row);
        - ``topups``: dict ``j -> {"row_list", "y", "picked_rows"}`` for
          deterministic top-up hints that guarantee every known label has at
          least ``min_coverage`` hints in the pool (the random hint pool alone
          would leave a label with only ~M/(2p) hints on average, which can be
          exhausted mid-batch).

        Top-up hints are valid R_y samples: they take the half of a fresh hint
        ID that does not select y's partition and use y as the extra index
        (exactly the replenishment construction).
        """
        self._known_labels = [int(y) for y in known_labels]
        req: Dict[int, List[int]] = {}
        for j in range(self.params.M):
            rows_a, _, extra, _ = hint_half_rows(self.seed, self.params, j)
            rows = dict(rows_a)
            row_list = list(rows.values()) + [extra]
            req[j] = row_list
            # Pre-register bookkeeping lazily in add_hint; here we just store
            # the structure to be completed when the parity arrives.
            self.hints[j] = {
                "rows": rows,
                "extra": int(extra),
                "parity_path": str(self._parity_path(j)),
                "_pending": True,
            }
        # Ensure every known label has at least one hint; raise early on
        # pathological parameter choices instead of failing mid-training.
        coverage = {y: 0 for y in self._known_labels}
        for j, h in self.hints.items():
            for y in coverage:
                if h["extra"] == y or any(row == y for row in h["rows"].values()):
                    coverage[y] += 1
        if int(min_coverage) < 1:
            raise ValueError("rms_min_coverage must be >= 1")

        # Deterministic top-ups so every known label has >= min_coverage hints.
        topups: Dict[int, Dict] = {}
        J = self.params.M
        for y, c in coverage.items():
            need = max(0, int(min_coverage) - c)
            ell = int(y) // self.params.p
            for _ in range(need):
                rows_a, rows_b, _, _ = hint_half_rows(self.seed, self.params, J)
                picked = rows_b if ell in rows_a else rows_a
                topups[J] = {
                    "row_list": list(picked.values()) + [int(y)],
                    "y": int(y),
                    "picked_rows": picked,
                }
                J += 1
        self.next_j = J
        if topups:
            logger.info(
                "RMS top-ups: %d additional hints to guarantee coverage "
                ">= %d per label", len(topups), int(min_coverage),
            )
        logger.info(
            "RMS initial pool: M=%d + %d top-ups, coverage=%s",
            self.params.M, len(topups), coverage,
        )
        return req, topups

    def complete_hint(self, j: int, parity: bytes) -> None:
        """Store the parity for a pending hint and index it by label."""
        h = self.hints[j]
        h.pop("_pending", None)
        self._parity_path(j).write_bytes(parity)
        h["parity_path"] = str(self._parity_path(j))
        self.label_hints.setdefault(int(h["extra"]), []).append(j)
        for row in h["rows"].values():
            self.label_hints.setdefault(int(row), []).append(j)

    # ------------------------------------------------------------------ #
    #  Online query
    # ------------------------------------------------------------------ #
    def pop_hint(self, y: int) -> Tuple[int, Dict[int, int], int, bytes]:
        """Take a hint containing ``y`` (extra index or one of its rows)."""
        y = int(y)
        pool = self.label_hints.get(y)
        if not pool:
            logger.error(
                "RMS pop_hint(%d) failed: pool empty. consumed=%s replenished=%s "
                "total_hints=%d next_j=%d",
                y, self.debug_consumed, self.debug_replenished,
                len(self.hints), self.next_j,
            )
            raise RuntimeError(
                f"RMS: no hint containing label {y} (pool empty); "
                "increase rms_lam"
            )
        j = pool.pop(0)
        self.debug_consumed[y] = self.debug_consumed.get(y, 0) + 1
        h = self.hints[j]
        parity = Path(h["parity_path"]).read_bytes()
        return j, h["rows"], int(h["extra"]), parity

    def build_query(
        self,
        j: int,
        rows: Dict[int, int],
        extra: int,
        y: int,
    ) -> Tuple[List[int], List[int], int]:
        """Build (real_subset, dummy_subset, permutation_bit).

        Real subset = hint subset minus ``y``; dummy subset covers all
        remaining partitions with one random row each.  Both subsets have
        exactly n/2 rows.  ``permutation_bit`` is 0 if real comes first.
        """
        rng = secrets.SystemRandom()
        n = self.params.n
        p = self.params.p
        real_rows = list(rows.values()) + [int(extra)]
        real_set = set(real_rows)
        real_set.discard(int(y))
        covered_parts = {row // p for row in real_set}
        dummy_parts = [k for k in range(n) if k not in covered_parts]
        dummy_rows = [k * p + rng.randrange(p) for k in dummy_parts]
        real_rows = list(real_set)
        if len(real_rows) != n // 2 or len(dummy_rows) != n // 2:
            raise RuntimeError(
                f"RMS query size mismatch: real={len(real_rows)} dummy={len(dummy_rows)} "
                f"expected {n // 2}"
            )
        perm = rng.randrange(2)
        if perm:
            return dummy_rows, real_rows, 1
        return real_rows, dummy_rows, 0

    def plan_replenish(self, y: int) -> Tuple[int, List[int], List[int]]:
        """Plan replenishment for target ``y``: next hint ID + both halves.

        Returns ``(J, half_a_rows, half_b_rows)``.  The caller asks S for the
        two half parities; the half whose partitions do not include y's
        partition is picked and y becomes the new extra index.
        """
        J = self.next_j
        self.next_j += 1
        rows_a, rows_b, _, _ = hint_half_rows(self.seed, self.params, J)
        return J, list(rows_a.values()), list(rows_b.values())

    def add_replenished(
        self,
        J: int,
        y: int,
        picked_half_rows: Dict[int, int],
        parity: bytes,
    ) -> None:
        """Store the replenished hint (picked half + ``y`` as extra index)."""
        self.add_hint(J, picked_half_rows, y, parity)
        self.debug_replenished[int(y)] = self.debug_replenished.get(int(y), 0) + 1


def pick_replenish_half(
    params: RMSHintParams,
    rows_a: Dict[int, int],
    rows_b: Dict[int, int],
    y: int,
) -> Tuple[Dict[int, int], int]:
    """Pick the half that does not select y's partition.

    Returns ``(picked_rows, picked_index)`` where picked_index is 0 for half A
    and 1 for half B, matching the order of the two parities returned by S.
    """
    ell = int(y) // params.p
    if ell in rows_a:
        return rows_b, 1
    return rows_a, 0
