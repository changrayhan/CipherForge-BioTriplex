"""S3PIR Hint Table implementation for SLG-HE-PIR.

S3PIR (SublinearPIR) uses a hint-based approach to reduce communication cost.
Instead of downloading the entire encrypted database for each query, the client
pre-computes "hints" during an offline phase that allow answering queries with
sublinear communication.

The hint table divides the vocabulary into partitions and stores metadata
(cutoff values, extra indices) for each partition. The actual hints are stored
in JSON format and loaded on demand.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HintTable:
    """S3PIR hint table for sublinear PIR queries.

    The hint table divides n_entries into partitions of size partition_size.
    For each partition, we store:
      - cutoff: a threshold value for the partition
      - extra_index: an additional index within the partition
      - n_indices: number of indices in the partition

    The "skeleton" hints are the minimal metadata needed for query construction.
    The full hint table (with real/dummy indices) is built from this metadata.

    Attributes:
        n_entries: Total number of vocabulary entries (e.g., 128256 for Llama).
        partition_size: Size of each partition (sqrt(n_entries)).
        n_partitions: Number of partitions.
        lam: Security parameter (lambda).
        main_hints: List of main hint metadata per partition.
        backup_hints: List of backup hint lists per partition.
    """

    def __init__(
        self,
        n_entries: int,
        partition_size: int,
        lam: int,
        cache_dir: Optional[str] = None,
    ):
        self.n_entries = n_entries
        self.partition_size = partition_size
        self.n_partitions = math.ceil(n_entries / partition_size)
        self.lam = lam
        self.cache_dir = cache_dir
        self.main_hints: List[Dict[str, Any]] = []
        self.backup_hints: List[List[Dict[str, int]]] = []

    # ------------------------------------------------------------------------- #
    #  Skeleton hint computation (offline phase)                                #
    # ------------------------------------------------------------------------- #

    def compute_main_hints_skeleton(self) -> None:
        """Compute the main hint skeleton for each partition.

        This is the offline computation phase. For each partition i, we compute:
          - cutoff: the maximum index in the partition
          - extra_index: an additional index within the partition (for redundancy)
          - n_indices: number of indices in the partition
        """
        self.main_hints = []
        for i in range(self.n_partitions):
            start = i * self.partition_size
            end = min(start + self.partition_size, self.n_entries)
            n_indices = end - start

            # Cutoff is the maximum index in this partition
            cutoff = end - 1

            # Extra index: last index in the partition (for robustness)
            extra_index = end - 1

            self.main_hints.append({
                "partition_idx": i,
                "cutoff": cutoff,
                "extra_index": extra_index,
                "n_indices": n_indices,
            })

        logger.info(
            "Computed main hints skeleton: %d partitions, partition_size=%d",
            self.n_partitions, self.partition_size,
        )

    def compute_backup_hints_skeleton(self) -> None:
        """Compute backup hints for each partition.

        Backup hints provide an additional layer of indirection for query construction.
        For each partition, we generate lam random index pairs.
        """
        import random
        random.seed(42)  # Deterministic for reproducibility

        self.backup_hints = []
        for i in range(self.n_partitions):
            start = i * self.partition_size
            end = min(start + self.partition_size, self.n_entries)
            partition_indices = list(range(start, end))

            # Generate lambda random pairs
            pairs = []
            for _ in range(self.lam):
                idx_a = random.choice(partition_indices)
                idx_b = random.choice(partition_indices)
                pairs.append({"index_a": idx_a, "index_b": idx_b})

            self.backup_hints.append(pairs)

        logger.info(
            "Computed backup hints: %d pairs per partition, lam=%d",
            self.lam, self.lam,
        )

    # ------------------------------------------------------------------------- #
    #  Query construction                                                       #
    # ------------------------------------------------------------------------- #

    def find_hint_for(self, index: int) -> Optional[Dict[str, Any]]:
        """Find the main hint for a given index.

        Args:
            index: The vocabulary index to look up.

        Returns:
            The hint dict for the partition containing index, or None if not found.
        """
        if index < 0 or index >= self.n_entries:
            return None

        partition_idx = index // self.partition_size
        if partition_idx >= len(self.main_hints):
            partition_idx = len(self.main_hints) - 1

        return self.main_hints[partition_idx]

    def build_query_for(self, index: int) -> Tuple[List[int], List[int], int]:
        """Build a PIR query for the given index.

        Returns:
            (real_indices, dummy_indices, permutation_bit)

            - real_indices: the indices that encode the target (including the index itself)
            - dummy_indices: additional indices for the query
            - permutation_bit: whether to permute the indices
        """
        hint = self.find_hint_for(index)
        if hint is None:
            raise ValueError(f"Index {index} out of range")

        partition_idx = hint["partition_idx"]
        cutoff = hint["cutoff"]
        start = partition_idx * self.partition_size
        end = min(start + self.partition_size, self.n_entries)

        # In a full S3PIR implementation, real_indices would be determined by
        # the hint's structure (e.g., prefix bits). Here we use a simplified
        # scheme where real_indices contains the target and dummy_indices
        # contains other indices in the partition.

        # The target index is always in real_indices
        real_indices = [index]

        # Dummy indices: all other indices in the partition except the target
        all_partition_indices = list(range(start, end))
        dummy_indices = [i for i in all_partition_indices if i != index]

        # Permutation bit: based on whether index is before or after cutoff
        permutation_bit = 0 if index <= cutoff else 1

        return real_indices, dummy_indices, permutation_bit

    # ------------------------------------------------------------------------- #
    #  Persistence                                                              #
    # ------------------------------------------------------------------------- #

    def to_cache_files(self) -> None:
        """Save hint table to JSON cache files."""
        if self.cache_dir is None:
            raise ValueError("cache_dir is required for persistence")

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "n_entries": self.n_entries,
            "partition_size": self.partition_size,
            "n_partitions": self.n_partitions,
            "lam": self.lam,
            "main_hints": self.main_hints,
            "backup_hints": self.backup_hints,
        }

        path = cache_dir / "hint_table.json"
        with open(path, "w") as f:
            json.dump(data, f)

        logger.info("Saved hint table to %s", path)

    @classmethod
    def from_cache_files(cls, cache_dir: str) -> "HintTable":
        """Load hint table from JSON cache files."""
        path = Path(cache_dir) / "hint_table.json"
        if not path.exists():
            raise FileNotFoundError(f"hint_table.json not found in {cache_dir}")

        with open(path) as f:
            data = json.load(f)

        instance = cls(
            n_entries=data["n_entries"],
            partition_size=data["partition_size"],
            lam=data["lam"],
            cache_dir=cache_dir,
        )
        instance.main_hints = data["main_hints"]
        instance.backup_hints = data["backup_hints"]
        instance.n_partitions = data["n_partitions"]

        logger.info(
            "Loaded hint table from cache: n_entries=%d, partitions=%d, lam=%d",
            instance.n_entries, instance.n_partitions, instance.lam,
        )
        return instance
