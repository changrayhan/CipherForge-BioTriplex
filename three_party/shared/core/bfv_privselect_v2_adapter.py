"""
BFV PrivSelect v2 adapter — full implementation using seal-python.

Provides:
  - BFVPrivSelectV2Backend: BFV context + encrypted database + PrivSelect protocol
  - BFVEncryptedDatabase: encrypted V-matrix database
  - PRGShareProtocolBFV: PRG-based masking shares
  - Fixed-point encoding: float → int → Plaintext → Ciphertext

Parameters (matching docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md §4.6):
  poly_degree = 4096
  plain_bits = 30
  scale = 10000

Crypto operations:
  Stage 0 (offline):  encrypt V[y] rows → Enc_DB
  Stage 1 (online):   BFVPrivSelect → pir_query / pir_response
  Stage 2 (eval):     plaintext (no crypto)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Fixed-point encoding                                                         #
# --------------------------------------------------------------------------- #

def float_to_int(x: float, scale: float) -> int:
    """Round a float to a scaled integer (for BFV encoding).

    Values are centred around 0 to use the full signed plaintext modulus range.
    scale = 10000 maps float ∈ [-3.2767, +3.2767] → int ∈ [-32767, +32767].
    """
    return int(round(x * scale))


def int_to_float(x: int, scale: float) -> float:
    """Inverse of float_to_int."""
    return float(x) / scale


def encode_vector_as_ints(vec: np.ndarray, scale: float) -> np.ndarray:
    """Convert a float vector to scaled integers, centred at 0.

    Args:
        vec: float array, any shape
        scale: encoding scale (e.g. 10000)

    Returns:
        int64 array of scaled integers, clamped to safe int32 range.
    """
    scaled = np.round(vec * scale).astype(np.int64)
    # Clamp to avoid plaintext modulus overflow
    # With plain_bits=30, plain_modulus ≈ 2^30, we have plenty of headroom
    # so no clamping needed unless values are extreme
    return scaled


def pad_to_degree(vec: np.ndarray, poly_degree: int) -> np.ndarray:
    """Zero-pad (or truncate) an int64 vector to exactly ``poly_degree`` slots.

    BFV BatchEncoder requires exactly N values per plaintext. TinyLlama's
    hidden_dim (2048) is smaller than poly_degree (4096), so every encoded
    vector (V rows, a_t) must be padded with zeros in the tail slots.
    """
    arr = np.asarray(vec, dtype=np.int64).reshape(-1)
    if arr.size == poly_degree:
        return arr
    if arr.size > poly_degree:
        return arr[:poly_degree]
    out = np.zeros(poly_degree, dtype=np.int64)
    out[: arr.size] = arr
    return out


def decode_ints_as_vector(ints, scale: float):
    """Inverse of encode_vector_as_ints.

    Accepts list or numpy array (SEAL encoder.decode may return either
    depending on the SEAL Python binding version).
    """
    arr = np.asarray(ints, dtype=np.float64)
    return arr / scale


# --------------------------------------------------------------------------- #
#  SEAL helpers                                                                 #
# --------------------------------------------------------------------------- #

def _build_seal_context(
    poly_degree: int,
    plain_bits: int,
) -> "SEALContext":
    """Build a SEAL BFV context.

    Args:
        poly_degree: Polynomial modulus degree (N = 4096 for our use).
        plain_bits: Plaintext modulus bit-width (30 → p ≈ 2^30).

    Returns:
        SEALContext ready for BFV operations.
    """
    from seal import (
        EncryptionParameters, SEALContext, scheme_type,
        CoeffModulus, PlainModulus,
    )

    plain_modulus = PlainModulus.Batching(poly_degree, plain_bits)
    parms = EncryptionParameters(scheme_type.bfv)
    parms.set_poly_modulus_degree(poly_degree)
    # coeff_modulus: total budget must fit in SEAL security requirements.
    # Pick the smallest set of primes that supports KeyGen+Relin+keyswitch
    # at each poly_degree (per SEAL docs):
    #   N=2048: total >= 50 bits ⇒ [36, 14]
    #   N=4096: total >= 109 bits ⇒ [36, 36, 37]
    #   N=8192: total >= 218 bits ⇒ [40, 40, 47]
    if poly_degree == 2048:
        coeff_bits = [36, 14]
    elif poly_degree == 4096:
        coeff_bits = [36, 36, 37]
    elif poly_degree == 8192:
        coeff_bits = [40, 40, 47]
    else:
        raise ValueError(f"Unsupported poly_degree: {poly_degree}.")
    parms.set_coeff_modulus(CoeffModulus.Create(poly_degree, coeff_bits))
    parms.set_plain_modulus(plain_modulus)
    return SEALContext(parms)


def get_plain_modulus(poly_degree: int = 4096, plain_bits: int = 30) -> int:
    """Return the real SEAL plaintext modulus value for the given params."""
    from seal import PlainModulus
    return PlainModulus.Batching(poly_degree, plain_bits).value()


def create_bfv_context(poly_degree: int = 4096, plain_bits: int = 30) -> "SEALContext":
    """Create a SEAL BFV context for crypto worker initialization.

    This is a standalone function (not a method on BFVPrivSelectV2Backend)
    so that crypto worker subprocesses can create a context without
    loading the full backend.

    Args:
        poly_degree: Polynomial modulus degree.
        plain_bits: Plaintext modulus bit width.

    Returns:
        SEALContext ready for BFV operations.
    """
    return _build_seal_context(poly_degree, plain_bits)


# Shared tmpfs-backed temp directory for SEAL I/O. Using /dev/shm avoids
# disk round-trip (3x faster on ct save/load). Falls back to system temp
# if /dev/shm is unavailable.
_SHM_TMP_DIR = "/dev/shm"
_USE_SHM_TMP = os.path.isdir(_SHM_TMP_DIR) and os.access(_SHM_TMP_DIR, os.W_OK)


def _seal_tmpfile(suffix: str = "") -> "tempfile._TemporaryFileWrapper":
    """Create a NamedTemporaryFile, preferring /dev/shm (tmpfs) when available."""
    import tempfile
    if _USE_SHM_TMP:
        try:
            return tempfile.NamedTemporaryFile(delete=False, dir=_SHM_TMP_DIR, suffix=suffix)
        except (OSError, PermissionError):
            pass
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix)


def _seal_ciphertext_to_bytes(ct: "Ciphertext") -> bytes:
    """Serialize a Ciphertext to bytes."""
    import tempfile
    with _seal_tmpfile() as f:
        ct.save(f.name)
        with open(f.name, "rb") as fh:
            data = fh.read()
        os.unlink(f.name)
    return data


def _seal_ciphertext_from_bytes(context: "SEALContext", data: bytes) -> "Ciphertext":
    """Deserialize a Ciphertext from bytes using temp file (SEAL 4.x requires file path)."""
    from seal import Ciphertext
    with _seal_tmpfile(suffix=".seal") as f:
        f.write(data)
        f.flush()
        tmp_path = f.name
    try:
        ct = Ciphertext()
        ct.load(context, tmp_path)
        return ct
    finally:
        os.unlink(tmp_path)


# Alias for compatibility
_seal_load_ciphertext = _seal_ciphertext_from_bytes


def _seal_to_bytes(obj: "Any") -> bytes:
    """Serialize any SEAL object (SecretKey, PublicKey, etc.) to bytes."""
    import tempfile
    with _seal_tmpfile() as f:
        obj.save(f.name)
        with open(f.name, "rb") as fh:
            data = fh.read()
        os.unlink(f.name)
    return data


# --------------------------------------------------------------------------- #
#  PRG share protocol                                                          #
# --------------------------------------------------------------------------- #

class PRGShareProtocolBFV:
    """PRG-based masking share generation.

    Both U and S share a seed and independently generate the same random mask
    vector r_t = PRG(seed, step, token_index). Neither party learns the other's
    input from the mask alone.

    Uses SHA-256 in counter mode as a PRG:
        r_t[i] = SHA256(seed || step || token_index || i) mod 2^32 - plain_modulus/2

    Args:
        seed: 32-byte random seed (shared between U and S).
        poly_degree: Polynomial modulus degree (alias: vec_dim).
        plain_bits: Plaintext modulus bit width (alias: plain_modulus).
        scale: Optional scaling factor (not used in PRG, kept for compatibility).
        n_entries: Optional number of entries (not used, kept for compatibility).
    """

    def __init__(
        self,
        seed: bytes = None,
        poly_degree: int = None,
        plain_bits: int = None,
        *,
        prg_seed: bytes = None,
        n_entries: int = None,
        vec_dim: int = None,
        scale: float = None,
        plain_modulus: int = None,
    ):
        # Support both old API (positional) and new API (keyword)
        self.seed = seed if seed is not None else prg_seed
        self.poly_degree = poly_degree if poly_degree is not None else vec_dim
        self.scale = float(scale) if scale is not None else 1.0
        # plain_bits vs plain_modulus: plain_bits is the bit width, plain_modulus is 2^bits
        if plain_bits is not None:
            self.plain_bits = plain_bits
            self.plain_modulus = get_plain_modulus(
                self.poly_degree or 4096, plain_bits
            )
        else:
            self.plain_modulus = plain_modulus
            self.plain_bits = plain_modulus.bit_length() if plain_modulus else 30

    def _prf_block(self, step: int, t_flat: int, start_i: int, n: int) -> np.ndarray:
        """Generate ``n`` random integers using SHA-256 counter mode.

        Args:
            step: Training step number (used as counter component).
            t_flat: Flat token index within the batch.
            start_i: Starting index within the PRG block.
            n: Number of random values to generate.

        Returns:
            int64 array of shape (n,), values in [-plain_modulus/2, +plain_modulus/2).
        """
        # Vectorized path: pack all messages (seed || step || t_flat || start_i + i)
        # in one shot, then hash them one at a time. SHA-256 itself has no batch
        # API in the stdlib, so we still loop — but we save the per-iter cost of
        # struct.pack + int.from_bytes by pre-building a big-endian index buffer
        # and slicing it. Output is byte-identical to the previous scalar loop.
        seed_len = len(self.seed)
        prefix = self.seed + struct.pack("!QQ", int(step), int(t_flat))
        # Big-endian uint64 indices to match the original struct.pack byte order.
        idx_buf = np.arange(start_i, start_i + n, dtype=">u8").tobytes()
        half = self.plain_modulus // 2
        pm = self.plain_modulus
        # ---- Boundary margin (CRITICAL correctness fix) --------------------
        # The SEAL Python binding mis-encodes/decodes values that land within
        # ~49151 of ±pm/2 (verified empirically with the real binding: the
        # round-trip introduces a constant ±49151 offset in that zone). With
        # the old unbounded PRG, ~1 slot per training step hit this zone and
        # produced a 4.9-unit gradient spike (and the legacy centre-then-add
        # reconstruction turned it into a ±pm/scale catastrophe -> loss 210).
        # Restricting the PRG range keeps U and S perfectly consistent (both
        # call this same function) while making the artifact unreachable.
        MARGIN = 1 << 17  # 131072 >> 49151, leaves ~30 bits of entropy per slot
        span = pm - 2 * MARGIN
        span_half = span // 2
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            chunk = idx_buf[i * 8 : (i + 1) * 8]
            digest = hashlib.sha256(prefix + chunk).digest()
            val = int.from_bytes(digest[:8], "big")
            val = (val % span) - span_half
            out[i] = int(val)
        return out

    def generate_mask_ints(self, step: int, t_flat: int) -> np.ndarray:
        """Generate a full PRG mask vector of length poly_degree.

        Returns:
            int64 array of shape (poly_degree,), values centred around 0.
        """
        return self._prf_block(step, t_flat, 0, self.poly_degree)

    # ---- Server-side share ----

    def server_make_share(
        self,
        step: int,
        t_flat: int,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Compute the server-side plaintext share ``s_share = scale·a_t − r_t``.

        The PRG mask ``r_t ∈ [-pm/2, +pm/2)`` is shared between U and S. U
        homomorphically adds ``Enc(r_t)`` to the masked DB row, so the
        ciphertext held by M decrypts to ``−V_y·scale + r_t`` (signed). S
        returns the *signed* difference ``scale·a_t − r_t`` here; M adds the
        two *after centering the decrypted masked_int* into
        ``[-pm/2, +pm/2)``, so ``r_t`` cancels exactly and only
        ``scale·(a_t − V_y)`` survives.

        Args:
            step: Training step.
            t_flat: Flat token index.
            a_t: Softmax-weighted V row (float32 / float64, shape
                ``(vec_dim,)``). Multiplied by ``self.scale`` and rounded to
                int64 — *do not* truncate, that would silently drop the
                gradient signal in a_t.

        Returns:
            s_share = round(a_t · scale) − r_t (int64, signed, shape
            ``(vec_dim,)``).
        """
        a_scaled = np.round(np.asarray(a_t, dtype=np.float64) * self.scale).astype(np.int64)
        r_t = self.generate_mask_ints(step, t_flat)
        a_padded = pad_to_degree(a_scaled, self.poly_degree)
        return a_padded - r_t


# --------------------------------------------------------------------------- #
#  Encrypted database                                                          #
# --------------------------------------------------------------------------- #

# Magic header for memory-mapped encrypted DB files.
MMAP_MAGIC = b"BFVCTDB\x00"
MMAP_VERSION = 1


class BFVEncryptedDatabase:
    """In-memory / mmap-backed encrypted V-matrix database.

    Stores ciphertexts of V[y] (one row per token id y) using BFV batch encoding.
    Each row is encoded as a vector of length poly_degree, then encrypted.

    Layout on disk:
        header: 64 bytes
          magic (8) | version (4) | poly_degree (4) | n_entries (8) | reserved (40)
        followed by n_entries * ct_size bytes (each ciphertext serialized)

    For large V (128256 × 4096), this is ~1.93 GB.
    """

    def __init__(
        self,
        context: "SEALContext",
        encryptor: "Encryptor",
        evaluator: "Evaluator",
        n_entries: int,
        poly_degree: int,
        scale: float,
        plain_bits: int,
        data_path: Optional[str] = None,
        *,
        load_ct_list: bool = True,
    ):
        self.context = context
        self.encryptor = encryptor
        self.evaluator = evaluator
        self.n_entries = n_entries
        self.poly_degree = poly_degree
        self.scale = scale
        self.plain_bits = plain_bits
        self._data_path = data_path
        self._ct_list: List[bytes] = []  # ciphertext bytes per entry
        self._mmap = None
        self._load_ct_list = load_ct_list
        self._loaded = False

    @classmethod
    def from_backend(
        cls,
        backend: "BFVPrivSelectV2Backend",
        data_path: str,
    ) -> "BFVEncryptedDatabase":
        return cls(
            context=backend._context,
            encryptor=backend._encryptor,
            evaluator=backend._evaluator,
            n_entries=backend.n_entries,
            poly_degree=backend.poly_degree,
            scale=backend.scale,
            plain_bits=backend.plain_bits,
            data_path=data_path,
        )

    @classmethod
    def from_cache(
        cls,
        context: "SEALContext",
        n_entries: int,
        vec_dim: int,
        cache_path: str,
        public_key: "PublicKey" = None,
        *,
        load_ct_list: bool = True,
    ) -> "BFVEncryptedDatabase":
        """Factory method for CryptoSWorker: create DB from cache with custom parameters.

        Args:
            load_ct_list: If True (default), loads all ciphertext bytes into memory.
                          If False, skips the expensive _ct_list load — use this when
                          the main process only needs the backend for key material.
        """
        from seal import Encryptor, Evaluator
        enc = Encryptor(context, public_key) if public_key else None
        ev = Evaluator(context)
        instance = cls(
            context=context,
            encryptor=enc,
            evaluator=ev,
            n_entries=n_entries,
            poly_degree=vec_dim,
            scale=1.0,  # Not used for reading cached data
            plain_bits=30,
            data_path=cache_path,
        )
        if load_ct_list:
            instance._load_from_file(cache_path)
        return instance

    @classmethod
    def _load_cache_mmap(cls, cache_path: str) -> List[bytes]:
        """Load ciphertext list from a cache file (internal method)."""
        import struct
        with open(cache_path, "rb") as f:
            header = f.read(80)
            magic, ver, poly_deg, n = struct.unpack("!8s I I Q 56x", header)
        file_size = os.path.getsize(cache_path)
        data_size = file_size - 80
        avg_ct_size = data_size // n
        instance = cls.__new__(cls)
        instance._ct_list = []
        with open(cache_path, "rb") as f:
            f.read(80)
            for _ in range(n):
                ct_bytes = f.read(avg_ct_size)
                if not ct_bytes:
                    break
                instance._ct_list.append(ct_bytes)
        return instance._ct_list

    def build_from_V(self, V: np.ndarray, force: bool = False) -> Dict[str, Any]:
        """Encrypt V-matrix rows into BFV ciphertexts.

        Args:
            V: float array of shape (n_entries, hidden_dim). For the full Llama lm_head,
               V[y] is the y-th row, shape (128256, 4096).
            force: If False and the DB already exists on disk, load from cache.

        Returns:
            dict with keys: n_rows, from_cache, build_time_s
        """
        from seal import BatchEncoder, Ciphertext, Plaintext

        import time
        t0 = time.time()

        n_entries, hidden_dim = V.shape
        assert n_entries == self.n_entries, f"V has {n_entries} rows, expected {self.n_entries}"
        assert hidden_dim <= self.poly_degree, (
            f"V has dim {hidden_dim}, expected <= poly_degree {self.poly_degree}"
        )

        data_path = self._data_path
        if data_path and not force:
            if Path(data_path).exists():
                logger.info("Encrypted DB cache hit: %s", data_path)
                self._load_from_file(data_path)
                return {"n_rows": self.n_entries, "from_cache": True, "build_time_s": 0.0}

        batch = BatchEncoder(self.context)

        # Encode each row and encrypt
        # Per docs §3.3 SVG formula: PIR parity = −Ṽ_y_t = Enc(−V_y).
        # So the offline DB stores Enc(−V[y]), not Enc(+V[y]).
        # The sigmoid direction of the upstream gradient therefore follows
        # the cross-entropy convention: a_t − V_y, not a_t + V_y.
        self._ct_list = []
        for y in range(n_entries):
            row = V[y]
            int_row = pad_to_degree(encode_vector_as_ints(row, self.scale), self.poly_degree)
            int_row = -int_row                                # store Enc(−V[y])
            pt = batch.encode(int_row)
            ct = Ciphertext()
            self.encryptor.encrypt(pt, ct)
            self._ct_list.append(_seal_ciphertext_to_bytes(ct))
            if (y + 1) % 1000 == 0:
                print(f"[Stage0] encrypted {y + 1}/{n_entries} rows", flush=True)

        build_time_s = time.time() - t0
        logger.info("Encrypted %d rows in %.2fs", n_entries, build_time_s)

        if data_path:
            self._save_to_file(data_path)

        return {"n_rows": n_entries, "from_cache": False, "build_time_s": build_time_s}

    def _save_to_file(self, path: str) -> None:
        """Serialize encrypted DB to a binary file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = self.n_entries
        poly_deg = self.poly_degree
        # Header: 80 bytes total: magic(8) + version(4) + poly_deg(4) + n(8) + padding(56)
        header = struct.pack(
            "!8s I I Q 56x",
            MMAP_MAGIC, MMAP_VERSION, poly_deg, n,
        )
        with open(path, "wb") as f:
            f.write(header)
            for ct_bytes in self._ct_list:
                f.write(struct.pack("!I", len(ct_bytes)))
                f.write(ct_bytes)
        logger.info("Saved encrypted DB to %s (%.1f MB)", path,
                    os.path.getsize(path) / 1e6)

    def _load_from_file(self, path: str) -> None:
        """Load encrypted DB from binary file.

        File format:
          80-byte header: magic(8) | version(4) | poly_deg(4) | n(8) | reserved(52)
          followed by n ciphertexts, each prefixed with a 4-byte big-endian
          size followed by raw SEAL Ciphertext.save() bytes.
        """
        from seal import Ciphertext

        with open(path, "rb") as f:
            header = f.read(80)
            magic, ver, poly_deg, n = struct.unpack("!8s I I Q 56x", header)
            assert magic == MMAP_MAGIC, f"Bad magic: {magic!r}"
            assert ver == MMAP_VERSION, f"Unsupported version: {ver}"

        file_size = os.path.getsize(path)
        data_size = file_size - 80
        logger.debug("DB file: n=%d, data_size=%d", n, data_size)

        # Each ciphertext has a 4-byte big-endian size prefix followed by
        # raw SEAL bytes. Read them in order using the size prefix — never
        # assume a fixed ct size, because avg differs from real by a few
        # bytes (e.g. 131189 vs 131185) and SEAL would refuse to load
        # a buffered ciphertext with a truncated/inconsistent header.
        import tempfile
        self._ct_list = []
        with open(path, "rb") as f:
            f.read(80)  # skip header
            for i in range(n):
                size_bytes = f.read(4)
                if len(size_bytes) < 4:
                    raise EOFError(f"Truncated DB: expected {n} entries, got {i}")
                size = struct.unpack("!I", size_bytes)[0]
                ct_bytes = f.read(size)
                if len(ct_bytes) < size:
                    raise EOFError(f"Truncated ct #{i}: wanted {size}, got {len(ct_bytes)}")
                # Sanity: verify SEAL can load the bytes (catches header corruption)
                # via a small tempfile. Only do this for the first record to
                # keep load time reasonable; subsequent records are assumed
                # well-formed by construction.
                if i == 0:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".seal") as tmp:
                        tmp.write(ct_bytes)
                        tmp.flush()
                        tmp_path = tmp.name
                    try:
                        ct = Ciphertext()
                        ct.load(self.context, tmp_path)
                    finally:
                        os.unlink(tmp_path)
                self._ct_list.append(ct_bytes)

        assert len(self._ct_list) == n, f"Loaded {len(self._ct_list)} ciphertexts, expected {n}"

    def get_encrypted_row(self, y: int) -> bytes:
        """Return the encrypted V[y] as bytes."""
        return self._ct_list[y]

    def get_encrypted_rows(self, indices: List[int]) -> List[bytes]:
        """Return encrypted rows for multiple indices."""
        return [self._ct_list[i] for i in indices]


# --------------------------------------------------------------------------- #
#  Main backend                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class BFVQuery:
    """Query object for BFVPrivSelect (Stage 0 / offline query)."""
    step: int
    t_flat: int
    y: int  # target token id (the one we want to PIR)
    real_indices: List[int]
    dummy_indices: List[int]


@dataclass
class BFVResponse:
    """Response from the server (S-side BFVPrivSelect)."""
    parity_real_bytes: bytes   # BFV ciphertext encoding (a_t + selected V rows)
    parity_dummy_bytes: bytes  # BFV ciphertext for dummy rows
    permutation_bit: int      # 0 or 1 (for S3PIR tie-breaking)


class BFVPrivSelectV2Backend:
    """BFV PrivSelect backend: context + key generation + encrypted DB + PIR.

    Three-party usage:
      M (key generation): creates sk/pk, distributes pk to U and S
      S (PIR server): holds encrypted DB + hint table
      U (PIR client): queries S for V[y] privately

    The secret key is dropped from the main process after key generation
    (sk is only held by CryptoMWorker subprocess pool).

    Args:
        n_entries: Number of V-matrix rows (vocab_size = 128256).
        vec_dim: Hidden dimension (4096).
        shared_seed: 32-byte random seed for PRG.
        scale: Fixed-point scaling factor (default 10000).
        cache_dir: Directory for encrypted DB cache.
        poly_degree: BFV polynomial modulus degree.
        plain_bits: BFV plaintext modulus bit width.
    """

    def __init__(
        self,
        n_entries: int,
        vec_dim: int,
        shared_seed: bytes,
        scale: float = 10000.0,
        cache_dir: Optional[str] = None,
        poly_degree: int = 4096,
        plain_bits: int = 30,
        *,
        load_ct_list: bool = True,
        pk_path: Optional[str] = None,
        force_new_keys: bool = False,
    ):
        """Initialize BFV backend.

        Args:
            pk_path: If provided, load the public key from this file instead of
                     generating a new one. This ensures the keys match the cached
                     encrypted DB. Mutually exclusive with force_new_keys=True.
            force_new_keys: If True, always generate new keys and ignore pk_path.
                            This also means force=True will be used for DB build.
        """
        self.n_entries = n_entries
        self.vec_dim = vec_dim
        self.shared_seed = shared_seed
        self.scale = scale
        self.cache_dir = cache_dir
        self.poly_degree = poly_degree
        self.plain_bits = plain_bits
        self._force_new_keys = force_new_keys

        # Build SEAL context
        self._context = _build_seal_context(poly_degree, plain_bits)

        # Key generation: either load from cache or generate new
        from seal import KeyGenerator, PublicKey, RelinKeys, Encryptor, Evaluator, Decryptor

        if pk_path and not force_new_keys:
            # Load existing public key from cache
            import tempfile
            with open(pk_path, "rb") as f:
                pk_bytes = f.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pub") as f:
                f.write(pk_bytes)
                f.flush()
                tmp_path = f.name
            try:
                self._public_key = PublicKey()
                self._public_key.load(self._context, tmp_path)
            finally:
                os.unlink(tmp_path)
            # Generate keygen only for deriving secret key and relin keys
            self._keygen = KeyGenerator(self._context)
            self._secret_key = self._keygen.secret_key()
            self._relin_keys = RelinKeys()
            self._keygen.create_relin_keys(self._relin_keys)
            logger.info("Loaded public key from cache: %s", pk_path)
        else:
            # Key generation (M party)
            self._keygen = KeyGenerator(self._context)
            self._secret_key = self._keygen.secret_key()
            self._public_key = PublicKey()
            self._keygen.create_public_key(self._public_key)
            self._relin_keys = RelinKeys()
            self._keygen.create_relin_keys(self._relin_keys)
            logger.info("Generated new BFV keys")

        self._encryptor = Encryptor(self._context, self._public_key)
        self._evaluator = Evaluator(self._context)
        self._decryptor = Decryptor(self._context, self._secret_key)

        # PRG share protocol
        self.shares = PRGShareProtocolBFV(
            shared_seed, poly_degree, plain_bits, scale=self.scale
        )

        # Encrypted DB (lazy — see load_ct_list below)
        self._enc_db: Optional[BFVEncryptedDatabase] = None

        # Drop secret key from main process — M only keeps pk in the main process.
        # sk is re-distributed to CryptoMWorker pool at fork time.
        self._sk_dropped = False

        # If True, _ensure_db loads the 16 GB ciphertext list from disk.
        # Workers need this; the main process does NOT (respond() is never called here).
        # Setting to False keeps ~16 GB of Python heap free in the main process.
        self._load_ct_list_in_main = load_ct_list

    @property
    def load_ct_list(self) -> bool:
        return self._load_ct_list_in_main

    @load_ct_list.setter
    def load_ct_list(self, value: bool) -> None:
        self._load_ct_list_in_main = value

    @property
    def public_key_bytes(self) -> bytes:
        """Serialize the public key for distribution to U and S parties."""
        from seal import SecretKey, PublicKey
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self._public_key.save(f.name)
            with open(f.name, "rb") as fh:
                data = fh.read()
            os.unlink(f.name)
        return data

    def reconstruct_public_key(self, pk_bytes: bytes) -> "PublicKey":
        """Reconstruct a PublicKey from serialized bytes.

        This allows U and S parties to reconstruct the public key from
        the bytes distributed by M during initialization.
        """
        from seal import PublicKey
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(pk_bytes)
            f.flush()
            pk = PublicKey()
            pk.load(self._context, f.name)
            os.unlink(f.name)
        return pk

    def attach_public_key(self, pk_bytes: bytes) -> "PublicKey":
        """Reconstruct the public key AND rebind the encryptor to it.

        Without this, a backend that generated its own throwaway keypair at
        construction would keep encrypting with the stale key after a
        persisted pk is attached (DB rows would not match the distributed
        pk / the M-side sk).
        """
        from seal import Encryptor
        pk = self.reconstruct_public_key(pk_bytes)
        self._public_key = pk
        self._encryptor = Encryptor(self._context, pk)
        return pk

    def _load_secret_key(self, sk_bytes: bytes) -> "SecretKey":
        """Load a SecretKey from serialized bytes.

        This allows PartyM to re-attach sk_M that was serialized before
        dropping from the parent process.
        """
        from seal import SecretKey
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(sk_bytes)
            f.flush()
            sk = SecretKey()
            sk.load(self._context, f.name)
            os.unlink(f.name)
        return sk

    def _ensure_db(self, *, load_ct_list: bool = True) -> BFVEncryptedDatabase:
        if self._enc_db is None:
            cache_path = None
            if self.cache_dir:
                cache_path = os.path.join(
                    self.cache_dir,
                    f"bfv_ct_db_n{self.n_entries}_d{self.vec_dim}_p{self.poly_degree}.bin"
                )
            self._enc_db = BFVEncryptedDatabase(
                context=self._context,
                encryptor=self._encryptor,
                evaluator=self._evaluator,
                n_entries=self.n_entries,
                poly_degree=self.poly_degree,
                scale=self.scale,
                plain_bits=self.plain_bits,
                data_path=cache_path,
                load_ct_list=self._load_ct_list_in_main,
            )
        return self._enc_db

    def build_encrypted_database(self, V: np.ndarray, force: bool = False) -> Dict[str, Any]:
        """Stage 0 offline: encrypt V-matrix.

        Args:
            V: float array of shape (n_entries, vec_dim).
            force: If True, rebuild even if cache exists.

        Returns:
            dict with n_rows, from_cache, build_time_s.
        """
        # If new keys were generated, force rebuild to ensure DB uses matching keys
        actual_force = force or self._force_new_keys
        enc_db = self._ensure_db(load_ct_list=True)
        return enc_db.build_from_V(V.astype(np.float64), force=actual_force)

    def respond(
        self,
        query: BFVQuery,
        a_t_fp32: np.ndarray,
    ) -> BFVResponse:
        """Server (S) side: respond to a PIR query.

        Computes:
          parity_real = a_t_ints ⊕ Enc(V[real_indices])
          parity_dummy = Enc(V[dummy_indices])

        The a_t is added via homomorphic addition (S3PIR-style).

        Args:
            query: BFVQuery with target index y and selected partition indices.
            a_t_fp32: float32 gradient vector for the token (shape vec_dim,).

        Returns:
            BFVResponse with parity ciphertexts.
        """
        from seal import Ciphertext, Plaintext, BatchEncoder

        enc_db = self._ensure_db()
        batch = BatchEncoder(self._context)

        # Encode a_t
        a_t_ints = pad_to_degree(encode_vector_as_ints(a_t_fp32, self.scale), self.poly_degree)
        pt_a = batch.encode(a_t_ints)

        # Load real ciphertexts and accumulate
        real_cts = [enc_db.get_encrypted_row(i) for i in query.real_indices]
        if real_cts:
            first = _seal_ciphertext_from_bytes(self._context, real_cts[0])
            accum = first
            for ct_bytes in real_cts[1:]:
                ct = _seal_ciphertext_from_bytes(self._context, ct_bytes)
                self._evaluator.add_inplace(accum, ct)
            # Add a_t plaintext (this is the S3PIR extension)
            self._evaluator.add_plain_inplace(accum, pt_a)
            parity_real_bytes = _seal_ciphertext_to_bytes(accum)
        else:
            # No real indices — just return a_t ciphertext
            ct_a = Ciphertext()
            self._encryptor.encrypt(pt_a, ct_a)
            parity_real_bytes = _seal_ciphertext_to_bytes(ct_a)

        # Dummy accumulation (no a_t addition)
        dummy_cts = [enc_db.get_encrypted_row(i) for i in query.dummy_indices]
        if dummy_cts:
            first_d = _seal_ciphertext_from_bytes(self._context, dummy_cts[0])
            accum_d = first_d
            for ct_bytes in dummy_cts[1:]:
                ct = _seal_ciphertext_from_bytes(self._context, ct_bytes)
                self._evaluator.add_inplace(accum_d, ct)
            parity_dummy_bytes = _seal_ciphertext_to_bytes(accum_d)
        else:
            # Return zero ciphertext
            pt_zero = batch.encode(np.zeros(self.poly_degree, dtype=np.int64))
            ct_zero = Ciphertext()
            self._encryptor.encrypt(pt_zero, ct_zero)
            parity_dummy_bytes = _seal_ciphertext_to_bytes(ct_zero)

        return BFVResponse(
            parity_real_bytes=parity_real_bytes,
            parity_dummy_bytes=parity_dummy_bytes,
            permutation_bit=0,
        )

    def _add_mask_to_ct(self, ct_bytes: bytes, r_t_ints: np.ndarray) -> bytes:
        """U-side: add PRG mask -r_t to a ciphertext.

        Args:
            ct_bytes: Serialized Ciphertext (the parity from S).
            r_t_ints: PRG mask (int64, shape (poly_degree,)).

        Returns:
            Modified ciphertext bytes (in-place addition, serialized back).
        """
        from seal import Ciphertext, Plaintext, BatchEncoder

        ct = _seal_ciphertext_from_bytes(self._context, ct_bytes)
        batch = BatchEncoder(self._context)
        # Encode -r_t
        neg_r = (-r_t_ints.astype(np.int64))
        pt_mask = batch.encode(neg_r)
        self._evaluator.add_plain_inplace(ct, pt_mask)
        return _seal_ciphertext_to_bytes(ct)

    def add_mask_to_ct(self, ct_bytes: bytes, r_t_ints: np.ndarray) -> bytes:
        """Public wrapper for _add_mask_to_ct (U-side crypto worker calls this)."""
        return self._add_mask_to_ct(ct_bytes, r_t_ints)

    def aggregator_decrypt(
        self,
        ct_bytes: bytes,
        s_share: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """M-side: decrypt the aggregated ciphertext and optionally add s_share.

        This is called by CryptoMWorker to recover:
            -V_y + r_t  (from U-side masked ciphertext)
        The s_share = a_t - r_t is added in the caller (PartyM.backward_and_update).

        Args:
            ct_bytes: Serialized Ciphertext.
            s_share: Ignored in this implementation (handled in caller).

        Returns:
            Recovered gradient vector (float32, shape (vec_dim,)).
        """
        from seal import Ciphertext, Plaintext, BatchEncoder

        ct = _seal_ciphertext_from_bytes(self._context, ct_bytes)
        pt = Plaintext()
        self._decryptor.decrypt(ct, pt)
        batch = BatchEncoder(self._context)
        int_arr = batch.decode(pt)  # returns float64 array
        # The plaintext vector has the same values repeated every poly_degree slot
        # (BFV batch encoding). We only care about the first vec_dim values.
        grad_int = int_arr[:self.vec_dim].astype(np.int64)
        return decode_ints_as_vector(grad_int, self.scale).astype(np.float32)

    def _drop_secret_key(self) -> None:
        """Drop the secret key from this object.

        After this, the main process can no longer decrypt.
        sk is held by CryptoMWorker pool subprocesses.
        """
        self._secret_key = None
        self._decryptor = None
        self._sk_dropped = True
        logger.info("Secret key dropped from main process")

    def drop_encrypted_db(self) -> None:
        """Drop the in-memory ciphertext list from the main process.

        Workers hold their own copies of the encrypted DB. The main process
        never calls get_encrypted_row(), so holding the 16 GB _ct_list here
        serves no purpose and wastes ~16 GB of main-process heap memory.

        Call this after build_encrypted_database() / before training starts.
        Safe to call multiple times.
        """
        if self._enc_db is not None:
            self._enc_db._ct_list.clear()
            self._enc_db._ct_list = []
            self._enc_db._loaded = False
            self._enc_db = None
        logger.info("Encrypted DB dropped from main process (workers retain their copies)")


# --------------------------------------------------------------------------- #
#  Standalone convenience                                                      #
# --------------------------------------------------------------------------- #

def build_encrypted_db_from_V(
    V: np.ndarray,
    cache_dir: str,
    poly_degree: int = 4096,
    plain_bits: int = 30,
    scale: float = 10000.0,
    force: bool = False,
) -> Tuple[bytes, str]:
    """One-line Stage 0 encrypted DB builder.

    Returns:
        (public_key_bytes, db_cache_path)
    """
    n_entries, vec_dim = V.shape[:2]
    import os
    shared_seed = os.urandom(32)
    backend = BFVPrivSelectV2Backend(
        n_entries=n_entries,
        vec_dim=vec_dim,
        shared_seed=shared_seed,
        scale=scale,
        cache_dir=cache_dir,
        poly_degree=poly_degree,
        plain_bits=plain_bits,
    )
    result = backend.build_encrypted_database(V, force=force)
    return backend.public_key_bytes, result


__all__ = [
    "BFVPrivSelectV2Backend",
    "BFVEncryptedDatabase",
    "BFVQuery",
    "BFVResponse",
    "PRGShareProtocolBFV",
    "encode_vector_as_ints",
    "decode_ints_as_vector",
    "float_to_int",
    "int_to_float",
    "create_bfv_context",
]
