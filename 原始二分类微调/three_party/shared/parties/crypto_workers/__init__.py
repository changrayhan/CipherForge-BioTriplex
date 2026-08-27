"""
CPU Crypto Workers for SLG-HE-PIR v2.0 heterogeneous architecture.

This package contains long-lived CPU worker subprocesses spawned by
``HeterogeneousProtocol``. Each worker:

  * owns its own SEAL context (independent of the GPU Fusion process)
  * never touches CUDA / torch tensors — operates on ``bytes`` / ``numpy``
  * is registered as the runtime's only handler for one specific
    cryptographic primitive (``add_mask``, ``decrypt_only``, or
    ``process_logits``)

Why CPU-only processes?
-----------------------
The Design-2 protocol contract has three CPU-bound primitives that, in the
old ``FusionProtocol`` / ``IPCProtocol`` designs, ran either in the same
Python process as the GPU forward (competing for the GIL and CUDA caching
allocator) or in three separate spawn processes (each with its own CUDA
context — ~4.5 GB wasted on RTX 5090).

By spinning up dedicated **CPU-only** worker processes (no CUDA context, no
torch tensors in their heap), we get:

  * true parallelism — workers don't fight the GIL of the GPU Fusion driver
  * clean separation — workers cannot accidentally access GPU model weights
  * memory isolation — ``sk_M`` only lives inside the ``Crypto_M`` worker

Privacy boundary
----------------
The CPU workers themselves don't enforce the privacy contract — they are
mechanically incapable of violating it because:

  * ``Crypto_UWorker`` never receives ``sk_M`` (the driver only forwards
    ``pk_pem`` to it), so it cannot decrypt.
  * ``Crypto_MWorker`` never receives ``prg_seed`` — the driver passes only
    masked ciphertexts (which are useless without ``R_t``).
  * ``Crypto_SWorker`` never receives ``sk_M`` and never receives U's mask.

The privacy contract is enforced by the ``HeterogeneousProtocol`` driver,
which is the only component that knows all three secrets (``pk_M``,
``sk_M``, ``prg_seed``) before distribution.
"""
