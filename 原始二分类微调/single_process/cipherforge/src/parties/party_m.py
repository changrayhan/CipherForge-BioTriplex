"""
Party M (Model) — decoder[16..32) + LoRA + sk_M.

Privacy contract:
  - M holds decoder[16..32) + norm + LoRA parameters
  - M is the ONLY party that ever holds ``sk_M`` (re-attached in __init__
    after the main process called ``_drop_secret_key()`` on the parent
    backend)
  - M decrypts ciphertexts from U and plaintext-adds ``s_share`` from S to
    recover ``a_t - V_y`` gradients
  - M is the only party that updates trainable parameters

In the heterogeneous protocol, the BFV decryption step is delegated to a
``CryptoMWorker`` pool (long-lived fork). The decryptor lives inside the
worker, not in this object — this object only orchestrates: receives the
plaintext ``g_H`` numpy array, moves it back to GPU, and runs the autograd
+ LoRA step.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PartyM:
    """Model-side party (M) for SLG-HE-PIR."""

    def __init__(
        self,
        model_path: str,
        bfv_sk_pem: bytes,
        bfv_pk_pem: bytes,
        config: Dict,
        crypto_m_pool=None,
    ):
        self.config = config
        self.crypto_m_pool = crypto_m_pool
        self._setup_device()
        self._setup_submodel(model_path)
        self._setup_bfv(bfv_sk_pem, bfv_pk_pem)
        self._setup_optimizer()

    # ------------------------------------------------------------------------- #
    #  Setup
    # ------------------------------------------------------------------------- #
    def _setup_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[M-device] device={self.device}", flush=True)

    def _setup_submodel(self, model_path: str) -> None:
        from ..model.model_splitting import (
            detect_model_spec,
            load_m_submodel_with_lora,
            _get_shared_weights,
        )
        self.spec = detect_model_spec(model_path, u_layers=int(self.config.get("u_layers", 0)))
        # Pre-load weights once to avoid loading the 16GB model twice (U + M)
        all_weights = _get_shared_weights(model_path)
        self.model = load_m_submodel_with_lora(
            spec=self.spec,
            model_path=model_path,
            device=str(self.device),
            lora_rank=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 16)),
            lora_dropout=float(self.config.get("lora_dropout", 0.0)),
            use_flash_attention=bool(self.config.get("use_flash_attention", True)),
            use_sage_attention=bool(self.config.get("use_sage_attention", True)),
            gradient_checkpointing_style=self.config.get("gradient_checkpointing_style", "reentrant"),
            use_deepspeed_zero=bool(self.config.get("use_deepspeed_zero", False)),
            zero_stage=int(self.config.get("zero_stage", 1)),
            all_weights=all_weights,
        )
        self._hidden_dim = self.config["hidden_dim"]

    def _setup_bfv(self, bfv_sk_pem: bytes, bfv_pk_pem: bytes) -> None:
        """Construct M's *local* BFV backend with sk_M attached.

        The backend here is used for legacy ``decrypt_only`` calls and tests;
        production decryption is routed through the ``CryptoMWorker`` pool.
        """
        from ..core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend

        self.bfv_backend = BFVPrivSelectV2Backend(
            n_entries=self.config["vocab_size"],
            vec_dim=self.config["hidden_dim"],
            shared_seed=os.urandom(32),  # Required by backend __init__
        )
        # Re-attach sk_M (the parent dropped it before forking).
        # sk_pem is raw SEAL bytes, pk_pem may be pickle or raw bytes.
        self.bfv_backend._secret_key = self.bfv_backend._load_secret_key(bfv_sk_pem)
        # Handle both raw bytes and pickle format for pk
        import pickle as _pickle
        try:
            pk_data = _pickle.loads(bfv_pk_pem)
            pk_bytes = pk_data["pk_bytes"]
        except Exception:
            pk_bytes = bfv_pk_pem
        self.bfv_backend._public_key = self.bfv_backend.reconstruct_public_key(pk_bytes)
        from seal import Decryptor
        self.bfv_backend._decryptor = Decryptor(self.bfv_backend._context, self.bfv_backend._secret_key)
        logger.info("PartyM BFV secret key loaded")

    def _setup_optimizer(self) -> None:
        import torch.optim as optim
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 3.5e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        self.gradient_clip_norm = self.config.get("gradient_clip_norm", 1.0)
        self.warmup_steps = max(0, int(self.config.get("warmup_steps", 200)))
        self.lr_scheduler_kind = self.config.get("lr_scheduler", "cosine_with_warmup")
        self._build_lr_scheduler()

        # DeepSpeed ZeRO integration for optimizer state partitioning
        use_deepspeed = bool(self.config.get("use_deepspeed_zero", False))
        if use_deepspeed:
            self._setup_deepspeed_zero()

    def _build_lr_scheduler(self) -> None:
        import math
        import torch.optim.lr_scheduler as ls

        peak_lr = self.config.get("learning_rate", 3.5e-4)
        warmup = self.warmup_steps

        if self.lr_scheduler_kind == "none":
            self.lr_scheduler = None
            return

        try:
            n_train = int(self.config.get("n_train_samples", 0))
        except Exception:
            n_train = 0
        bs = max(1, int(self.config.get("batch_size", 48)))
        max_epochs = max(1, int(self.config.get("max_epochs", 10)))
        steps_per_epoch = max(1, math.ceil(max(n_train, 0) / bs)) if n_train else 600
        total_steps = max(warmup + 1, max_epochs * steps_per_epoch)

        if self.lr_scheduler_kind == "linear":
            self.lr_scheduler = ls.LambdaLR(
                self.optimizer,
                lr_lambda=lambda s: min((s + 1) / warmup, 1.0) if warmup else 1.0,
            )
        elif self.lr_scheduler_kind == "cosine":
            self.lr_scheduler = ls.CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup
            )
        else:
            def lr_lambda(s: int) -> float:
                if warmup and s < warmup:
                    return (s + 1) / warmup
                t = (s - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
            self.lr_scheduler = ls.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logger.info(
            "PartyM LR scheduler: %s, warmup=%d, total=%d, peak_lr=%.2e",
            self.lr_scheduler_kind, warmup, total_steps, peak_lr,
        )

    def _setup_deepspeed_zero(self) -> None:
        """Initialize DeepSpeed ZeRO for optimizer state partitioning.

        ZeRO-1 partitions optimizer states across GPUs, reducing memory by ~4x.
        For single-GPU training, this partitions across gradient accumulation steps.
        """
        try:
            from ..utils.deepspeed_zero import (
                DeepSpeedZeROManager,
                create_ds_config,
                create_zero_config,
                create_optimizer_config,
                is_deepspeed_available,
            )
        except ImportError:
            logger.warning("DeepSpeed ZeRO module not found; skipping ZeRO init")
            self._ds_manager = None
            return

        if not is_deepspeed_available():
            logger.warning("DeepSpeed not installed; install with: pip install deepspeed")
            self._ds_manager = None
            return

        zero_stage = int(self.config.get("zero_stage", 1))
        bf16_enabled = True  # BF16 is more stable than FP16
        gradient_clipping = float(self.config.get("gradient_clip_norm", 1.0))
        learning_rate = float(self.config.get("learning_rate", 1e-4))
        weight_decay = float(self.config.get("weight_decay", 0.01))

        # Build DeepSpeed config
        zero_config = create_zero_config(stage=zero_stage)
        optimizer_config = create_optimizer_config(
            optimizer="adamw",
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        ds_config = create_ds_config(
            fp16_enabled=False,
            bf16_enabled=bf16_enabled,
            zero_stage=zero_stage,
            gradient_clipping=gradient_clipping,
            zero_config=zero_config,
            optimizer_config=optimizer_config,
        )

        # Build param groups with DeepSpeed
        param_groups = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
        ]

        import deepspeed
        import torch.distributed as dist

        # Initialize distributed if needed
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl")
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
            except RuntimeError as e:
                logger.warning(f"Distributed init failed: {e}; running without ZeRO")
                self._ds_manager = None
                return

        try:
            self._ds_manager = DeepSpeedZeROManager(
                model=self.model,
                optimizer=self.optimizer,
                zero_stage=zero_stage,
                bf16_enabled=bf16_enabled,
                gradient_clipping=gradient_clipping,
                config_params={"optimizer": self.optimizer},
            )

            if self._ds_manager._initialized:
                # Replace optimizer and model with DeepSpeed-wrapped versions
                self.optimizer = self._ds_manager.optimizer
                logger.info(
                    f"DeepSpeed ZeRO-{zero_stage} initialized successfully: "
                    f"bf16={bf16_enabled}, grad_clip={gradient_clipping}"
                )
            else:
                self._ds_manager = None
                logger.warning("DeepSpeed initialization returned uninitialized engine; continuing without ZeRO")
        except Exception as e:
            logger.warning(f"DeepSpeed ZeRO setup failed: {e}; continuing without ZeRO")
            self._ds_manager = None

    # ------------------------------------------------------------------------- #
    #  Forward — keeps H_M on GPU
    # ------------------------------------------------------------------------- #
    def forward(
        self,
        H_U: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute H_M = decoder[16..32)(H_U) on GPU and cache for backward.

        Returns H_M as a **GPU tensor** (heterogeneous protocol does in-process
        hand-off to PartyS without CPU copy).
        """
        H_U = H_U.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        H_M = self._m_forward(H_U, attention_mask=attention_mask)
        # Cache refs for backward+update. We intentionally keep the
        # non-detached tensor here so ``H_M.backward(g_H)`` can traverse the
        # graph; the activations themselves are minimized by the per-layer
        # reentrant checkpointing in ``_m_forward``. F-step (post-backward)
        # clears these caches immediately after ``optimizer.step()``.
        self._last_H_U = H_U
        self._last_H_M = H_M
        self._last_attention_mask = attention_mask
        return {"H_M": H_M}

    def _m_forward(
        self,
        H_U: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """M's forward: ALL decoder layers + norm with per-layer checkpoint.

        Under the new split (u_split_layer=0), M runs the entire decoder;
        ``H_U`` is the *embedding output* coming from U's embed_tokens lookup.

        Reentrant activation checkpointing (use_reentrant=True) trades a
        small amount of recompute for ~50% lower activation memory across
        the 32-layer stack. Set "full" style for maximum memory savings.

        Gradient flow (fix, v2.1):
            ``H_U`` is intentionally **not** detached here. Detaching would
            strip ``requires_grad`` from the checkpoint input, causing
            ``torch.utils.checkpoint`` to silently zero out every gradient
            inside the reentrant segment (PyTorch emits
            ``UserWarning: None of the inputs have requires_grad=True.
            Gradients will be None``). The result is LoRA params with ``None``
            gradients and AdamW stepping with a zero-magnitude update.

            U's ``embed_tokens.weight`` remains frozen (``requires_grad=False``
            set in ``model_splitting.py``), so the autograd graph flows
            *through* U's embedding into M's decoder layers without producing
            any embedding update — exactly what we want, with the side
            benefit of restoring LoRA gradients.

            Privacy: this does NOT leak ``sk_M`` or ``V`` to U. U's optimizer
            is disabled (see ``PartyU._setup_optimizer``), so the gradient
            that lands on ``embed_tokens.weight`` is silently ignored. The
            private forward direction (U → M) is unchanged: H_U travels as a
            plain GPU tensor, not a ciphertext; M never sees ``V`` or ``x``;
            S never sees ``sk_M``.
        """
        # Use the shard's forward method with gradient checkpointing
        # Reentrant checkpointing (document §3.2.1) saves ~50% activation memory
        # with the cost of one forward recompute per backward pass.
        # "full" style recomputes everything for maximum memory savings.
        import torch.utils.checkpoint as ckpt

        ckpt_style = self.config.get("gradient_checkpointing_style", "reentrant")
        use_reentrant = (ckpt_style == "reentrant")

        # For SDPA mode: pass hidden states only, model computes rotary internally
        def _forward_with_ckpt(hidden):
            return self.model.forward(hidden)

        return ckpt.checkpoint(
            _forward_with_ckpt,
            H_U,
            use_reentrant=use_reentrant,
        )

    # ------------------------------------------------------------------------- #
    #  Backward + LoRA update (CPU decrypt via CryptoMWorker + GPU autograd)
    # ------------------------------------------------------------------------- #
    def backward_and_update(self, payload: Dict) -> Dict:
        """Decrypt U's masked ciphertexts (CPU) + LoRA update (GPU).

        Two phases:
          1. **CPU**: forward ct_list + s_share_list to ``CryptoMWorker`` pool,
             which decrypts each ciphertext to ``-V_y + R_t``. The driver then
             plaintext-adds ``s_share = a_t - R_t`` to get ``a_t - V_y``.
          2. **GPU**: reshape the gradient tensor to (B, S, hidden_dim), inject
             into the cached H_M autograd graph, run backward + LoRA step.
        """
        import numpy as np
        import torch

        ct_list = payload.get("ct_from_U") or []
        s_share_list = payload.get("s_share") or []
        step = payload.get("step", 0)
        expected_shape = payload.get("expected_shape")

        if not ct_list:
            raise RuntimeError(f"step {step}: design 2 requires ct_from_U list")
        if not s_share_list:
            raise RuntimeError(f"step {step}: design 2 requires s_share list from S")

        n_pir = len(ct_list)
        vec_dim = self.bfv_backend.vec_dim
        scale = self.bfv_backend.scale
        valid_mask = payload.get("valid_mask")
        valid_indices = payload.get("valid_indices")

        # ---------- Phase 1: CPU decrypt (via CryptoMWorker) ----------
        if self.crypto_m_pool is None:
            raise RuntimeError(
                "CryptoMWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_m_pool() after init."
            )
        decrypt_result = self.crypto_m_pool.submit({
            "ct_list": ct_list,
            "scale": scale,
            "vec_dim": vec_dim,
        })
        masked_arr = decrypt_result["decrypted"]  # (n_pir, vec_dim) float32

        # Plaintext-add s_share: (−V_y·scale + r_t) + (scale·a_t − r_t) = scale·(a_t − V_y).
        # The decrypted masked_arr comes back as floats in [0, pm/scale) because
        # SEAL's BatchEncoder.decode returns values in [0, pm) and the worker
        # divides by scale. We have to *re-multiply by scale*, lift to int64,
        # then centre into [−pm/2, +pm/2) before adding the signed s_share —
        # otherwise the positive-only ciphertext encoding can never cancel the
        # signed PRG mask, and r_t leaks into g_accum as a constant offset
        # (~ pm/scale on every half-token).
        # Determine the full (B, S) token grid so decrypted answer-position
        # gradients can be placed back at their original row/column.
        if expected_shape is not None:
            B, S = expected_shape
        else:
            H_M = getattr(self, "_last_H_M", None)
            B, S = (H_M.shape[0], H_M.shape[1]) if H_M is not None else (n_pir, 1)
        total_tokens = B * S

        plain_modulus = 1 << int(self.bfv_backend.plain_bits)
        half_pm = plain_modulus // 2
        g_accum = np.zeros((total_tokens, vec_dim), dtype=np.float32)

        # Figure out where the decrypted answer gradients land. With
        # answer-token-only PIR, ``valid_indices`` gives the flat positions
        # (one per gold token); legacy dense callers omit it and we fill the
        # first ``n_pir`` rows (later zeroed by ``valid_mask``).
        if valid_indices is not None and len(valid_indices) == n_pir:
            rows = [int(i) for i in valid_indices]
            if rows and (max(rows) >= total_tokens or min(rows) < 0):
                logger.warning(
                    "step %d: valid_indices out of range [0,%d): %s — dense fallback",
                    step, total_tokens, rows,
                )
                rows = list(range(min(n_pir, total_tokens)))
        else:
            rows = list(range(min(n_pir, total_tokens)))

        for local_i, row in enumerate(rows):
            # ``s_share_list`` is aligned with the masked ciphertexts (one
            # share per valid token), so we index by local position.
            s_share = s_share_list[local_i]
            s_arr = np.asarray(s_share[:vec_dim], dtype=np.int64)
            if s_arr.size < vec_dim:
                s_arr = np.pad(s_arr, (0, vec_dim - s_arr.size))

            masked_int = np.round(masked_arr[local_i] * scale).astype(np.int64)
            # Correct modular reconstruction. The ciphertext decrypts to
            # (-V_y*scale + r_t) mod pm as an *unsigned* value in [0, pm).
            # Add the signed s_share (a_t*scale - r_t) FIRST, reduce mod pm,
            # then centre into [-pm/2, +pm/2). Centring before the addition
            # is wrong: whenever the masked sum wraps around pm (which
            # happens for roughly half the slots), it leaves a ±pm offset in
            # the gradient — the source of the occasional huge loss values
            # (210/296) observed in the first smoke run.
            diff_mod = (masked_int + s_arr) % plain_modulus
            diff_centered = np.where(
                diff_mod > half_pm, diff_mod - plain_modulus, diff_mod
            )
            g_accum[row] = diff_centered.astype(np.float32) / scale

        # Legacy dense path: zero the gradient at non-gold positions
        # (prompt/pad, label=-100) using the per-position valid mask.
        if valid_indices is None and valid_mask is not None:
            vm = np.asarray(valid_mask, dtype=bool)
            if vm.size != total_tokens:
                logger.warning(
                    "step %d: valid_mask len %d != total_tokens %d; ignoring mask",
                    step, vm.size, total_tokens,
                )
            else:
                g_accum = np.where(vm[:, None], g_accum, np.float32(0.0))

        # ---------- Phase 2: GPU autograd injection + LoRA step ----------
        if B * S != total_tokens:
            logger.warning(
                "step %d: B*S=%d != total_tokens=%d — reshaping mismatch, using flat",
                step, B * S, total_tokens,
            )
            B, S = total_tokens, 1

        # Reshape per-token gradients to (B, S, vec_dim) — each token gets
        # its own upstream gradient ``a_t - V_y``. The protocol contract
        # specifies a per-token gradient; averaging across tokens would
        # destroy signal and break training stability.
        # Use bfloat16 to match model dtype (document §4.4)
        g_H = torch.from_numpy(g_accum[: B * S]).float().to(self.device).bfloat16()
        g_H = g_H.view(B, S, vec_dim).contiguous()

        loss_proxy = self._inject_and_backward(g_H, step)

        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.gradient_clip_norm,
        )

        if loss_proxy is not None:
            self.optimizer.step()
            if getattr(self, "lr_scheduler", None) is not None:
                self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # F-step: drop the cached forward activations once backward+update
        # completes. These tensors otherwise keep the autograd graph (and the
        # underlying activations for non-checkpointed segments) pinned to GPU
        # until the next forward overwrites them.
        self._last_H_U = None
        self._last_H_M = None
        self._last_attention_mask = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        attack_dumps = {}
        if self.config.get("dump_attack_intermediates", False):
            attack_dumps = self._dump_attack_intermediates(step, g_H)

        return {
            "loss": float(loss_proxy) if loss_proxy is not None else 0.0,
            "gpu_mem_mb": gpu_mem_mb,
            "attack_dumps": attack_dumps,
            "mode": "heterogeneous",
        }

    # Dispatcher alias kept for API uniformity.
    def backward_and_update_dispatch(self, payload: Dict) -> Dict:
        return self.backward_and_update(payload)

    def _inject_and_backward(self, g_H: torch.Tensor, step: int) -> Optional[float]:
        """Seed H_M's upstream gradient with the protocol gradient."""
        if g_H.shape[-1] != self._hidden_dim:
            raise ValueError(
                f"g_H last dim {g_H.shape[-1]} ≠ model hidden_dim {self._hidden_dim}"
            )

        H_M = getattr(self, "_last_H_M", None)
        if H_M is None or H_M.device != g_H.device or H_M.shape != g_H.shape:
            logger.warning(
                "step %d: H_M cache miss (H_M=%s, g_H.device=%s, shape=%s vs %s) — "
                "recomputing H_M via cached H_U", step,
                None if H_M is None else H_M.device, g_H.device,
                None if H_M is None else tuple(H_M.shape), tuple(g_H.shape),
            )
            H_U_cached = getattr(self, "_last_H_U", None)
            if H_U_cached is None:
                logger.error("step %d: no cached H_U; cannot inject gradient", step)
                return None
            H_M = self._m_forward(H_U_cached.to(g_H.device), attention_mask=getattr(self, "_last_attention_mask", None))

        if not H_M.requires_grad:
            H_M.requires_grad_(True)
        H_M.backward(gradient=g_H)

        loss_proxy = float(g_H.detach().norm() / max(1, g_H.shape[0] * g_H.shape[1]))
        return loss_proxy

    def _dump_attack_intermediates(self, step: int, g_H: torch.Tensor) -> Dict:
        import os
        import json

        dump_dir = self.config.get("attack_dump_dir", "/tmp/attack_dumps")
        os.makedirs(dump_dir, exist_ok=True)

        info = {
            "step": step,
            "g_H_shape": list(g_H.shape),
            "g_H_l2_norm": float(g_H.norm().cpu()),
            "g_H_mean": float(g_H.mean().cpu()),
            "g_H_std": float(g_H.std().cpu()),
        }
        try:
            with open(os.path.join(dump_dir, f"step_{step:05d}.json"), "w") as f:
                json.dump(info, f, indent=2)
        except Exception:
            pass
        return info

    def save_checkpoint(self) -> Dict[str, Any]:
        """Return LoRA + optimizer state (only M has trainable params)."""
        lora_state = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                lora_state[name] = p.detach().cpu().clone()
        optimizer_state = {}
        try:
            optimizer_state = {
                k: v for k, v in self.optimizer.state_dict().items()
            }
        except Exception:
            pass
        scheduler_state = {}
        if getattr(self, "lr_scheduler", None) is not None:
            try:
                scheduler_state = self.lr_scheduler.state_dict()
            except Exception:
                pass
        return {
            "party": "M",
            "lora_state": lora_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
        }
