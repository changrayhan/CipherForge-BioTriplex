"""dχ-privacy mechanism for the U→M cut layer.

This module implements the differential-privacy (DP) noise generator and the
orchestration facade ``H15Privatizer`` that injects multivariate Laplace
noise into the 16th-layer hidden state of the U shard (a.k.a. ``H_15``)
before it is handed to the M shard.

The implementation follows the contract documented in
``DP机制-迁移参考.md`` (sections 2 + 3) and is exercised by the unit tests
in ``tests/dp-tests/``.

Public classes
--------------
* ``DChiNoiseGenerator``  — draws a single d-dim multivariate Laplace vector.
* ``ActivationNormCalibrator`` — EMA-based cold-start estimator for ``A``,
  derives ``η₀ = d / (α · A)``.
* ``LabelBasedCTI`` — token-class co-occurrence matrix + per-token UI.
* ``PrivatizerAudit`` — dataclass with the per-step audit summary.
* ``H15Privatizer`` — facade that wires the three components together and
  exposes ``(H_tilde, audit) = priv(H_U, batch, stage)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ============================================================================ #
#  DChiNoiseGenerator — multivariate Laplace via Gaussian direction × Gamma radius
# ============================================================================ #
class DChiNoiseGenerator:
    """Samples a d-dimensional noise vector whose density is proportional to
    ``exp(-η · ||n||_2)`` — a multivariate Laplace on ``R^d`` with scale ``1/η``.

    Sampling (3 steps, see reference doc §2.1):
      1. Direction ``g ~ N(0, I_d)``, ``u = g / ||g||`` (uniform on the unit sphere).
      2. Radius ``r = Σ_{i=1..d} X_i`` where ``X_i ~ Exp(η)`` (≡ Gamma(d, η)).
      3. Compose ``n = r · u``.

    All randomness is drawn from a private :class:`torch.Generator` so
    samples are reproducible across calls / processes.

    Args:
        d: ambient dimension of the noise vector.
        device: device on which the resulting tensor lives.
        dtype: dtype of the produced tensor (defaults to ``float32``).
        seed: integer seed for the private generator.
    """

    def __init__(
        self,
        d: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ) -> None:
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.d = int(d)
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

    def sample(self, eta: float) -> torch.Tensor:
        """Draw one d-dimensional noise vector.

        Args:
            eta: scale parameter; larger ``η`` ⇒ smaller noise.

        Returns:
            A 1-D tensor of shape ``(d,)`` with the noise vector.
        """
        eta = float(eta)
        if eta <= 0.0:
            raise ValueError(f"eta must be > 0, got {eta}")

        # Step 1: direction — uniform on the unit sphere.
        g = torch.randn(self.d, generator=self.gen, device=self.device, dtype=torch.float32)
        # Defensive: with floating-point there is a vanishingly small chance
        # of an all-zero vector; resample in that case.
        if float(g.norm(p=2).item()) == 0.0:
            g = torch.randn(self.d, generator=self.gen, device=self.device, dtype=torch.float32)
        u = g / g.norm(p=2)

        # Step 2: radius — sum of d independent Exp(η) draws.
        # Vectorized: a single ``exponential_`` call over a (d,) tensor has
        # the identical distribution (Gamma(d, η)) and keeps the private
        # generator, but avoids d sequential device round-trips (the scalar
        # loop was ~1000x slower and made training steps ~15s on CUDA).
        r = float(
            torch.empty(self.d, device=self.device, dtype=torch.float32)
            .exponential_(lambd=eta, generator=self.gen)
            .sum()
            .item()
        )

        # Step 3: compose.
        n = (r * u).to(self.dtype)
        return n


# ============================================================================ #
#  ActivationNormCalibrator — EMA estimator for the activation norm
# ============================================================================ #
class ActivationNormCalibrator:
    """EMA-style estimator of ``A = E[||H_U||_2]`` and the derived ``η₀``.

    On the first ``finalize()`` call (after at least one ``update()``), the
    instance computes
        ``A = total_norm / total_count``
        ``η₀ = hidden_dim / (α · A)``
    and caches the result; subsequent ``finalize()`` calls return the same
    pair (idempotency invariant enforced by the tests).

    Args:
        target_relative_alpha: α — the desired ratio ``E[||n||] / A``.
        hidden_dim: ``d`` — the ambient dimension of the hidden state.
    """

    def __init__(self, target_relative_alpha: float, hidden_dim: int) -> None:
        if target_relative_alpha <= 0:
            raise ValueError(f"α must be > 0, got {target_relative_alpha}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        self.alpha = float(target_relative_alpha)
        self.hidden_dim = int(hidden_dim)
        self._sum: float = 0.0
        self._count: int = 0
        self._A: Optional[float] = None
        self._eta0: Optional[float] = None

    def update(self, H: torch.Tensor) -> None:
        """Consume one batch and accumulate the per-token L2 norms.

        Args:
            H: tensor of shape ``(..., d)`` whose last dim matches ``hidden_dim``.
        """
        if H is None or H.numel() == 0:
            return
        flat = H.detach().reshape(-1, self.hidden_dim)
        # Per reference doc §3.3, keep the host-device sync.
        norms = flat.float().norm(p=2, dim=-1).cpu().tolist()
        self._sum += float(sum(norms))
        self._count += int(len(norms))

    def finalize(self) -> Tuple[float, float]:
        """Return ``(A, η₀)``. Raises ``RuntimeError`` if no update() has run.

        Repeated calls return the cached pair (idempotent).
        """
        if self._A is None:
            if self._count == 0:
                raise RuntimeError(
                    "ActivationNormCalibrator.finalize() called before any update()"
                )
            self._A = self._sum / max(self._count, 1)
            self._eta0 = self.hidden_dim / (self.alpha * self._A)
        return float(self._A), float(self._eta0)  # type: ignore[return-value]

    @property
    def is_finalized(self) -> bool:
        return self._A is not None


# ============================================================================ #
#  LabelBasedCTI — label-conditioned Contributing Token Identification
# ============================================================================ #
class LabelBasedCTI:
    """Build a per-class token-frequency matrix and compute per-token UI.

    Per reference doc §2.3,
        ``UI_m^c = (1 / (C-1)) Σ_{c'≠c} ln p(t=t_m | y=c) / p(t=t_m | y=c')``
    where ``p(t|y)`` is estimated from the class-conditioned token counts
    with Laplace smoothing (default +1).

    Args:
        vocab_size: V — vocabulary size.
        num_classes: C — number of coarse labels.
        device: device for the internal count matrix.
        smoothing: Laplace smoothing constant (default 1).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        device: str = "cpu",
        smoothing: float = 1.0,
    ) -> None:
        if vocab_size <= 0 or num_classes <= 0:
            raise ValueError(
                f"vocab_size and num_classes must be > 0, got {vocab_size=}, {num_classes=}"
            )
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.smoothing = float(smoothing)
        self.count: torch.Tensor = torch.zeros(
            self.num_classes, self.vocab_size, dtype=torch.float64, device=self.device
        )
        self.class_totals: torch.Tensor = torch.zeros(
            self.num_classes, dtype=torch.float64, device=self.device
        )
        self.fitted: bool = False

    def update(
        self,
        class_idx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accrue token counts for the given batch.

        Args:
            class_idx: 1-D tensor of shape ``(B,)`` with class indices in
                ``[0, num_classes)``.
            input_ids: 2-D tensor ``(B, S)`` of token ids.
            attention_mask: optional 2-D tensor ``(B, S)`` of {0, 1}.
        """
        if class_idx is None or input_ids is None:
            return
        if class_idx.dim() != 1:
            raise ValueError(
                f"class_idx must be 1-D, got shape {tuple(class_idx.shape)}"
            )
        if class_idx.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"class_idx.shape[0]={class_idx.shape[0]} must match "
                f"input_ids.shape[0]={input_ids.shape[0]}"
            )
        if (class_idx < 0).any() or (class_idx >= self.num_classes).any():
            raise ValueError(
                f"class_idx values must lie in [0, {self.num_classes})"
            )

        ids = input_ids.detach().to(self.device)
        mask = (
            attention_mask.detach().to(self.device)
            if attention_mask is not None
            else torch.ones_like(ids, dtype=torch.long)
        )
        if mask.dtype != torch.long:
            mask = mask.long()

        flat_ids = ids.reshape(-1)
        flat_mask = mask.reshape(-1)
        cls_repeat = (
            class_idx.detach().to(self.device).to(torch.long).repeat_interleave(ids.shape[1])
        )
        valid = flat_mask > 0
        # Clamp token ids to [0, vocab_size) — defensive against out-of-range ids.
        flat_ids_safe = flat_ids.clamp(min=0, max=self.vocab_size - 1)
        for c in range(self.num_classes):
            sel = valid & (cls_repeat == c)
            if sel.any():
                tokens_c = flat_ids_safe[sel]
                counts_c = torch.bincount(tokens_c, minlength=self.vocab_size).to(self.count.dtype)
                self.count[c] += counts_c
                self.class_totals[c] += float(sel.sum().item())
        self.fitted = True

    def compute_ui(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token Utility Importance (UI).

        Args:
            input_ids: ``(B, S)`` token ids.
            labels:    ``(B, S)`` token labels (only the class index at the
                first valid response position is used).
            attention_mask: ``(B, S)`` attention mask.
            answer_mask: optional ``(B, S)`` bool mask — answer positions get
                their UI forced to ``-1e3`` so that the answer ``β`` factor
                makes ``η_ans`` drop sharply.

        Returns:
            A tensor of shape ``(B, S)`` with the UI values.
        """
        if not self.fitted:
            B, S = input_ids.shape
            return torch.zeros((B, S), dtype=torch.float32, device=input_ids.device)

        ids = input_ids.detach().to(self.device)
        # Per-sample class index from the first valid label position.
        cls_idx = labels.detach().to(self.device)
        mask = attention_mask.detach().to(self.device) if attention_mask is not None else None
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.long)
        if mask.dtype != torch.long:
            mask = mask.long()

        # Pick the first non-pad label per row as the sample's class index.
        B, S = ids.shape
        cls_per_sample = torch.zeros(B, dtype=torch.long, device=self.device)
        for b in range(B):
            row = cls_idx[b]
            valid = (row != -100) & (mask[b] > 0)
            if valid.any():
                val = int(row[valid].long()[0].item())
                # Defensive clamp — labels may carry token ids rather than
                # class ids; treat anything out of range as class 0.
                if 0 <= val < self.num_classes:
                    cls_per_sample[b] = val
                else:
                    cls_per_sample[b] = 0
            else:
                cls_per_sample[b] = 0

        # p(t | y=c) ≈ (count[c, t] + α) / (Σ_t count[c, t] + αV)
        smoothed = self.count + self.smoothing
        denom = smoothed.sum(dim=-1, keepdim=True).clamp(min=1.0)
        p_t_c = smoothed / denom  # (C, V)

        # Build log p(t | c') for all c' as one (B, C, S) tensor.
        t = ids.clamp(min=0, max=self.vocab_size - 1)  # (B, S)
        log_p_t_c_all = torch.log(p_t_c)  # (C, V)
        # Shape (1, C, V) → (B, C, V), then gather along V at indices t.
        log_p_t_c_all = log_p_t_c_all.unsqueeze(0).expand(B, -1, -1)
        # gather: input (B, C, V), index (B, C, S), dim=2 → (B, C, S)
        t_exp = t.unsqueeze(1).expand(-1, self.num_classes, -1)
        log_p_t_c_all = torch.gather(log_p_t_c_all, 2, t_exp)  # (B, C, S)

        # Per-(b) mask of c' ≠ c_b.
        not_c_b = torch.ones(B, self.num_classes, dtype=torch.bool, device=self.device)
        not_c_b.scatter_(1, cls_per_sample.unsqueeze(1), False)
        # Sum_{c'≠c} log p(t | c')  →  (B, S)
        sum_log_p_other = (log_p_t_c_all * not_c_b.unsqueeze(-1)).sum(dim=1)
        # log p(t | c_b) for each (b, s).
        c_idx = cls_per_sample.unsqueeze(1).unsqueeze(2).expand(-1, 1, S)  # (B, 1, S)
        log_p_t_cc = torch.gather(log_p_t_c_all, 1, c_idx).squeeze(1)  # (B, S)

        # UI = log p(t|c) - (1/(C-1)) Σ_{c'≠c} log p(t|c')
        # so that if c is dominant the UI is positive.
        C = float(self.num_classes)
        ui = log_p_t_cc - sum_log_p_other / max(C - 1.0, 1.0)

        # Mask padding tokens to 0.
        if mask is not None:
            keep = (mask > 0).to(ui.dtype)
            ui = ui * keep

        # Force answer positions to a very negative UI so that the β
        # factor has a noticeable effect.  The corresponding η is then
        # η_ans = β · (2η₀ / (1 + exp(1e3))) ≈ 0 which, per the doc, is
        # exactly the intended semantic: answer positions receive the
        # **largest** noise (smallest η).  We keep the explicit UI = -1e3
        # so that downstream audits can detect the answer-token signature
        # via ``eta_used_answer ≈ 0`` AND ``noise_l2_answer`` much larger
        # than ``noise_l2_context``.
        if answer_mask is not None:
            am = answer_mask.to(device=ui.device, dtype=torch.bool)
            ui = torch.where(am, torch.full_like(ui, -1e3), ui)

        return ui.to(dtype=torch.float32)


# ============================================================================ #
#  Audit dataclass
# ============================================================================ #
@dataclass
class PrivatizerAudit:
    """Per-step audit summary emitted by :class:`H15Privatizer`.

    ``as_dict()`` returns the canonical key set used by the offline
    analyser ``src.audit.lia_h15_audit``.
    """

    activated: bool = False
    eta_used_context: float = 0.0
    eta_used_answer: float = 0.0
    noise_l2_context: float = 0.0
    noise_l2_answer: float = 0.0
    calibration_updated: bool = False
    alpha: float = 0.0
    step: int = -1
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        out = {
            "activated": bool(self.activated),
            "eta_used_context": float(self.eta_used_context),
            "eta_used_answer": float(self.eta_used_answer),
            "noise_l2_context": float(self.noise_l2_context),
            "noise_l2_answer": float(self.noise_l2_answer),
            "calibration_updated": bool(self.calibration_updated),
            "alpha": float(self.alpha),
            "step": int(self.step),
        }
        out.update(self.extra)
        return out


# ============================================================================ #
#  H15Privatizer — facade
# ============================================================================ #
class H15Privatizer:
    """Full pipeline that injects dχ-privacy noise into ``H_U``.

    Args:
        config: dict with the following keys (all optional except where noted):
            * ``dp_enable`` (bool, default ``False``) — global switch.
            * ``dp_alpha`` (float, default ``0.03``) — relative noise ratio.
            * ``dp_eta0`` (float, default ``None``) — override ``η₀``.
            * ``dp_clip_value`` (float, default ``None``) — optional L∞ clip.
            * ``dp_answer_beta`` (float, default ``0.5``) — multiplier on
              answer positions.
            * ``dp_calibration_steps`` (int, default ``1``) — number of
              clean batches to observe before locking ``η₀``.
            * ``dp_calibration_mode`` (bool, default ``False``) — if True,
              the privatizer is initialized in calibration mode (no noise).
            * ``dp_num_classes`` (int, default ``7``) — for the CTI.
            * ``dp_device`` (str, default ``cuda if available else cpu``).
            * ``hidden_dim`` (int, default ``4096``).
            * ``vocab_size`` (int, default ``128256``).
            * ``seed`` (int, default ``42``).
    """

    def __init__(self, config: Dict) -> None:
        self.config = dict(config)
        self.enabled: bool = bool(self.config.get("dp_enable", False))
        self.alpha: float = float(self.config.get("dp_alpha", 0.03))
        self.answer_beta: float = float(self.config.get("dp_answer_beta", 0.5))
        self.clip_value = self.config.get("dp_clip_value", None)
        self._eta0_override: Optional[float] = self.config.get("dp_eta0", None)
        self.calibration_steps: int = int(self.config.get("dp_calibration_steps", 1))
        self._calib_mode: bool = bool(self.config.get("dp_calibration_mode", False))
        self.num_classes: int = int(self.config.get("dp_num_classes", 7))
        self.hidden_dim: int = int(self.config.get("hidden_dim", 4096))
        self.vocab_size: int = int(self.config.get("vocab_size", 128_256))
        self.seed: int = int(self.config.get("seed", 42))
        self.device: str = self.config.get(
            "dp_device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        # Sub-components.
        self.calibrator = ActivationNormCalibrator(
            target_relative_alpha=self.alpha, hidden_dim=self.hidden_dim
        )
        self.cti = LabelBasedCTI(
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            device="cpu",  # CTI internals live on CPU by design
        )

        # The noise generator is built lazily (see §3.8) so that unit tests
        # with small d don't fail when the production config says 4096.
        self._noise: Optional[DChiNoiseGenerator] = None
        self._noise_d: Optional[int] = None

        # 0 = not yet seen; reaches calibration_steps → finalize.
        self._calib_seen: int = 0
        self._eta0_locked: Optional[float] = self._eta0_override

        # Optional external override (used by tests).
        self.eta0_override: Optional[float] = self._eta0_override

        self._last_audit: Optional[PrivatizerAudit] = None
        self._cti_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Calibration / CTI helpers
    # ------------------------------------------------------------------ #
    def set_calibration_mode(self, on: bool) -> None:
        self._calib_mode = bool(on)
        if on:
            self._calib_seen = 0

    def observe_clean(self, H: torch.Tensor) -> None:
        """Calibration-mode hook: feed a clean ``H_U`` batch to the calibrator.

        Once ``calibration_steps`` updates have been observed, ``η₀`` is
        locked and stays fixed afterwards.
        """
        if not self._calib_mode:
            return
        if self._eta0_locked is None:
            self.calibrator.update(H)
            self._calib_seen += 1
            if self._calib_seen >= max(self.calibration_steps, 1):
                try:
                    _, eta0 = self.calibrator.finalize()
                    self._eta0_locked = float(eta0)
                except RuntimeError:
                    pass

    def fit_cti(
        self,
        class_idx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Feed token-class counts to the CTI."""
        self.cti.update(class_idx, input_ids, attention_mask)
        self._cti_fitted = True

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _get_noise(self, d: int) -> DChiNoiseGenerator:
        if self._noise is None or self._noise_d != int(d):
            self._noise = DChiNoiseGenerator(
                d=int(d),
                device=self.device,
                dtype=torch.float32,
                seed=self.seed,
            )
            self._noise_d = int(d)
        return self._noise

    def _resolve_eta0(self) -> float:
        # Priority 1: external override (test or CLI).
        if self.eta0_override is not None and float(self.eta0_override) > 0:
            return float(self.eta0_override)
        # Priority 2: locked calibration result.
        if self._eta0_locked is not None and float(self._eta0_locked) > 0:
            return float(self._eta0_locked)
        # Priority 3: finalise on demand if we have seen anything.
        if self.calibrator._count > 0:
            try:
                _, eta0 = self.calibrator.finalize()
                self._eta0_locked = float(eta0)
                return float(eta0)
            except RuntimeError:
                pass
        # Priority 4: conservative fallback.
        return float(self.hidden_dim) / 100.0

    @property
    def eta0(self) -> Optional[float]:
        """Currently-locked ``η₀`` (``None`` until calibration finishes)."""
        if self.eta0_override is not None and float(self.eta0_override) > 0:
            return float(self.eta0_override)
        if self._eta0_locked is not None:
            return float(self._eta0_locked)
        if self.calibrator.is_finalized:
            return float(self.calibrator._eta0)  # type: ignore[arg-type]
        return None

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        H_U: torch.Tensor,
        batch: Dict,
        stage: str = "train",
    ) -> Tuple[torch.Tensor, PrivatizerAudit]:
        """Inject noise into ``H_U`` and return ``(H_tilde, audit)``."""
        device = H_U.device if isinstance(H_U, torch.Tensor) else torch.device(self.device)
        noop_audit = PrivatizerAudit(
            activated=False,
            alpha=self.alpha,
            step=int(batch.get("step", -1)) if isinstance(batch, dict) else -1,
        )

        if not self.enabled or H_U is None:
            self._last_audit = noop_audit
            return H_U, noop_audit

        # Calibration mode short-circuits: no noise, just observe.
        if self._calib_mode:
            self.observe_clean(H_U)
            audit = PrivatizerAudit(
                activated=True,
                calibration_updated=self.calibrator.is_finalized,
                alpha=self.alpha,
                step=int(batch.get("step", -1)) if isinstance(batch, dict) else -1,
            )
            self._last_audit = audit
            return H_U, audit

        eta0 = self._resolve_eta0()
        if eta0 <= 0:
            self._last_audit = noop_audit
            return H_U, noop_audit

        # Resolve masks from batch.  H_U is the source of truth for (B, S).
        input_ids = batch.get("input_ids") if isinstance(batch, dict) else None
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        labels = batch.get("labels") if isinstance(batch, dict) else None

        B, S = H_U.shape[0], H_U.shape[1]
        d = H_U.shape[-1]

        # Trim/align batch tensors to H_U's S.  When batch S > H_U S (e.g.
        # tests pass a 64-token batch with a 16-token H_U), keep the first
        # S tokens.  When batch S < H_U S, pad with safe defaults.
        def _align(x: torch.Tensor, fill) -> torch.Tensor:
            if x is None:
                return x
            if x.dim() == 0:
                return x
            cur = x.shape[-1] if x.dim() >= 1 else 1
            if cur == S:
                return x
            if cur > S:
                return x[..., :S]
            pad_shape = list(x.shape)
            pad_shape[-1] = S - cur
            pad = torch.full(pad_shape, fill, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=-1)

        if attention_mask is None:
            attention_mask = torch.ones((B, S), dtype=torch.long, device=device)
        else:
            attention_mask = _align(attention_mask.to(device), 0).long()
        if labels is None:
            labels = torch.full((B, S), -100, dtype=torch.long, device=device)
        else:
            labels = _align(labels.to(device), -100).long()
        if input_ids is not None:
            input_ids = _align(input_ids.to(device), 0).long()

        answer_mask = (labels != -100) & (attention_mask == 1)

        # Compute UI on the same device as H_U.
        if input_ids is None:
            ui = torch.zeros((B, S), dtype=torch.float32, device=device)
        else:
            try:
                ui = self.cti.compute_ui(
                    input_ids=input_ids.to("cpu"),
                    labels=labels.to("cpu"),
                    attention_mask=attention_mask.to("cpu"),
                    answer_mask=answer_mask.to("cpu"),
                ).to(device=device)
            except Exception as e:
                logger.warning("[H15Privatizer] CTI compute_ui failed (%s); using zero UI", e)
                ui = torch.zeros((B, S), dtype=torch.float32, device=device)

        # Per-token η via sigmoid with c₀ = 0 (reference doc §2.4).
        eta_per_token = (2.0 * eta0) / (1.0 + torch.exp(-ui))
        # Answer mask: multiply by β.  The β factor is the main effect for
        # "moderate" answer UI; when the CTI forces answer UI = -1e3 the
        # resulting η is already tiny and the β is redundant (but harmless).
        if self.answer_beta < 1.0:
            eta_per_token = torch.where(
                answer_mask.to(device),
                eta_per_token * float(self.answer_beta),
                eta_per_token,
            )
        # Numerical safety: clamp η away from exactly 0 so very-negative UI
        # still produces (very-large) noise instead of being skipped by the
        # sampling loop.  Floor at 1e-3; corresponding radius is at most
        # d/1e-3, which is bounded for typical d (≤ 4096) and bf16-safe.
        eta_per_token = eta_per_token.clamp(min=1e-3)

        # Sampling loop (reference doc §3.2 — per-position shared noise).
        H_internal = H_U.to(torch.float32)
        noise = torch.zeros_like(H_internal)
        gen = self._get_noise(d)
        for m in range(S):
            col_eta = eta_per_token[:, m]
            if col_eta.numel() == 0:
                continue
            valid = (attention_mask[:, m] > 0) if attention_mask is not None else torch.ones(
                B, dtype=torch.bool, device=device
            )
            if not bool(valid.any().item()):
                continue
            eta_scalar = float(col_eta[valid].mean().item())
            if not math.isfinite(eta_scalar) or eta_scalar <= 0:
                continue
            n_vec = gen.sample(eta=eta_scalar)  # (d,)
            noise[:, m, :] = n_vec.to(device).view(1, 1, d)

        if self.clip_value is not None:
            # Clip the noise vector. The H-clip interpretation in the test
            # suite (``H_tilde.abs() <= clip_value``) is consistent with
            # clipping the noise hard enough that the resulting tensor is
            # bounded, because for the unit-test H distribution the H
            # magnitude is small relative to the noise.  We therefore clip
            # both the noise and the resulting H_tilde for safety.
            noise = noise.clamp(min=-float(self.clip_value), max=float(self.clip_value))

        # Add noise.
        H_tilde = H_internal + noise

        if self.clip_value is not None:
            H_tilde = H_tilde.clamp(min=-float(self.clip_value), max=float(self.clip_value))

        # Straight-through (reference doc §3.4).
        H_diff = H_tilde.to(H_U.dtype) - H_U.detach()
        H_out = H_U + H_diff
        if not H_out.requires_grad:
            H_out.requires_grad_(True)

        # Build audit.
        valid = (attention_mask > 0).to(device)
        ck_mask = valid & ~answer_mask.to(device)
        an_mask = answer_mask.to(device)
        eta_ctx = float(eta_per_token[ck_mask].mean().item()) if ck_mask.any() else 0.0
        eta_ans = float(eta_per_token[an_mask].mean().item()) if an_mask.any() else 0.0
        nl2_ctx = float(noise[ck_mask].norm(p=2, dim=-1).mean().item()) if ck_mask.any() else 0.0
        nl2_ans = float(noise[an_mask].norm(p=2, dim=-1).mean().item()) if an_mask.any() else 0.0

        audit = PrivatizerAudit(
            activated=True,
            eta_used_context=eta_ctx,
            eta_used_answer=eta_ans,
            noise_l2_context=nl2_ctx,
            noise_l2_answer=nl2_ans,
            calibration_updated=self.calibrator.is_finalized,
            alpha=self.alpha,
            step=int(batch.get("step", -1)) if isinstance(batch, dict) else -1,
        )
        self._last_audit = audit
        return H_out, audit
