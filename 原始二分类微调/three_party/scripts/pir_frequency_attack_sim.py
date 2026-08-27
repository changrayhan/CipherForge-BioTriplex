"""Frequency-attack simulation for block PIR v1 vs v2.

v1: uniform dummies, block=8 (old default).
v2: label-aligned dummies (w = {space:0.5, yes:0.25, no:0.25}), block=64,
    fake query blocks at real:fake = 8:2.

The server-side attacker knows w and that each real block contains exactly one
real label; it does NOT know which blocks are fake.  Per-block guess uses the
exact multinomial likelihood ratio; a global chi-square test measures the
residual statistical signal over many queries.
"""

from __future__ import annotations

import math
import random
from collections import Counter

VOCAB = 32000
LABELS = [29871, 3869, 1939]          # space / yes / no
W = {29871: 0.5, 3869: 0.25, 1939: 0.25}
W_LIST = [(t, W[t]) for t in LABELS]


def sample_block_v1(y: int, block: int = 8) -> list:
    others = [i for i in range(VOCAB) if i != y]
    rows = random.sample(others, block - 1) + [y]
    random.shuffle(rows)
    return rows


def sample_block_v2(y: int, block: int = 64, fake: bool = False) -> list:
    n = block if fake else block - 1
    dummies = random.choices(LABELS, weights=[W[t] for t in LABELS], k=n)
    rows = dummies + ([] if fake else [y])
    random.shuffle(rows)
    return rows


def multinom_logp(counts: Counter, n: int) -> float:
    """log P(counts | n iid draws from w)."""
    if sum(counts.values()) != n:
        return -math.inf
    logp = math.lgamma(n + 1)
    for t in LABELS:
        c = counts.get(t, 0)
        logp -= math.lgamma(c + 1)
        logp += c * math.log(W[t])
    return logp


def best_guess_v2(block: list) -> tuple:
    """Per-block maximum-likelihood guess: which label is the real row,
    and is this block real at all?"""
    c = Counter(block)
    lr_real = {}
    for t in LABELS:
        c_minus = Counter(c)
        c_minus[t] -= 1
        if c_minus[t] < 0:
            lr_real[t] = -math.inf
        else:
            lr_real[t] = multinom_logp(c_minus, len(block) - 1) - multinom_logp(c, len(block))
    best_t = max(lr_real, key=lr_real.get)
    return best_t, lr_real[best_t], max(lr_real.values())


def run(v1: bool, n_real: int, fake_ratio: float, block: int, seed: int = 0):
    rng = random.Random(seed)
    random.seed(seed)
    n_fake = int(round(n_real * fake_ratio))
    ground = []
    blocks = []
    for _ in range(n_real):
        y = rng.choice(LABELS)
        blocks.append(sample_block_v1(y, block) if v1 else sample_block_v2(y, block, False))
        ground.append(y)
    for _ in range(n_fake):
        blocks.append(sample_block_v2(0, block, True) if not v1 else sample_block_v1(0, block))
        ground.append(None)
    # Shuffle so order reveals nothing
    zipped = list(zip(blocks, ground))
    rng.shuffle(zipped)
    blocks, ground = zip(*zipped)

    correct = 0
    guessed_real = 0
    real_scores = []
    fake_scores = []
    for b, g in zip(blocks, ground):
        if v1:
            # v1 attack: the real label is the label token present in the block
            present = [t for t in LABELS if t in b]
            guess = present[0] if len(present) == 1 else (random.choice(present) if present else None)
            lr_max = 1.0
        else:
            guess, _, lr_max = best_guess_v2(b)
        if g is not None:
            guessed_real += 1
            if guess == g:
                correct += 1
            real_scores.append(lr_max)
        else:
            fake_scores.append(lr_max)
    acc = correct / max(guessed_real, 1)

    # AUC of "is this block real?" using the per-block likelihood-ratio score
    n_pos, n_neg = len(real_scores), len(fake_scores)
    auc = 0.0
    if n_pos and n_neg:
        ranks = sorted([(s, 1) for s in real_scores] + [(s, 0) for s in fake_scores],
                       key=lambda x: (x[0], x[1]))
        sum_ranks_pos = sum(i + 1 for i, (_, lab) in enumerate(ranks) if lab == 1)
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # Global chi-square on total label counts vs all-fake expectation
    total = Counter()
    for b in blocks:
        total.update(b)
    exp_n = len(blocks) * block
    chi2 = sum((total[t] - exp_n * W[t]) ** 2 / (exp_n * W[t]) for t in LABELS)
    return acc, chi2, len(blocks), auc


if __name__ == "__main__":
    random.seed(42)
    print("=== v1: uniform dummies, block=8, no fakes ===")
    for n_real in (32, 128, 512):
        acc, chi2, nb, auc = run(v1=True, n_real=n_real, fake_ratio=0.0, block=8)
        print(f"  real={n_real:5d} blocks={nb:5d} recovery={acc*100:5.1f}%  chi2={chi2:8.1f}")

    print("=== v2: aligned dummies, block=64, fakes 8:2 ===")
    for n_real in (32, 128, 512, 2048):
        acc, chi2, nb, auc = run(v1=False, n_real=n_real, fake_ratio=0.25, block=64)
        print(f"  real={n_real:5d} blocks={nb:5d} recovery={acc*100:5.1f}%  chi2={chi2:8.1f}  "
              f"real-vs-fake AUC={auc:.3f}")
