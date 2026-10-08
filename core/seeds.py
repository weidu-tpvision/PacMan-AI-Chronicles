"""
Canonical seed ranges and global seeding helpers.

Disjoint ranges guarantee that no seed used for training or model selection
is ever reused by the tournament test set (prevents selection leakage).

    TRAIN : 100_000 .. 999_999   (sampled randomly / sequentially by trainers)
    VAL   :  20_000 ..  20_999   (checkpoint / elite selection only)
    TEST  :   1_000 ..   1_999   (tournament benchmark only)
"""

import random

TRAIN_SEED_MIN = 100_000
TRAIN_SEED_MAX = 999_999
VAL_SEED_BASE = 20_000
TEST_SEED_BASE = 1_000


def train_seed(rng: random.Random) -> int:
    """Draw a training seed from the TRAIN range."""
    return rng.randint(TRAIN_SEED_MIN, TRAIN_SEED_MAX)


def val_seeds(n: int):
    return [VAL_SEED_BASE + i for i in range(n)]


def test_seeds(n: int):
    return [TEST_SEED_BASE + i for i in range(n)]


def seed_everything(seed: int) -> None:
    """Seed Python's global `random`, NumPy and (if available) PyTorch."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
