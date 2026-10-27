"""Shared helpers for the NN training notebooks and data pipelines."""

import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED):
    # Seeds do not guarantee deterministic cuDNN results with benchmarking on.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
