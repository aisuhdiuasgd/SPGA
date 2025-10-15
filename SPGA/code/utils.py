
from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Sequence, Optional
from dataloader import GetData

__all__ = [
    "seed_everything",
    "sample_uniform_triplets",
    "build_edge_index_from_matrix",
    "set_seed",
    "UniformSample",
    "get_edge_index",
]


def seed_everything(seed: int) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _sample_negative_disease(m_disease: int, positive_set: Sequence[int]) -> Optional[int]:

    if len(positive_set) >= m_disease:
        return None

    pos = set(map(int, positive_set))
    for _ in range(32):
        j = np.random.randint(0, m_disease)
        if j not in pos:
            return int(j)

    complement = np.setdiff1d(np.arange(m_disease), np.fromiter(pos, dtype=int))
    if complement.size == 0:
        return None
    return int(np.random.choice(complement))


def sample_uniform_triplets(dataset: GetData, n_samples: int = 2500) -> np.ndarray:

    snoRNAs = np.random.randint(0, dataset.n_snoRNA, int(n_samples))
    allPos: List[np.ndarray] = dataset.allPos
    S: List[Tuple[int, int, int]] = []

    for s in snoRNAs:
        pos_list = allPos[s]
        if len(pos_list) == 0:
            continue
        d_pos = int(np.random.choice(pos_list))
        d_neg = _sample_negative_disease(dataset.m_disease, pos_list)
        if d_neg is None:
            continue
        S.append((int(s), d_pos, d_neg))

    if len(S) == 0:
        print("warning: could not build triplets; please check training graph density.")
    return np.asarray(S, dtype=np.int64)


def build_edge_index_from_matrix(matrix: np.ndarray) -> torch.LongTensor:

    rows, cols = np.nonzero(matrix)
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    return torch.as_tensor(edge_index, dtype=torch.long)


def set_seed(seed: int) -> None:

    seed_everything(seed)


def UniformSample(dataset: GetData) -> np.ndarray:

    return sample_uniform_triplets(dataset, n_samples=2500)


def get_edge_index(matrix: np.ndarray) -> torch.LongTensor:

    return build_edge_index_from_matrix(matrix)
