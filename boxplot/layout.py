from __future__ import annotations

import numpy as np


CATEGORY_GAP = 1.42
GROUP_SLOT_WIDTH = 0.32
GROUPED_BAR_FILL_RATIO = 0.94


def category_centers(category_count: int, *, gap: float = CATEGORY_GAP) -> np.ndarray:
    return np.arange(category_count, dtype=float) * gap


def dataset_offsets(dataset_count: int, *, slot_width: float = GROUP_SLOT_WIDTH) -> np.ndarray:
    if dataset_count <= 0:
        return np.empty(0, dtype=float)
    return np.linspace(-slot_width, slot_width, num=dataset_count)


def grouped_bar_width(
    dataset_count: int,
    *,
    slot_width: float = GROUP_SLOT_WIDTH,
    fill_ratio: float = GROUPED_BAR_FILL_RATIO,
) -> float:
    if dataset_count <= 0:
        return 0.0
    return slot_width * 2.0 / dataset_count * fill_ratio
