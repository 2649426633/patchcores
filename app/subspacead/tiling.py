from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size >= length:
        return [0]
    stride = max(1.0, tile_size * (1.0 - overlap))
    count = max(2, int(math.ceil((length - tile_size) / stride)) + 1)
    values = np.linspace(0, length - tile_size, count)
    starts = sorted({int(round(v)) for v in values})
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def compute_tile_windows(
    image_size: tuple[int, int],
    tile_fraction: float = 0.50,
    overlap: float = 0.25,
) -> list[TileWindow]:
    w, h = image_size
    if not (0.2 <= tile_fraction <= 1.0):
        raise ValueError("tile_fraction must be in [0.2, 1.0]")
    if not (0.0 <= overlap < 0.9):
        raise ValueError("overlap must be in [0.0, 0.9)")

    tile_size = max(32, int(round(min(w, h) * tile_fraction)))
    tile_size = min(tile_size, w, h)
    xs = _axis_starts(w, tile_size, overlap)
    ys = _axis_starts(h, tile_size, overlap)
    return [
        TileWindow(x, y, x + tile_size, y + tile_size)
        for y in ys
        for x in xs
    ]
