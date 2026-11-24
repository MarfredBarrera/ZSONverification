"""Stable object ids for planners without object clusters of their own (VLFM, StopWatch)."""
from typing import Optional

import numpy as np


class Clusters:
    """Stable object ids from per-step detections: a detection joins the nearest cluster within ``match_m``, else opens a new one."""

    def __init__(self, match_m: float) -> None:
        self.match_m = match_m
        self.sums: list = []   # per id: summed xy
        self.counts: list = []

    def centroid(self, cid: int) -> np.ndarray:
        return self.sums[cid] / self.counts[cid]

    def nearest(self, xy, within: Optional[float] = None) -> Optional[int]:
        if not self.sums:
            return None
        d = np.linalg.norm(np.array([self.centroid(i) for i in range(len(self.sums))]) - np.asarray(xy, float), axis=1)
        i = int(d.argmin())
        return i if within is None or d[i] <= within else None

    def add(self, xy) -> int:
        xy = np.asarray(xy, float)
        cid = self.nearest(xy, self.match_m)
        if cid is None:
            self.sums.append(np.zeros(2))
            self.counts.append(0)
            cid = len(self.sums) - 1
        self.sums[cid] += xy
        self.counts[cid] += 1
        return cid

    def fusion(self) -> dict:
        return {"clusters": [{"id": i, "centroid": tuple(map(float, self.centroid(i)))} for i in range(len(self.sums))]}
