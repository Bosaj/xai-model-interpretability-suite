"""Zero-dependency telemetry sink for XAI explainability latencies."""

from __future__ import annotations

import math
from collections import defaultdict


class InMemoryMetricsSink:
    """Collects counter increments and percentile execution timings."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.timings: dict[str, list[float]] = defaultdict(list)

    def increment(self, metric: str, value: int = 1, **tags: str) -> None:
        self.counters[self._format_key(metric, tags)] += value

    def observe_ms(self, metric: str, duration_ms: float, **tags: str) -> None:
        self.timings[self._format_key(metric, tags)].append(duration_ms)

    def percentile_ms(self, metric: str, pct: float, **tags: str) -> float:
        samples = sorted(self.timings.get(self._format_key(metric, tags), []))
        if not samples:
            return 0.0
        idx = min(len(samples) - 1, max(0, math.ceil((pct / 100.0) * len(samples)) - 1))
        return samples[idx]

    @staticmethod
    def _format_key(metric: str, tags: dict[str, str]) -> str:
        if not tags:
            return metric
        return f"{metric}{{{','.join(f'{k}={v}' for k, v in sorted(tags.items()))}}}"
