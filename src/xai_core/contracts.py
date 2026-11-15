"""Typed DTOs, protocols, errors, and dependency injection container for XAI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class DomainError(Exception):
    """Base domain exception."""


class ValidationError(DomainError):
    """Raised when attribution or model inputs violate domain constraints."""


@dataclass(frozen=True, slots=True)
class FeatureAttribution:
    """Exact Shapley decomposition for a single prediction."""

    feature_names: tuple[str, ...]
    shapley_values: tuple[float, ...]
    expected_value: float
    predicted_value: float


@dataclass(frozen=True, slots=True)
class LimeExplanation:
    """Local linear surrogate weights and weighted fidelity score."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    local_r2: float


@dataclass(frozen=True, slots=True)
class AleCurve:
    """Centered 1D Accumulated Local Effects curve across quantile bins."""

    feature_index: int
    bin_edges: tuple[float, ...]
    effects: tuple[float, ...]
    sample_counts: tuple[int, ...]


class MetricsSink(Protocol):
    """Observability contract for counters and timings."""

    def increment(self, metric: str, value: int = 1, **tags: str) -> None: ...

    def observe_ms(self, metric: str, duration_ms: float, **tags: str) -> None: ...


class PredictorProtocol(Protocol):
    """Batch prediction interface decoupled from framework-specific models."""

    def predict_batch(self, rows: Sequence[Sequence[float]]) -> list[float]: ...


class XaiContainer:
    """Singleton dependency injection container for XAI explainers."""

    def __init__(self) -> None:
        self._factories: dict[type[Any], Callable[[XaiContainer], Any]] = {}
        self._singletons: dict[type[Any], Any] = {}

    def register_singleton(
        self, interface: type[Any], factory: Callable[[XaiContainer], Any]
    ) -> None:
        self._factories[interface] = factory
        self._singletons.pop(interface, None)

    def resolve(self, interface: type[Any]) -> Any:
        if interface in self._singletons:
            return self._singletons[interface]
        factory = self._factories.get(interface)
        if factory is None:
            raise KeyError(f"No provider registered for {interface.__name__}")
        instance = factory(self)
        self._singletons[interface] = instance
        return instance
