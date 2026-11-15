"""Modular domain architecture for Explainable AI (SHAP, LIME, ALE)."""

from .contracts import (
    AleCurve,
    DomainError,
    FeatureAttribution,
    LimeExplanation,
    ValidationError,
    XaiContainer,
)
from .explainers import (
    AleCurveCalculator,
    ExactShapleyEstimator,
    LinearLimeSurrogate,
)
from .telemetry import InMemoryMetricsSink

__all__ = [
    "AleCurve",
    "AleCurveCalculator",
    "DomainError",
    "ExactShapleyEstimator",
    "FeatureAttribution",
    "InMemoryMetricsSink",
    "LimeExplanation",
    "LinearLimeSurrogate",
    "ValidationError",
    "XaiContainer",
]
