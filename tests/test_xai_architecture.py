"""Unit, axiomatic, observability, and load tests for xai_core."""

from __future__ import annotations

import math
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from xai_core import (  # noqa: E402
    AleCurveCalculator,
    ExactShapleyEstimator,
    InMemoryMetricsSink,
    LinearLimeSurrogate,
    ValidationError,
    XaiContainer,
)


class _NonlinearPredictor:
    """f(x0, x1, x2) = 2 * x0 + x0 * x1 (x2 is a strict dummy feature)."""

    def predict_batch(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [2.0 * r[0] + r[0] * r[1] for r in rows]


def _build_container() -> XaiContainer:
    container = XaiContainer()
    container.register_singleton(InMemoryMetricsSink, lambda _: InMemoryMetricsSink())
    container.register_singleton(
        ExactShapleyEstimator,
        lambda c: ExactShapleyEstimator(metrics=c.resolve(InMemoryMetricsSink)),
    )
    container.register_singleton(
        LinearLimeSurrogate,
        lambda c: LinearLimeSurrogate(metrics=c.resolve(InMemoryMetricsSink)),
    )
    container.register_singleton(
        AleCurveCalculator,
        lambda c: AleCurveCalculator(metrics=c.resolve(InMemoryMetricsSink)),
    )
    return container


class TestXaiArchitecture(unittest.TestCase):
    def test_exact_shapley_efficiency_and_dummy_axioms(self) -> None:
        container = _build_container()
        shap_est: ExactShapleyEstimator = container.resolve(ExactShapleyEstimator)
        predictor = _NonlinearPredictor()
        bg = [[0.0, 0.0, 1.0], [2.0, 2.0, 5.0]]
        instance = [3.0, 4.0, 9.0]

        attr = shap_est.explain(
            predictor, instance, bg, feature_names=["amount", "rate", "unused"]
        )
        # Efficiency axiom: sum(phi_i) == f(x) - E[f(X)]
        self.assertAlmostEqual(
            sum(attr.shapley_values),
            attr.predicted_value - attr.expected_value,
            places=9,
        )
        # Dummy axiom: unused feature gets zero attribution
        self.assertAlmostEqual(attr.shapley_values[2], 0.0, places=9)

    def test_lime_surrogate_recovers_local_gradients_and_validates(self) -> None:
        container = _build_container()
        lime: LinearLimeSurrogate = container.resolve(LinearLimeSurrogate)
        predictor = _NonlinearPredictor()

        # At x = [3.0, 4.0, 0.0], partial derivatives are df/dx0 = 2 + 4 = 6, df/dx1 = 3, df/dx2 = 0
        exp = lime.explain(
            predictor,
            instance=[3.0, 4.0, 0.0],
            feature_scales=[0.2, 0.2, 0.2],
            feature_names=["x0", "x1", "x2"],
            n_samples=300,
            kernel_width=1.0,
        )
        self.assertGreater(exp.local_r2, 0.99)
        self.assertTrue(math.isclose(exp.weights[0], 6.0, abs_tol=0.35))
        self.assertTrue(math.isclose(exp.weights[1], 3.0, abs_tol=0.35))
        self.assertTrue(math.isclose(exp.weights[2], 0.0, abs_tol=0.1))

        with self.assertRaises(ValidationError):
            lime.explain(predictor, [1.0, 2.0, 3.0], kernel_width=0.0)

    def test_ale_zero_centering_and_benchmark(self) -> None:
        container = _build_container()
        ale_calc: AleCurveCalculator = container.resolve(AleCurveCalculator)
        shap_est: ExactShapleyEstimator = container.resolve(ExactShapleyEstimator)
        metrics: InMemoryMetricsSink = container.resolve(InMemoryMetricsSink)
        predictor = _NonlinearPredictor()

        dataset = [[float(i), 2.0, 0.0] for i in range(1, 51)]
        curve = ale_calc.compute(predictor, dataset, feature_index=0, n_bins=5)

        weighted_mean = sum(
            e * c for e, c in zip(curve.effects, curve.sample_counts, strict=True)
        ) / sum(curve.sample_counts)
        self.assertAlmostEqual(weighted_mean, 0.0, places=9)
        self.assertLess(curve.effects[0], curve.effects[-1])

        bg = [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
        for k in range(150):
            shap_est.explain(predictor, [float(k % 10), 2.0, 0.0], bg)

        self.assertEqual(metrics.counters["xai.shap_explanations"], 150)
        self.assertLess(metrics.percentile_ms("xai.shap_ms", 95), 50.0)


if __name__ == "__main__":
    unittest.main()
