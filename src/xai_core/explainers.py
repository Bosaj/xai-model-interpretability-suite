"""Pure-Python Explainable AI engines: Exact Shapley, Weighted LIME, and 1D ALE."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from itertools import combinations

from .contracts import (
    AleCurve,
    FeatureAttribution,
    LimeExplanation,
    MetricsSink,
    PredictorProtocol,
    ValidationError,
)
from .telemetry import InMemoryMetricsSink

_MAX_EXACT_SHAP_FEATURES = 10


class ExactShapleyEstimator:
    """Computes exact cooperative Shapley values over all 2^d feature coalitions."""

    def __init__(self, metrics: MetricsSink | None = None) -> None:
        self._metrics = metrics or InMemoryMetricsSink()

    def explain(
        self,
        predictor: PredictorProtocol,
        instance: Sequence[float],
        background: Sequence[Sequence[float]],
        feature_names: Sequence[str] | None = None,
    ) -> FeatureAttribution:
        started = time.perf_counter()
        if not instance:
            raise ValidationError("instance cannot be empty")
        if not background:
            raise ValidationError("background dataset cannot be empty")
        n_features = len(instance)
        if n_features > _MAX_EXACT_SHAP_FEATURES:
            raise ValidationError(
                f"Exact coalition enumeration supports at most {_MAX_EXACT_SHAP_FEATURES} features"
            )
        for row in background:
            if len(row) != n_features:
                raise ValidationError("All background rows must match instance dimension")

        names = tuple(feature_names) if feature_names else tuple(f"f{i}" for i in range(n_features))
        if len(names) != n_features:
            raise ValidationError("feature_names length must match instance dimension")

        coalition_values: dict[int, float] = {}
        n_bg = len(background)
        all_rows: list[list[float]] = []
        masks: list[int] = list(range(1 << n_features))

        for mask in masks:
            for bg_row in background:
                all_rows.append(
                    [float(instance[j]) if (mask >> j) & 1 else float(bg_row[j]) for j in range(n_features)]
                )

        preds = predictor.predict_batch(all_rows)
        for idx, mask in enumerate(masks):
            chunk = preds[idx * n_bg : (idx + 1) * n_bg]
            coalition_values[mask] = sum(chunk) / n_bg

        fact = math.factorial
        shap_vals: list[float] = []
        others = list(range(n_features))
        for feat_idx in range(n_features):
            pool = [i for i in others if i != feat_idx]
            phi = 0.0
            for s_len in range(n_features):
                weight = (fact(s_len) * fact(n_features - s_len - 1)) / fact(n_features)
                for subset in combinations(pool, s_len):
                    mask_without = 0
                    for elem in subset:
                        mask_without |= 1 << elem
                    mask_with = mask_without | (1 << feat_idx)
                    phi += weight * (coalition_values[mask_with] - coalition_values[mask_without])
            shap_vals.append(phi)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._metrics.increment("xai.shap_explanations")
        self._metrics.observe_ms("xai.shap_ms", elapsed_ms)
        return FeatureAttribution(
            feature_names=names,
            shapley_values=tuple(shap_vals),
            expected_value=coalition_values[0],
            predicted_value=coalition_values[(1 << n_features) - 1],
        )


class LinearLimeSurrogate:
    """Weighted ridge regression local surrogate explainer using RBF distance kernels."""

    def __init__(self, metrics: MetricsSink | None = None) -> None:
        self._metrics = metrics or InMemoryMetricsSink()

    def explain(
        self,
        predictor: PredictorProtocol,
        instance: Sequence[float],
        feature_scales: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
        n_samples: int = 200,
        kernel_width: float = 0.75,
        l2_reg: float = 1e-3,
        seed: int = 42,
    ) -> LimeExplanation:
        started = time.perf_counter()
        if not instance:
            raise ValidationError("instance cannot be empty")
        if kernel_width <= 0:
            raise ValidationError("kernel_width must be positive")
        if n_samples < len(instance) + 2:
            raise ValidationError("n_samples too small relative to feature count")
        if l2_reg < 0:
            raise ValidationError("l2_reg cannot be negative")

        d = len(instance)
        scales = [abs(float(s)) if s != 0 else 1.0 for s in (feature_scales or [1.0] * d)]
        if len(scales) != d:
            raise ValidationError("feature_scales length must match instance")
        names = tuple(feature_names) if feature_names else tuple(f"f{i}" for i in range(d))
        if len(names) != d:
            raise ValidationError("feature_names length must match instance")

        rng = random.Random(seed)
        samples: list[list[float]] = [list(map(float, instance))]
        for _ in range(n_samples - 1):
            samples.append(
                [float(instance[j]) + rng.gauss(0.0, scales[j] * 0.5) for j in range(d)]
            )

        y_vals = predictor.predict_batch(samples)
        weights: list[float] = []
        for row in samples:
            sq_norm_dist = sum(
                ((row[j] - float(instance[j])) / scales[j]) ** 2 for j in range(d)
            )
            weights.append(math.exp(-sq_norm_dist / (kernel_width**2)))

        # Build (d + 1) x (d + 1) normal matrix A = X^T W X + l2_reg * I and b = X^T W y
        cols = d + 1
        matrix_a = [[0.0] * cols for _ in range(cols)]
        vec_b = [0.0] * cols

        for row, y, w in zip(samples, y_vals, weights, strict=True):
            aug = [1.0, *row]
            for i in range(cols):
                wi = w * aug[i]
                vec_b[i] += wi * y
                for j in range(cols):
                    matrix_a[i][j] += wi * aug[j]

        for j in range(1, cols):
            matrix_a[j][j] += l2_reg

        beta = _solve_linear_system(matrix_a, vec_b)
        intercept = beta[0]
        coef = tuple(beta[1:])

        # Weighted R^2
        total_w = sum(weights)
        w_mean = sum(w * y for w, y in zip(weights, y_vals, strict=True)) / total_w
        ss_tot = sum(w * ((y - w_mean) ** 2) for w, y in zip(weights, y_vals, strict=True))
        ss_res = 0.0
        for row, y, w in zip(samples, y_vals, weights, strict=True):
            pred_hat = intercept + sum(c * v for c, v in zip(coef, row, strict=True))
            ss_res += w * ((y - pred_hat) ** 2)
        local_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._metrics.increment("xai.lime_explanations")
        self._metrics.observe_ms("xai.lime_ms", elapsed_ms)
        return LimeExplanation(
            feature_names=names,
            weights=coef,
            intercept=intercept,
            local_r2=local_r2,
        )


class AleCurveCalculator:
    """Computes zero-centered 1D Accumulated Local Effects across quantile intervals."""

    def __init__(self, metrics: MetricsSink | None = None) -> None:
        self._metrics = metrics or InMemoryMetricsSink()

    def compute(
        self,
        predictor: PredictorProtocol,
        dataset: Sequence[Sequence[float]],
        feature_index: int,
        n_bins: int = 5,
    ) -> AleCurve:
        started = time.perf_counter()
        if not dataset:
            raise ValidationError("dataset cannot be empty")
        if not isinstance(feature_index, int) or isinstance(feature_index, bool):
            raise TypeError("feature_index must be an integer")
        n_features = len(dataset[0])
        if not 0 <= feature_index < n_features:
            raise ValidationError("feature_index out of bounds")
        if n_bins <= 0:
            raise ValidationError("n_bins must be positive")
        for row in dataset:
            if len(row) != n_features:
                raise ValidationError("Inconsistent row length in dataset")

        values = sorted(float(row[feature_index]) for row in dataset)
        n = len(values)
        edges: list[float] = [values[0]]
        for b in range(1, n_bins + 1):
            pos = min(n - 1, math.ceil((b / n_bins) * n) - 1)
            edges.append(values[pos])

        bin_deltas = [0.0] * n_bins
        bin_counts = [0] * n_bins
        low_rows: list[list[float]] = []
        high_rows: list[list[float]] = []
        row_bins: list[int] = []

        for row in dataset:
            x_val = float(row[feature_index])
            b_idx = n_bins - 1
            for k in range(n_bins):
                left, right = edges[k], edges[k + 1]
                if (k == 0 and x_val <= right) or (left < x_val <= right):
                    b_idx = k
                    break
            low_row = list(map(float, row))
            high_row = list(map(float, row))
            low_row[feature_index] = edges[b_idx]
            high_row[feature_index] = edges[b_idx + 1]
            low_rows.append(low_row)
            high_rows.append(high_row)
            row_bins.append(b_idx)

        y_low = predictor.predict_batch(low_rows)
        y_high = predictor.predict_batch(high_rows)
        for b_idx, lo, hi in zip(row_bins, y_low, y_high, strict=True):
            bin_deltas[b_idx] += hi - lo
            bin_counts[b_idx] += 1

        uncentered: list[float] = []
        running = 0.0
        for b_idx in range(n_bins):
            mean_diff = (bin_deltas[b_idx] / bin_counts[b_idx]) if bin_counts[b_idx] > 0 else 0.0
            running += mean_diff
            uncentered.append(running)

        total_count = sum(bin_counts)
        weighted_mean = (
            sum(eff * cnt for eff, cnt in zip(uncentered, bin_counts, strict=True)) / total_count
            if total_count > 0
            else 0.0
        )
        centered = tuple(eff - weighted_mean for eff in uncentered)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._metrics.increment("xai.ale_runs")
        self._metrics.observe_ms("xai.ale_ms", elapsed_ms)
        return AleCurve(
            feature_index=feature_index,
            bin_edges=tuple(edges),
            effects=centered,
            sample_counts=tuple(bin_counts),
        )


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solves Ax = b via partial-pivoting Gaussian elimination."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValidationError("Singular normal equations in LIME surrogate fit")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]
