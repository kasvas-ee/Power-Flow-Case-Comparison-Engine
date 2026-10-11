import numpy as np
from typing import List, Optional

from src.comparison.result_structures import (
    ComparisonResult,
    TopSevereResult,
)


def get_top_severe(
    values_a,
    values_b,
    comparison_result: ComparisonResult,
    row_ids,
    n: int = 5,
    asset_name: str = "Asset",
    angle: bool = False,
) -> List[TopSevereResult]:
    """
    Return the top N severe differences for one metric.

    A difference is considered severe when:

        abs(delta) > comparison_result.severity_threshold

    NaN-versus-value comparisons are excluded from severe ranking
    because they do not have a meaningful numerical magnitude.

    Parameters
    ----------
    values_a, values_b:
        Aligned Case A and Case B values.

    comparison_result:
        Result produced by compare_metric().

    row_ids:
        Asset identifiers corresponding to the values.

    n:
        Maximum number of severe results to return.

    asset_name:
        Prefix used to construct the reportable asset identifier.

    angle:
        If True, calculate signed angular differences using the
        shortest path around the 360-degree boundary.
    """

    if n <= 0:
        return []

    if comparison_result.severity_threshold is None:
        return []

    values_a = np.asarray(
        values_a,
        dtype=float,
    )

    values_b = np.asarray(
        values_b,
        dtype=float,
    )

    row_ids = np.asarray(row_ids)

    if values_a.shape != values_b.shape:
        raise ValueError(
            "values_a and values_b must have the same shape."
        )

    if values_a.ndim != 1:
        raise ValueError(
            "get_top_severe expects one-dimensional metric arrays."
        )

    if len(row_ids) != len(values_a):
        raise ValueError(
            "row_ids length must match the number of values."
        )

    severity_threshold = (
        comparison_result.severity_threshold
    )

    # ---------------------------------------------------------
    # Calculate signed differences.
    # ---------------------------------------------------------

    nan_a = np.isnan(values_a)
    nan_b = np.isnan(values_b)

    one_nan = nan_a ^ nan_b

    if angle:
        delta = (
            (values_b - values_a + 180.0) % 360.0
        ) - 180.0
    else:
        delta = values_b - values_a

    abs_delta = np.abs(delta)

    # ---------------------------------------------------------
    # Severe means strictly greater than the threshold.
    #
    # This matches compare_metric().
    # ---------------------------------------------------------

    severe_mask = (
        (abs_delta > severity_threshold)
        & ~one_nan
    )

    severe_indices = np.flatnonzero(
        severe_mask
    )

    if severe_indices.size == 0:
        return []

    # ---------------------------------------------------------
    # Rank by descending absolute difference.
    # ---------------------------------------------------------

    ordered_positions = severe_indices[
        np.argsort(
            abs_delta[severe_indices]
        )[::-1]
    ]

    top_positions = ordered_positions[:n]

    results = []

    for rank, index in enumerate(
        top_positions,
        start=1,
    ):
        results.append(
            TopSevereResult(
                rank=rank,
                asset=(
                    f"{asset_name}_{row_ids[index]}"
                ),
                value_a=float(values_a[index]),
                value_b=float(values_b[index]),
                delta=float(delta[index]),
                abs_delta=float(abs_delta[index]),
            )
        )

    return results


def rank_metric(
    values_a,
    values_b,
    comparison_result: ComparisonResult,
    row_ids,
    n: int = 5,
    asset_name: str = "Asset",
) -> List[TopSevereResult]:
    """
    Convenience wrapper for ordinary numerical metrics.
    """

    return get_top_severe(
        values_a=values_a,
        values_b=values_b,
        comparison_result=comparison_result,
        row_ids=row_ids,
        n=n,
        asset_name=asset_name,
        angle=False,
    )


def rank_angle_metric(
    values_a,
    values_b,
    comparison_result: ComparisonResult,
    row_ids,
    n: int = 5,
    asset_name: str = "Bus",
) -> List[TopSevereResult]:
    """
    Convenience wrapper for angular metrics.

    Uses the shortest signed angular difference.
    """

    return get_top_severe(
        values_a=values_a,
        values_b=values_b,
        comparison_result=comparison_result,
        row_ids=row_ids,
        n=n,
        asset_name=asset_name,
        angle=True,
    )