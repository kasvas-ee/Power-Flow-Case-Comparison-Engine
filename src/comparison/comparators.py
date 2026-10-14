"""Vectorized numerical comparison helpers for aligned arrays of power-system metrics."""

import numpy as np
from src.comparison.result_structures import ComparisonResult


def _calculate_differences(
    values_a,
    values_b,
    angle=False,
):
    """
    Calculate absolute differences between two aligned arrays.

    NaN handling:
        NaN vs NaN  -> treated as equal
        NaN vs value -> mismatch
        value vs NaN -> mismatch

    For angles, differences are wrapped to [-180, 180).
    """

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    nan_a = np.isnan(values_a)
    nan_b = np.isnan(values_b)

    # Build explicit validity masks before subtraction so paired and one-sided NaNs follow different rules.
    both_nan = nan_a & nan_b
    one_nan = nan_a ^ nan_b

    if angle:
        differences = (
            (values_a - values_b + 180.0) % 360.0
        ) - 180.0

        # NaN vs NaN is considered equal.
        differences[both_nan] = 0.0

        # One-sided NaN remains invalid for comparison.
        differences[one_nan] = np.nan

    else:
        # Avoid NaN propagation for values where both sides are NaN.
        a = values_a.copy()
        b = values_b.copy()

        a[both_nan] = 0.0
        b[both_nan] = 0.0

        differences = np.abs(a - b)

        # One-sided NaN is a mismatch.
        differences[one_nan] = np.nan

    abs_differences = np.abs(differences)

    return abs_differences, one_nan


def _build_comparison_result(
    *,
    metric,
    unit,
    abs_differences,
    one_nan,
    tol,
    severity_threshold=None,
    asset_name="Asset",
    row_ids=None,
):
    """
    Build the common statistics and mismatch information used by
    metric comparisons.
    """

    if row_ids is None:
        row_ids = np.arange(abs_differences.shape[0])
    else:
        row_ids = np.asarray(row_ids)

    # A one-sided NaN is always a mismatch.
    mismatch_mask = (abs_differences > tol) | one_nan

    if severity_threshold is not None:
        severe_mask = (
            (abs_differences > severity_threshold)
            & ~one_nan
        )
    else:
        severe_mask = np.zeros_like(
            abs_differences,
            dtype=bool,
        )

    mismatch_count = int(np.sum(mismatch_mask))
    severe_count = int(np.sum(severe_mask))

    severe_indices = np.flatnonzero(severe_mask)

    severe_rows = [
        f"{asset_name}_{row_ids[i]}"
        for i in severe_indices
    ]

    # Exclude one-sided NaNs from numerical statistics.
    valid_mask = ~one_nan
    valid_diffs = abs_differences[valid_mask]

    compared_count = int(valid_diffs.size)

    if valid_diffs.size > 0:
        valid_indices = np.flatnonzero(valid_mask)

        max_position = int(np.argmax(valid_diffs))
        max_abs_diff_index = int(
            valid_indices[max_position]
        )

        max_abs_diff = float(
            valid_diffs[max_position]
        )

        mean_abs_diff = float(
            np.mean(valid_diffs)
        )

        rms_diff = float(
            np.sqrt(np.mean(valid_diffs ** 2))
        )

        # One NumPy call for all requested percentiles.
        percentiles = np.percentile(
            valid_diffs,
            [50, 90, 95, 99],
        )

        p50_abs_diff, p90_abs_diff, p95_abs_diff, p99_abs_diff = map(
            float,
            percentiles,
        )

        max_abs_diff_asset = (
            f"{asset_name}_{row_ids[max_abs_diff_index]}"
        )

    else:
        max_abs_diff = None
        mean_abs_diff = None
        rms_diff = None

        p50_abs_diff = None
        p90_abs_diff = None
        p95_abs_diff = None
        p99_abs_diff = None

        max_abs_diff_index = None
        max_abs_diff_asset = None

    is_match = mismatch_count == 0

    if severity_threshold is not None:
        reason = (
            f"{metric}: {mismatch_count} values exceed "
            f"tolerance {tol}; {severe_count} are severe "
            f"(>{severity_threshold})."
        )
    else:
        reason = (
            f"{metric}: {mismatch_count} values exceed "
            f"tolerance {tol}."
        )

    return ComparisonResult(
        metric=metric,
        unit=unit,
        is_match=is_match,
        compared_count=compared_count,
        mismatch_count=mismatch_count,
        severe_count=severe_count,
        severe_rows=severe_rows,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rms_diff=rms_diff,
        p50_abs_diff=p50_abs_diff,
        p90_abs_diff=p90_abs_diff,
        p95_abs_diff=p95_abs_diff,
        p99_abs_diff=p99_abs_diff,
        max_abs_diff_index=max_abs_diff_index,
        max_abs_diff_asset=max_abs_diff_asset,
        tolerance=tol,
        severity_threshold=severity_threshold,
        reason=reason,
    )


def compare_np(
    np_a,
    np_b,
    filename,
    tol=0,
    check_row_size=False,
    check_col_size=False,
    sort_columns=None,
    match_columns=None,
):
    """
    Compare two NumPy arrays.

    Parameters
    ----------
    np_a, np_b:
        Arrays to compare.

    filename:
        Name used in the result reason.

    tol:
        Absolute numerical tolerance.

    check_row_size:
        Whether to require equal row counts.

    check_col_size:
        Whether to require equal column counts.

    sort_columns:
        Columns used to sort both arrays before comparison.

    match_columns:
        Columns that should actually be compared.
        If None, all columns are compared.
    """

    np_a = np.asarray(np_a)
    np_b = np.asarray(np_b)

    # ---------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------

    if check_row_size and np_a.shape[0] != np_b.shape[0]:
        reason = (
            f"Row size mismatch: A has {np_a.shape[0]} rows, "
            f"B has {np_b.shape[0]} rows."
        )

        return ComparisonResult(
            metric=filename,
            unit="",
            is_match=False,
            tolerance=tol,
            reason=reason,
        )

    if check_col_size and np_a.shape[1] != np_b.shape[1]:
        reason = (
            f"Column size mismatch: A has {np_a.shape[1]} columns, "
            f"B has {np_b.shape[1]} columns."
        )

        return ComparisonResult(
            metric=filename,
            unit="",
            is_match=False,
            tolerance=tol,
            reason=reason,
        )

    # ---------------------------------------------------------
    # Optional sorting
    # ---------------------------------------------------------

    if sort_columns is not None:
        np_a = np_a[
            np.lexsort(
                [
                    np_a[:, col]
                    for col in reversed(sort_columns)
                ]
            )
        ]

        np_b = np_b[
            np.lexsort(
                [
                    np_b[:, col]
                    for col in reversed(sort_columns)
                ]
            )
        ]

    # ---------------------------------------------------------
    # Select comparison columns
    # ---------------------------------------------------------

    if match_columns is not None:
        compare_a = np_a[:, match_columns].copy()
        compare_b = np_b[:, match_columns].copy()
    else:
        compare_a = np_a.copy()
        compare_b = np_b.copy()

    if compare_a.shape != compare_b.shape:
        return ComparisonResult(
            metric=filename,
            unit="",
            is_match=False,
            tolerance=tol,
            reason=(
                f"Comparison shape mismatch: "
                f"A={compare_a.shape}, B={compare_b.shape}"
            ),
        )

    # ---------------------------------------------------------
    # Difference calculation
    # ---------------------------------------------------------

    abs_differences, one_nan = _calculate_differences(
        compare_a,
        compare_b,
    )

    mismatch_mask = (
        (abs_differences > tol)
        | one_nan
    )

    compared_count = int(
        abs_differences.size
    )

    mismatch_count = int(
        np.sum(mismatch_mask)
    )

    if compared_count == 0:
        return ComparisonResult(
            metric=filename,
            unit="",
            is_match=False,
            compared_count=0,
            mismatch_count=0,
            tolerance=tol,
            reason=(
                f"{filename}: nothing to compare."
            ),
        )

    valid_mask = ~one_nan
    valid_diffs = abs_differences[valid_mask]

    if valid_diffs.size > 0:
        max_position = int(
            np.argmax(valid_diffs)
        )

        valid_indices = np.flatnonzero(
            valid_mask
        )

        max_abs_diff_index = int(
            valid_indices[max_position]
        )

        max_abs_diff = float(
            valid_diffs[max_position]
        )
    else:
        max_abs_diff = None
        max_abs_diff_index = None

    is_match = mismatch_count == 0

    if is_match:
        reason = (
            f"{filename}: data matches within "
            f"tolerance {tol}."
        )
    else:
        reason = (
            f"{filename}: {mismatch_count} values "
            f"exceed tolerance {tol}."
        )

    return ComparisonResult(
        metric=filename,
        unit="",
        is_match=is_match,
        compared_count=compared_count,
        mismatch_count=mismatch_count,
        max_abs_diff=max_abs_diff,
        max_abs_diff_index=max_abs_diff_index,
        tolerance=tol,
        reason=reason,
    )


def compare_metric(
    values_a,
    values_b,
    metric,
    unit="",
    tol=0.0,
    severity_threshold=None,
    asset_name="Asset",
    row_ids=None,
):
    """
    Compare one numerical power-flow metric.

    Examples
    --------
    Bus voltage magnitude:
        compare_metric(
            vm_a,
            vm_b,
            metric="Bus voltage magnitude",
            unit="pu",
            tol=1e-4,
            severity_threshold=1e-2,
            asset_name="Bus",
            row_ids=bus_ids,
        )

    Bus voltage angle:
        compare_metric(
            va_a,
            va_b,
            metric="Bus voltage angle",
            unit="degree",
            tol=1e-3,
            severity_threshold=1.0,
            asset_name="Bus",
            row_ids=bus_ids,
        )
    """

    values_a = np.asarray(
        values_a,
        dtype=float,
    )

    values_b = np.asarray(
        values_b,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------

    if values_a.shape != values_b.shape:
        return ComparisonResult(
            metric=metric,
            unit=unit,
            is_match=False,
            tolerance=tol,
            severity_threshold=severity_threshold,
            severe_rows=[],
            reason=(
                f"Shape mismatch: "
                f"A={values_a.shape}, "
                f"B={values_b.shape}"
            ),
        )

    # ---------------------------------------------------------
    # Row ID validation
    # ---------------------------------------------------------

    if row_ids is None:
        row_ids = np.arange(
            values_a.shape[0]
        )
    else:
        row_ids = np.asarray(row_ids)

        if len(row_ids) != values_a.shape[0]:
            return ComparisonResult(
                metric=metric,
                unit=unit,
                is_match=False,
                tolerance=tol,
                severity_threshold=severity_threshold,
                severe_rows=[],
                reason=(
                    f"row_ids length ({len(row_ids)}) "
                    f"does not match number of rows "
                    f"({values_a.shape[0]})."
                ),
            )

    # ---------------------------------------------------------
    # Difference calculation
    # ---------------------------------------------------------

    is_angle = metric == "Bus voltage angle"

    abs_differences, one_nan = _calculate_differences(
        values_a,
        values_b,
        angle=is_angle,
    )

    # ---------------------------------------------------------
    # Build result
    # ---------------------------------------------------------

    return _build_comparison_result(
        metric=metric,
        unit=unit,
        abs_differences=abs_differences,
        one_nan=one_nan,
        tol=tol,
        severity_threshold=severity_threshold,
        asset_name=asset_name,
        row_ids=row_ids,
    )