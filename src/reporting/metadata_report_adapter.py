"""Adapt metadata-comparison records to the existing text-report format.

The numerical engine remains the owner of comparisons. This module only
summarizes already-computed arrays for display and never changes match logic.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np


def _safe_float(value):
    """Convert a value to a finite float or return no value when it is invalid."""
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def calculate_relative_delta(delta, a, b, tolerance):
    """Calculate per-asset relative delta percentages.

    Differences within tolerance are treated as zero. Relative delta is
    undefined when the denominator is at or below tolerance.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if not (len(delta) == len(a) == len(b)):
        return None

    tolerance = max(0.0, float(tolerance or 0.0))
    abs_delta = np.abs(delta)
    denominator = np.maximum(np.abs(a), np.abs(b))

    valid = (
        np.isfinite(delta)
        & np.isfinite(a)
        & np.isfinite(b)
    )

    relative = np.full(delta.shape, np.nan, dtype=float)

    within_tolerance = valid & (abs_delta <= tolerance)
    relative[within_tolerance] = 0.0

    calculable = (
        valid
        & (abs_delta > tolerance)
        & (denominator > tolerance)
    )
    relative[calculable] = (
        100.0 * abs_delta[calculable] / denominator[calculable]
    )

    return relative


def make_asset_label(item, index):
    """Create a readable asset label from the available identity fields."""
    if index is None:
        return None

    ids = item.get("asset_ids")

    if ids is not None:
        ids_arr = np.asarray(ids).reshape(-1)

        if index < len(ids_arr):
            return (
                f"{item.get('asset_name', 'Asset')}_"
                f"{ids_arr[index]}"
            )

    return f"{item.get('asset_name', 'Asset')}_{index}"


def metadata_result_for_report(item):
    """Adapt one metadata comparison result to the shape used by existing reports."""
    stats = item.get("statistics") or {}
    a = item.get("case_a")
    b = item.get("case_b")
    delta = item.get("delta")
    delta_arr = None if delta is None else np.asarray(delta, dtype=float).reshape(-1)
    a_arr = None if a is None else np.asarray(a).reshape(-1)
    b_arr = None if b is None else np.asarray(b).reshape(-1)

    tolerance = _safe_float(item.get("tolerance")) or 0.0
    relative_delta_arr = (
        calculate_relative_delta(delta_arr, a_arr, b_arr, tolerance)
        if delta_arr is not None and a_arr is not None and b_arr is not None
        else None
    )

    finite_delta = np.array([], dtype=float)
    if delta_arr is not None:
        finite_delta = delta_arr[np.isfinite(delta_arr)]

    max_abs = _safe_float(np.max(np.abs(finite_delta))) if finite_delta.size else None
    mean_abs = _safe_float(np.mean(np.abs(finite_delta))) if finite_delta.size else None
    rms = _safe_float(np.sqrt(np.mean(np.square(finite_delta)))) if finite_delta.size else None
    percentiles = np.percentile(np.abs(finite_delta), [50, 90, 95, 99]) if finite_delta.size else [None] * 4

    # Find max absolute delta and corresponding asset
    max_index = None
    if finite_delta.size and delta_arr is not None:
        finite_indices = np.flatnonzero(np.isfinite(delta_arr))
        max_index = int(finite_indices[np.argmax(np.abs(delta_arr[finite_indices]))])
    elif item.get("mismatch_mask") is not None:
        mismatch_indices = np.flatnonzero(np.asarray(item["mismatch_mask"], dtype=bool).reshape(-1))
        if mismatch_indices.size:
            max_index = int(mismatch_indices[0])

    max_abs_diff_relative_delta = (
        None
        if relative_delta_arr is None or max_index is None or not np.isfinite(relative_delta_arr[max_index])
        else float(relative_delta_arr[max_index])
    )

    asset_label = make_asset_label(item, max_index)

    # Find max relative delta and corresponding asset

    max_relative_diff = None
    max_relative_diff_asset = None
    max_relative_diff_abs_delta = None

    if relative_delta_arr is not None:
        valid_relative_indices = np.flatnonzero(
            np.isfinite(relative_delta_arr)
        )

        if valid_relative_indices.size:
            relative_values = relative_delta_arr[valid_relative_indices]

            relative_max_position = int(np.argmax(relative_values))

            relative_max_index = int(valid_relative_indices[relative_max_position])

            max_relative_diff = float(relative_delta_arr[relative_max_index])

            max_relative_diff_asset = make_asset_label(item, relative_max_index)

            if delta_arr is not None:
                max_relative_diff_abs_delta = float(abs(delta_arr[relative_max_index]))

    raw_status = str(item.get("status", "NOT PERFORMED")).strip().upper()
    if raw_status in {"MATCH", "YES"}:
        status, is_match = "MATCH", True
    elif raw_status in {"MISMATCH", "SEVERE MISMATCH", "NO"}:
        status, is_match = "MISMATCH", False
    else:
        status, is_match = "NOT PERFORMED", None
    return SimpleNamespace(
        status=status,
        is_match=is_match,
        metric=item.get("display_name") or item.get("parameter_name") or item.get("parameter_id"),
        unit=item.get("unit"),
        compared_count=int(stats.get("comparable_count", 0) or 0),
        mismatch_count=int(stats.get("mismatch_count", 0) or 0),
        severe_count=int(stats.get("severe_mismatch_count", 0) or 0),
        tolerance=_safe_float(item.get("tolerance")) or 0.0,
        severity_threshold=_safe_float(item.get("severe_threshold")),
        max_abs_diff=max_abs,
        max_abs_diff_relative_delta=max_abs_diff_relative_delta,
        max_relative_diff=max_relative_diff,
        max_relative_diff_abs_delta=max_relative_diff_abs_delta,
        mean_abs_diff=mean_abs,
        rms_diff=rms,
        p50_abs_diff=_safe_float(percentiles[0]),
        p90_abs_diff=_safe_float(percentiles[1]),
        p95_abs_diff=_safe_float(percentiles[2]),
        p99_abs_diff=_safe_float(percentiles[3]),
        max_abs_diff_asset=asset_label,
        max_relative_diff_asset=max_relative_diff_asset,
        max_abs_diff_index=max_index,
        reason=item.get("reason"),
        _case_a=a_arr,
        _case_b=b_arr,
        _delta=delta_arr,
    )


def metadata_groups(results, key="asset_name"):
    """Group metadata comparison results for report presentation."""
    grouped = {}
    for item in results or []:
        grouped.setdefault(str(item.get(key, "unknown")).lower(), []).append(item)
    return grouped


def asset_summary_status(entries):
    """
    Determine the overall status for one asset from its metric results.

    Possible results:
        YES
        NO
        NOT PERFORMED
        Skipped because Ybus matched.
    """
    if not entries:
        return "NOT PERFORMED"

    statuses = {
        str(entry.get("status", "NOT PERFORMED")).strip().upper()
        for entry in entries
    }

    # If every metric was skipped for the same reason, report
    # the reason at asset level instead of listing each metric.
    if statuses == {"NOT PERFORMED"}:
        reasons = {
            entry.get("reason")
            for entry in entries
            if entry.get("reason")
        }

        if reasons == {"Skipped because Ybus matched."}:
            return "Skipped because Ybus matched."

        return "NOT PERFORMED"

    # Any mismatching metric means the asset does not match.
    if statuses.intersection({"NO", "MISMATCH", "SEVERE MISMATCH"}):
        return "NO"

    # Incomplete asset comparisons should not be reported as a match.
    if "NOT PERFORMED" in statuses:
        return "NOT PERFORMED"

    # All metrics match.
    if statuses.issubset({"YES", "MATCH"}) and statuses:
        return "YES"

    return "NOT PERFORMED"
