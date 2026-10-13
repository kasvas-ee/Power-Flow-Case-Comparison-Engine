from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


# ============================================================
# Input and statistics helpers
# ============================================================

def _get_metadata_comparisons(comparison: Any) -> list[dict]:
    """
    Extract metadata comparisons from a model or power-flow result.

    Model comparisons are wrapped in an ActionResult.
    Power-flow comparisons are passed as their result object.
    """

    if comparison is None:
        return []

    result = getattr(comparison, "result", comparison)

    metadata = getattr(
        result,
        "metadata_comparisons",
        [],
    )

    return list(metadata or [])


def _get_statistics(item: dict) -> dict:
    """Return the statistics calculated by the comparison engine."""

    statistics = item.get("statistics")

    if isinstance(statistics, dict):
        return statistics

    return {}


def _finite_float(value: Any) -> float | None:
    """Convert a value to a finite float, if possible."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if np.isfinite(number) else None


def _get_comparable_count(item: dict) -> int:
    statistics = _get_statistics(item)

    try:
        return max(
            0,
            int(statistics.get("comparable_count", 0) or 0),
        )
    except (TypeError, ValueError):
        return 0


def _get_mismatch_count(item: dict) -> int:
    statistics = _get_statistics(item)

    try:
        return max(
            0,
            int(statistics.get("mismatch_count", 0) or 0),
        )
    except (TypeError, ValueError):
        return 0


def _get_mismatch_percent(item: dict) -> float | None:
    statistics = _get_statistics(item)

    percentage = _finite_float(
        statistics.get("mismatch_percent")
    )

    if percentage is not None:
        return percentage

    comparable = _get_comparable_count(item)
    mismatches = _get_mismatch_count(item)

    if comparable == 0:
        return None

    return 100.0 * mismatches / comparable


def _get_metric_name(item: dict) -> str:
    return str(
        item.get("display_name")
        or item.get("parameter_name")
        or item.get("parameter_id")
        or "Unknown parameter"
    )


def _get_asset_name(item: dict) -> str:
    return str(
        item.get("asset_name")
        or "Unknown asset type"
    )


def _get_scope(item: dict) -> str:
    return str(
        item.get("scope")
        or "unknown"
    ).lower()


def _get_delta_array(item: dict) -> np.ndarray | None:
    """
    Return the comparator's existing delta array.

    This deliberately does not recalculate differences.
    """

    delta = item.get("delta")

    if delta is None:
        return None

    try:
        return np.asarray(
            delta,
            dtype=float,
        ).reshape(-1)
    except (TypeError, ValueError):
        return None


def _get_mismatch_mask(item: dict) -> np.ndarray | None:
    """Return the comparator's existing mismatch mask."""

    mask = item.get("mismatch_mask")

    if mask is None:
        return None

    try:
        return np.asarray(
            mask,
            dtype=bool,
        ).reshape(-1)
    except (TypeError, ValueError):
        return None


def _get_asset_id(item: dict, index: int) -> str:
    """Get an asset identifier for a mismatch at the given index."""

    asset_ids = item.get("asset_ids")

    if asset_ids is not None:
        try:
            identifiers = np.asarray(
                asset_ids,
                dtype=object,
            ).reshape(-1)

            if index < len(identifiers):
                value = identifiers[index]

                if value is not None:
                    try:
                        if bool(np.asarray(value != value).item()):
                            return f"index {index}"
                    except (TypeError, ValueError):
                        pass

                    return str(value)

        except (TypeError, ValueError):
            pass

    return f"index {index}"


def _make_finding(
    category: str,
    message: str,
    scope: str,
    asset_name: str | None = None,
    parameter_name: str | None = None,
    evidence: dict | None = None,
) -> dict:
    """
    Return a consistent finding structure.

    The structure is intended for both report formatting and
    future Investigation Priorities integration.
    """

    return {
        "category": category,
        "message": message,
        "scope": scope,
        "asset_name": asset_name,
        "parameter_name": parameter_name,
        "evidence": evidence or {},
    }


# ============================================================
# Scenario 1
# Isolated versus widespread mismatches
# ============================================================

def _isolated_vs_widespread(item: dict) -> list[dict]:
    findings = []

    mismatches = _get_mismatch_count(item)
    comparable = _get_comparable_count(item)
    percentage = _get_mismatch_percent(item)

    if mismatches == 0 or comparable == 0 or percentage is None:
        return findings

    metric = _get_metric_name(item)
    asset = _get_asset_name(item)
    scope = _get_scope(item)

    if percentage <= 10.0:
        category = "isolated_mismatches"
        description = "isolated"

    elif percentage >= 50.0:
        category = "widespread_mismatches"
        description = "widespread"

    else:
        return findings

    findings.append(
        _make_finding(
            category=category,
            message=(
                f"{metric} has {description} mismatches: "
                f"{mismatches} of {comparable} compared values "
                f"({percentage:.2f}%)."
            ),
            scope=scope,
            asset_name=asset,
            parameter_name=metric,
            evidence={
                "mismatch_count": mismatches,
                "comparable_count": comparable,
                "mismatch_percent": percentage,
            },
        )
    )

    return findings


# ============================================================
# Scenario 2
# Dominant asset-type mismatch groups
# ============================================================

def _dominant_mismatch_groups(items: list[dict]) -> list[dict]:
    findings = []

    # Group by scope and asset type.
    groups = defaultdict(
        lambda: {
            "mismatch_count": 0,
            "comparable_count": 0,
        }
    )

    total_mismatches = 0

    for item in items:
        comparable = _get_comparable_count(item)
        mismatches = _get_mismatch_count(item)

        if comparable == 0:
            continue

        key = (
            _get_scope(item),
            _get_asset_name(item),
        )

        groups[key]["mismatch_count"] += mismatches
        groups[key]["comparable_count"] += comparable

        total_mismatches += mismatches

    if total_mismatches == 0:
        return findings

    for (scope, asset_name), group in groups.items():
        mismatches = group["mismatch_count"]
        comparable = group["comparable_count"]

        if mismatches == 0 or comparable == 0:
            continue

        share = 100.0 * mismatches / total_mismatches
        rate = 100.0 * mismatches / comparable

        # Report a group only when it accounts for at least half
        # of all metric-value mismatches in this scope.
        scope_mismatches = sum(
            value["mismatch_count"]
            for (group_scope, _), value in groups.items()
            if group_scope == scope
        )

        if scope_mismatches == 0:
            continue

        scope_share = 100.0 * mismatches / scope_mismatches

        if scope_share < 50.0:
            continue

        findings.append(
            _make_finding(
                category="dominant_asset_group",
                message=(
                    f"{asset_name} accounts for "
                    f"{scope_share:.1f}% of the recorded "
                    f"metric-value mismatches in the {scope} "
                    f"scope ({mismatches} mismatches across "
                    f"{comparable} compared values; "
                    f"mismatch rate {rate:.2f}%)."
                ),
                scope=scope,
                asset_name=asset_name,
                evidence={
                    "mismatch_count": mismatches,
                    "comparable_count": comparable,
                    "mismatch_percent": rate,
                    "share_of_scope_mismatches_percent": scope_share,
                },
            )
        )

    return findings


# ============================================================
# Scenario 3
# Parameter-specific mismatch patterns
# ============================================================

def _parameter_specific_patterns(items: list[dict]) -> list[dict]:
    findings = []

    groups = defaultdict(list)

    for item in items:
        if _get_comparable_count(item) == 0:
            continue

        key = (
            _get_scope(item),
            _get_asset_name(item),
        )

        groups[key].append(item)

    for (scope, asset_name), group_items in groups.items():
        if len(group_items) < 2:
            continue

        rates = []

        for item in group_items:
            rate = _get_mismatch_percent(item)

            if rate is not None:
                rates.append((rate, item))

        if len(rates) < 2:
            continue

        rates.sort(
            key=lambda entry: entry[0],
            reverse=True,
        )

        highest_rate, highest_item = rates[0]

        other_rates = [
            rate
            for rate, _ in rates[1:]
        ]

        other_average = float(np.mean(other_rates))

        # These are pattern-detection heuristics, not comparison
        # tolerances. A minimum 10 percentage-point gap is required.
        if highest_rate < 10.0:
            continue

        if highest_rate - other_average < 10.0:
            continue

        metric = _get_metric_name(highest_item)

        # findings.append(
        #     _make_finding(
        #         category="parameter_specific_pattern",
        #         message=(
        #             f"{metric} has a higher mismatch rate for "
        #             f"{asset_name} than the other compared "
        #             f"parameters: {highest_rate:.2f}% versus "
        #             f"{other_average:.2f}% average for the others."
        #         ),
        #         scope=scope,
        #         asset_name=asset_name,
        #         parameter_name=metric,
        #         evidence={
        #             "parameter_mismatch_percent": highest_rate,
        #             "other_parameters_average_percent": other_average,
        #         },
        #     )
        # )

    return findings



# ============================================================
# Scenario 4
# Relative-delta patterns
# ============================================================

def _relative_delta_patterns(item: dict) -> list[dict]:
    findings = []

    if _get_mismatch_count(item) == 0:
        return findings

    statistics = _get_statistics(item)

    median_relative = _finite_float(
        statistics.get("median_relative_delta")
    )

    p95_relative = _finite_float(
        statistics.get("p95_relative_delta")
    )

    max_relative = _finite_float(
        statistics.get("max_relative_delta")
    )

    metric = _get_metric_name(item)
    scope = _get_scope(item)
    asset = _get_asset_name(item)

    # These statistics describe the full valid population.
    # Matching values contribute 0%; undefined relative deltas
    # are excluded by the existing comparator's statistics helper.
    if (
        median_relative is not None
        and p95_relative is not None
        and median_relative > 0
        and p95_relative <= 1.5 * median_relative
    ):
        findings.append(
            _make_finding(
                category="consistent_relative_delta",
                message=(
                    f"{metric} has a relatively consistent relative-delta distribution: median "
                    f"{median_relative:.2f}% and P95 {p95_relative:.2f}%. This may indicate a systematic discrepancy "
                    f"pattern i.e. a common scaling or modeling issue rather than random variation."
                ),
                scope=scope,
                asset_name=asset,
                parameter_name=metric,
                evidence={
                    "median_relative_delta": median_relative,
                    "p95_relative_delta": p95_relative,
                },
            )
        )

    if (
        p95_relative is not None
        and max_relative is not None
        and p95_relative > 0
        and max_relative >= 3.0 * p95_relative
    ):
        findings.append(
            _make_finding(
                category="relative_delta_outlier",
                message=(
                    f"{metric} has a maximum relative delta of "
                    f"{max_relative:.2f}%, at least three times "
                    f"its P95 of {p95_relative:.2f}%. Investigate "
                    f"whether a small number of assets dominate "
                    f"the relative differences."
                ),
                scope=scope,
                asset_name=asset,
                parameter_name=metric,
                evidence={
                    "p95_relative_delta": p95_relative,
                    "max_relative_delta": max_relative,
                },
            )
        )

    return findings


# ============================================================
# Scenario 5
# Input discrepancy investigation signals
# ============================================================

def _input_discrepancy_signals(items: list[dict]) -> list[dict]:
    findings = []

    for item in items:
        mismatches = _get_mismatch_count(item)

        if mismatches == 0:
            continue

        metric = _get_metric_name(item)
        asset = _get_asset_name(item)
        percentage = _get_mismatch_percent(item)

        percentage_text = (
            f" ({percentage:.2f}% of compared values)"
            if percentage is not None
            else ""
        )

        findings.append(
            _make_finding(
                category="input_discrepancy",
                message=(
                    f"Investigate the {asset} input parameter "
                    f"{metric}: {mismatches} mismatches"
                    f"{percentage_text}. Resolve or explain this "
                    f"difference before attributing downstream "
                    f"power-flow differences to operating conditions."
                ),
                scope="model",
                asset_name=asset,
                parameter_name=metric,
                evidence={
                    "mismatch_count": mismatches,
                    "mismatch_percent": percentage,
                },
            )
        )

    return findings


# ============================================================
# Scenario 6
# Asset-level outliers
# ============================================================

def _asset_level_outliers(item: dict) -> list[dict]:
    findings = []

    delta = _get_delta_array(item)
    mismatch_mask = _get_mismatch_mask(item)

    if delta is None or mismatch_mask is None:
        return findings

    if len(delta) != len(mismatch_mask):
        return findings

    # Use only entries already classified as mismatches by
    # metadata_comparator.py.
    selected = mismatch_mask & np.isfinite(delta)
    indices = np.flatnonzero(selected)

    if indices.size == 0:
        return findings

    # Delta is the comparator's absolute difference for numeric
    # comparisons, so descending order ranks by discrepancy size.
    ranked_indices = indices[
        np.argsort(delta[indices])[::-1]
    ]

    metric = _get_metric_name(item)
    asset = _get_asset_name(item)
    scope = _get_scope(item)

    for index in ranked_indices[:3]:
        asset_id = _get_asset_id(item, int(index))
        magnitude = float(delta[index])

        findings.append(
            _make_finding(
                category="asset_level_outlier",
                message=(
                    f"{asset} asset {asset_id} has one of the "
                    f"largest mismatches for {metric}: "
                    f"{magnitude:.6g}."
                ),
                scope=scope,
                asset_name=asset,
                parameter_name=metric,
                evidence={
                    "asset_id": asset_id,
                    "delta": magnitude,
                    "array_index": int(index),
                },
            )
        )

    return findings


# ============================================================
# Scenario 7
# Overall comparison patterns
# ============================================================

def _overall_comparison_patterns(
    items: list[dict],
    scope: str,
) -> list[dict]:

    comparable = sum(
        _get_comparable_count(item)
        for item in items
    )

    mismatches = sum(
        _get_mismatch_count(item)
        for item in items
    )

    if comparable == 0:
        return []

    percentage = 100.0 * mismatches / comparable

    return [
        _make_finding(
            category="overall_comparison_pattern",
            message=(
                f"In the {scope} scope, {mismatches} of "
                f"{comparable} metric-value comparisons are "
                f"mismatches ({percentage:.2f}%)."
            ),
            scope=scope,
            evidence={
                "mismatch_count": mismatches,
                "comparable_count": comparable,
                "mismatch_percent": percentage,
            },
        )
    ]


# ============================================================
# Scenario 8
# Percentile-based discrepancy patterns
# ============================================================

def _percentile_patterns(item: dict) -> list[dict]:
    findings = []

    if _get_mismatch_count(item) == 0:
        return findings

    statistics = _get_statistics(item)

    median = _finite_float(
        statistics.get("median_delta")
    )

    p95 = _finite_float(
        statistics.get("p95_delta")
    )

    p99 = _finite_float(
        statistics.get("p99_delta")
    )

    maximum = _finite_float(
        statistics.get("max_delta")
    )

    if p95 is None or maximum is None:
        return findings

    metric = _get_metric_name(item)
    asset = _get_asset_name(item)
    scope = _get_scope(item)

    if p95 > 0 and maximum >= 3.0 * p95:
        category = "percentile_outlier_pattern"

        message = (
            f"{metric} has a maximum delta of {maximum:.6g}, "
            f"at least three times its P95 of {p95:.6g}. "
            f"The largest difference may be concentrated in "
            f"a small number of compared values."
        )

    elif p95 > 0:
        category = "percentile_distribution"

        message = (
            f"{metric} has a P95 delta of {p95:.6g} and a maximum delta of {maximum:.6g}. "
            f"This suggests that some assets have considerably larger differences than most of the dataset."
        )

    else:
        return findings

    findings.append(
        _make_finding(
            category=category,
            message=message,
            scope=scope,
            asset_name=asset,
            parameter_name=metric,
            evidence={
                "median_delta": median,
                "p95_delta": p95,
                "p99_delta": p99,
                "max_delta": maximum,
                "population": "all valid compared values",
            },
        )
    )

    return findings


# ============================================================
# Main statistical insight generator
# ============================================================

def generate_statistical_insights(
    model_comparison: Any,
    power_flow_comparison: Any = None,
    results_comparable: bool = True,
) -> dict[str, list[dict]]:
    """
    Generate statistical findings from existing comparison results.

    Does not run comparisons or alter mismatch decisions.

    Returns
    -------
    {
        "model": [...],
        "power_flow": [...],
        "overall": [...]
    }
    """

    model_items = _get_metadata_comparisons(
        model_comparison
    )

    power_flow_items = _get_metadata_comparisons(
        power_flow_comparison
    )

    findings = {
        "model": [],
        "power_flow": [],
        "overall": [],
    }

    # --------------------------------------------------------
    # Model/input statistical insights
    # --------------------------------------------------------

    for item in model_items:
        if _get_comparable_count(item) == 0:
            continue

        findings["model"].extend(
            _isolated_vs_widespread(item)
        )

        findings["model"].extend(
            _relative_delta_patterns(item)
        )

        findings["model"].extend(
            _asset_level_outliers(item)
        )

        findings["model"].extend(
            _percentile_patterns(item)
        )

    findings["model"].extend(
        _dominant_mismatch_groups(model_items)
    )

    findings["model"].extend(
        _parameter_specific_patterns(model_items)
    )

    findings["model"].extend(
        _input_discrepancy_signals(model_items)
    )

    findings["overall"].extend(
        _overall_comparison_patterns(
            model_items,
            "model",
        )
    )

    # --------------------------------------------------------
    # Power-flow statistical insights
    # --------------------------------------------------------

    if results_comparable and power_flow_comparison is not None:

        for item in power_flow_items:
            if _get_comparable_count(item) == 0:
                continue

            findings["power_flow"].extend(
                _isolated_vs_widespread(item)
            )

            findings["power_flow"].extend(
                _relative_delta_patterns(item)
            )

            findings["power_flow"].extend(
                _asset_level_outliers(item)
            )

            findings["power_flow"].extend(
                _percentile_patterns(item)
            )

        findings["power_flow"].extend(
            _dominant_mismatch_groups(power_flow_items)
        )

        findings["power_flow"].extend(
            _parameter_specific_patterns(power_flow_items)
        )

        findings["overall"].extend(
            _overall_comparison_patterns(
                power_flow_items,
                "power-flow",
            )
        )

    return findings
