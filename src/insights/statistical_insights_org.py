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

    if percentage <= 5.0:
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

        findings.append(
            _make_finding(
                category="parameter_specific_pattern",
                message=(
                    f"{metric} has a higher mismatch rate for "
                    f"{asset_name} than the other compared "
                    f"parameters: {highest_rate:.2f}% versus "
                    f"{other_average:.2f}% average for the others."
                ),
                scope=scope,
                asset_name=asset_name,
                parameter_name=metric,
                evidence={
                    "parameter_mismatch_percent": highest_rate,
                    "other_parameters_average_percent": other_average,
                },
            )
        )

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
                    f"{metric} has a relatively consistent "
                    f"relative-delta distribution: median "
                    f"{median_relative:.2f}% and P95 "
                    f"{p95_relative:.2f}%. This may indicate "
                    f"a systematic discrepancy pattern."
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
# Statistical distribution interpretation
# ============================================================

def _distribution_interpretations(item: dict) -> list[dict]:
    """
    Interpret the available comparison statistics together.

    The statistics are calculated by metadata_comparator.py. This
    helper does not recompute deltas or change match decisions.
    Distribution statistics describe all valid values; signed
    statistics describe mismatches only.
    """
    findings = []
    statistics = _get_statistics(item)

    comparable = _get_comparable_count(item)
    mismatches = _get_mismatch_count(item)
    if comparable == 0 or mismatches == 0:
        return findings

    metric = _get_metric_name(item)
    asset = _get_asset_name(item)
    scope = _get_scope(item)
    unit = str(item.get("unit") or "").strip()
    unit_text = f" {unit}" if unit else ""

    def stat(name: str) -> float | None:
        return _finite_float(statistics.get(name))

    def add(category: str, message: str, evidence: dict) -> None:
        findings.append(
            _make_finding(
                category=category,
                message=message,
                scope=scope,
                asset_name=asset,
                parameter_name=metric,
                evidence=evidence,
            )
        )

    median = stat("median_delta")
    mean = stat("mean_delta")
    minimum = stat("min_delta")
    p90 = stat("p90_delta")
    p95 = stat("p95_delta")
    p99 = stat("p99_delta")
    maximum = stat("max_delta")
    mad = stat("mad_delta")
    mismatch_percent = _get_mismatch_percent(item)

    median_rel = stat("median_relative_delta")
    mean_rel = stat("mean_relative_delta")
    p90_rel = stat("p90_relative_delta")
    p95_rel = stat("p95_relative_delta")
    p99_rel = stat("p99_relative_delta")
    max_rel = stat("max_relative_delta")
    mad_rel = stat("mad_relative_delta")

    tolerance = _finite_float(item.get("tolerance"))
    severe_threshold = _finite_float(item.get("severe_threshold"))
    severe_count = int(statistics.get("severe_mismatch_count", 0) or 0)

    positive_pct = stat("positive_delta_percent")
    negative_pct = stat("negative_delta_percent")
    median_signed = stat("median_signed_delta")
    mean_signed = stat("mean_signed_delta")

    affected_count = stat("affected_asset_count")
    asset_count = stat("asset_count")
    affected_pct = stat("affected_asset_percent")

    base_evidence = {
        "comparable_count": comparable,
        "mismatch_count": mismatches,
        "mismatch_percent": mismatch_percent,
        "mean_delta": mean,
        "min_delta": minimum,
        "median_delta": median,
        "p90_delta": p90,
        "p95_delta": p95,
        "p99_delta": p99,
        "max_delta": maximum,
        "mad_delta": mad,
        "tolerance": tolerance,
        "severe_threshold": severe_threshold,
        "severe_mismatch_count": severe_count,
        "unit": unit or None,
    }

    # 1. Fill the middle mismatch-prevalence band. The existing
    # _isolated_vs_widespread detector already handles <=5% and >=50%.
    if mismatch_percent is not None and 5.0 < mismatch_percent < 50.0:
        add(
            "mismatch_prevalence_moderate",
            (
                f"{mismatch_percent:.2f}% of compared {metric} values mismatch "
                f"({mismatches} of {comparable}). Compare the affected-asset list "
                f"with the delta distribution to establish whether the differences "
                f"cluster by asset or parameter."
            ),
            {
                "mismatch_count": mismatches,
                "comparable_count": comparable,
                "mismatch_percent": mismatch_percent,
            },
        )

    # 2. Put the delta distribution in context of the configured tolerance.
    if tolerance is not None and tolerance >= 0 and p95 is not None and maximum is not None:
        if p95 <= tolerance and maximum > tolerance:
            add(
                "tolerance_tail",
                (
                    f"For {metric}, the P95 absolute delta ({p95:.6g}{unit_text}) "
                    f"is at or below the configured tolerance ({tolerance:.6g}{unit_text}), "
                    f"but the maximum is {maximum:.6g}{unit_text}. This suggests "
                    f"the larger tolerance exceedances are in the upper tail; "
                    f"inspect the largest-delta assets."
                ),
                {**base_evidence, "interpretation": "upper_tail_exceeds_tolerance"},
            )
        elif p95 > tolerance:
            add(
                "tolerance_broad",
                (
                    f"For {metric}, the P95 absolute delta ({p95:.6g}{unit_text}) "
                    f"exceeds the configured tolerance ({tolerance:.6g}{unit_text}). "
                    f"The tolerance is exceeded within the upper portion of the "
                    f"distribution, not only by the single maximum. Review the "
                    f"mismatch rate and affected assets to assess the breadth."
                ),
                {**base_evidence, "interpretation": "p95_exceeds_tolerance"},
            )

    if severe_threshold is not None and severe_count > 0:
        add(
            "severe_mismatches",
            (
                f"{metric} has {severe_count} value(s) exceeding the configured "
                f"severe threshold of {severe_threshold:.6g}{unit_text}. "
                f"Prioritize these values for review, especially if they occur "
                f"on electrically important assets."
            ),
            {**base_evidence, "severe_mismatch_count": severe_count},
        )

    # 3. Compare central tendency with the upper tail.
    if median is not None and p95 is not None and p95 > 0:
        if median == 0:
            add(
                "zero_median_upper_tail",
                (
                    f"{metric} has a median absolute delta of zero, while its "
                    f"P95 is {p95:.6g}{unit_text}. At least half of the valid "
                    f"absolute deltas are zero, but the upper tail contains "
                    f"non-zero differences. Investigate the mismatching assets "
                    f"rather than assuming the whole population differs uniformly."
                ),
                {**base_evidence, "interpretation": "zero_median_nonzero_tail"},
            )
        elif p95 >= 3.0 * median:
            add(
                "upper_tail_vs_median",
                (
                    f"For {metric}, the P95 absolute delta ({p95:.6g}{unit_text}) "
                    f"is at least three times the median ({median:.6g}{unit_text}). "
                    f"The upper tail is much larger than the typical absolute "
                    f"difference; differences may be concentrated among a subset "
                    f"of values. Inspect the P95-to-maximum range and top assets."
                ),
                {**base_evidence, "p95_to_median_ratio": p95 / median},
            )
        elif p95 >= 1.5 * median:
            add(
                "upper_tail_moderate",
                (
                    f"For {metric}, the P95 ({p95:.6g}{unit_text}) is notably above "
                    f"the median ({median:.6g}{unit_text}), indicating that the "
                    f"larger differences exceed the typical difference. Review "
                    f"the upper-tail assets to see whether the pattern is localized."
                ),
                {**base_evidence, "p95_to_median_ratio": p95 / median},
            )

    # 4. P99/P95 and maximum/P95 relationships help identify extreme tails.
    if p95 is not None and p99 is not None and p95 > 0:
        ratio_99_95 = p99 / p95
        if ratio_99_95 >= 3.0:
            add(
                "extreme_upper_tail",
                (
                    f"For {metric}, P99 ({p99:.6g}{unit_text}) is at least three "
                    f"times P95 ({p95:.6g}{unit_text}). The most extreme part of "
                    f"the distribution is much larger than the broader upper tail; "
                    f"inspect the highest-delta assets for localized anomalies."
                ),
                {**base_evidence, "p99_to_p95_ratio": ratio_99_95},
            )
        elif ratio_99_95 >= 1.5:
            add(
                "upper_tail_growth",
                (
                    f"For {metric}, P99 ({p99:.6g}{unit_text}) is meaningfully "
                    f"above P95 ({p95:.6g}{unit_text}), showing that differences "
                    f"increase toward the extreme upper tail. Check whether the "
                    f"largest deltas share an asset type or operating context."
                ),
                {**base_evidence, "p99_to_p95_ratio": ratio_99_95},
            )

    if p95 is not None and maximum is not None and p95 > 0:
        ratio_max_95 = maximum / p95
        if ratio_max_95 >= 3.0:
            add(
                "maximum_isolated_candidate",
                (
                    f"For {metric}, the maximum absolute delta ({maximum:.6g}{unit_text}) "
                    f"is at least three times P95 ({p95:.6g}{unit_text}). The largest "
                    f"difference may be concentrated in a small number of compared "
                    f"values. Inspect the top-delta assets to verify whether this is "
                    f"an isolated deviation or part of a wider pattern."
                ),
                {**base_evidence, "max_to_p95_ratio": ratio_max_95},
            )
        elif ratio_max_95 >= 1.75:
            add(
                "maximum_above_p95",
                (
                    f"For {metric}, the maximum absolute delta ({maximum:.6g}{unit_text}) "
                    f"is about {ratio_max_95:.1f} times P95 ({p95:.6g}{unit_text}). "
                    f"The largest difference may be concentrated in a smaller number "
                    f"of compared values. Inspect the highest-delta assets to determine "
                    f"whether the maximum is isolated or part of a broader pattern."
                ),
                {**base_evidence, "max_to_p95_ratio": ratio_max_95},
            )
        elif ratio_max_95 >= 1.5:
            add(
                "maximum_above_p95",
                (
                    f"For {metric}, the maximum absolute delta ({maximum:.6g}{unit_text}) "
                    f"is above P95 ({p95:.6g}{unit_text}). The most discrepant value(s) "
                    f"are larger than the broader upper-tail level; inspect the top "
                    f"assets before deciding whether the issue is isolated."
                ),
                {**base_evidence, "max_to_p95_ratio": ratio_max_95},
            )

    # 5. Mean versus median: non-negative deltas with a much larger mean
    # are often influenced by a high-value tail.
    if mean is not None and median is not None and mean > 0:
        if median == 0 and mean > 0:
            add(
                "mean_median_skew",
                (
                    f"For {metric}, the mean absolute delta is {mean:.6g}{unit_text} "
                    f"while the median is zero. A subset of non-zero differences is "
                    f"pulling the average above the typical value; examine the upper "
                    f"tail and largest-delta assets."
                ),
                {**base_evidence, "mean_to_median_ratio": None},
            )
        elif median > 0 and mean >= 2.0 * median:
            add(
                "mean_median_skew",
                (
                    f"For {metric}, the mean absolute delta ({mean:.6g}{unit_text}) "
                    f"is at least twice the median ({median:.6g}{unit_text}). Larger "
                    f"differences may be pulling the average upward, so the median "
                    f"alone would understate the upper-tail behavior."
                ),
                {**base_evidence, "mean_to_median_ratio": mean / median},
            )

    # 6. MAD describes the typical distance from the median absolute delta.
    if mad is not None and median is not None:
        if median == 0 and mad == 0 and p95 is not None and p95 > 0:
            add(
                "mad_zero_with_tail",
                (
                    f"For {metric}, both the median absolute delta and MAD are zero, "
                    f"but P95 is {p95:.6g}{unit_text}. This can occur when at least "
                    f"half the observations are exact matches while a smaller portion "
                    f"has non-zero differences. Use percentiles and affected-asset "
                    f"statistics alongside MAD; MAD alone does not reveal the tail."
                ),
                {**base_evidence, "interpretation": "zero_mad_does_not_rule_out_tail"},
            )
        elif median > 0:
            mad_ratio = mad / median
            if mad_ratio <= 0.1:
                add(
                    "low_mad_relative_to_median",
                    (
                        f"For {metric}, MAD ({mad:.6g}{unit_text}) is small relative "
                        f"to the median absolute delta ({median:.6g}{unit_text}). The "
                        f"middle of the delta distribution is comparatively tightly "
                        f"clustered around its median, although the upper tail should "
                        f"still be checked separately."
                    ),
                    {**base_evidence, "mad_to_median_ratio": mad_ratio},
                )
            elif mad_ratio >= 0.5:
                add(
                    "high_mad_relative_to_median",
                    (
                        f"For {metric}, MAD ({mad:.6g}{unit_text}) is at least half "
                        f"the median absolute delta ({median:.6g}{unit_text}). Delta "
                        f"magnitudes vary substantially around their median; inspect "
                        f"whether the differences separate into distinct asset groups "
                        f"or operating patterns."
                    ),
                    {**base_evidence, "mad_to_median_ratio": mad_ratio},
                )

    # 7. Directionality: signed statistics are computed on mismatches only,
    # with the convention Case B - Case A.
    if positive_pct is not None and negative_pct is not None:
        direction_evidence = {
            **base_evidence,
            "mean_signed_delta": mean_signed,
            "median_signed_delta": median_signed,
            "positive_delta_percent": positive_pct,
            "negative_delta_percent": negative_pct,
            "signed_delta_convention": "Case B - Case A",
        }
        if positive_pct >= 80.0:
            add(
                "directional_positive",
                (
                    f"Among mismatching {metric} values, {positive_pct:.1f}% have "
                    f"positive signed deltas (Case B - Case A). Case B is higher for "
                    f"most mismatching observations. Check whether the direction is "
                    f"consistent with input changes or a systematic calculation difference."
                ),
                direction_evidence,
            )
        elif negative_pct >= 80.0:
            add(
                "directional_negative",
                (
                    f"Among mismatching {metric} values, {negative_pct:.1f}% have "
                    f"negative signed deltas (Case B - Case A). Case B is lower for "
                    f"most mismatching observations. Check whether the direction is "
                    f"consistent with input changes or a systematic calculation difference."
                ),
                direction_evidence,
            )
        elif positive_pct >= 60.0 or negative_pct >= 60.0:
            dominant = "positive" if positive_pct > negative_pct else "negative"
            share = max(positive_pct, negative_pct)
            add(
                "directional_tendency",
                (
                    f"Mismatching {metric} values show a {dominant} direction tendency: "
                    f"{share:.1f}% have that sign under the Case B - Case A convention. "
                    f"This may indicate a directional bias, but the mixed signs mean it "
                    f"is not uniform; compare the signed-delta median and asset locations."
                ),
                direction_evidence,
            )

    # 8. Relative deltas: interpret their spread separately because they are percentages.
    if p95_rel is not None and max_rel is not None and p95_rel > 0:
        ratio_rel = max_rel / p95_rel
        if ratio_rel >= 3.0:
            add(
                "relative_extreme_tail",
                (
                    f"For {metric}, the maximum relative delta ({max_rel:.4g}%) is at "
                    f"least three times the relative P95 ({p95_rel:.4g}%). Relative "
                    f"differences have an extreme upper tail; inspect assets with large "
                    f"relative deltas, especially where the underlying values are small."
                ),
                {
                    "median_relative_delta": median_rel,
                    "mean_relative_delta": mean_rel,
                    "p90_relative_delta": p90_rel,
                    "p95_relative_delta": p95_rel,
                    "p99_relative_delta": p99_rel,
                    "max_relative_delta": max_rel,
                    "mad_relative_delta": mad_rel,
                    "max_to_p95_relative_ratio": ratio_rel,
                },
            )

    if median_rel is not None and p95_rel is not None and median_rel > 0:
        rel_ratio = p95_rel / median_rel
        if rel_ratio >= 3.0:
            add(
                "relative_upper_tail",
                (
                    f"For {metric}, the relative P95 ({p95_rel:.4g}%) is at least three "
                    f"times the median relative delta ({median_rel:.4g}%). Proportional "
                    f"differences are much larger in the upper tail than for a typical "
                    f"observation; review the affected assets and their baseline values."
                ),
                {
                    "median_relative_delta": median_rel,
                    "p95_relative_delta": p95_rel,
                    "p95_to_median_relative_ratio": rel_ratio,
                },
            )

    if mad_rel is not None and median_rel is not None and median_rel > 0:
        mad_rel_ratio = mad_rel / median_rel
        if mad_rel_ratio >= 0.5:
            add(
                "relative_delta_dispersion",
                (
                    f"For {metric}, the relative-delta MAD ({mad_rel:.4g} percentage "
                    f"points) is large relative to its median ({median_rel:.4g}%). "
                    f"Proportional differences vary considerably; compare assets with "
                    f"large relative deltas to see whether small baseline values are involved."
                ),
                {
                    "median_relative_delta": median_rel,
                    "mad_relative_delta": mad_rel,
                    "mad_to_median_relative_ratio": mad_rel_ratio,
                },
            )

    # 9. Unique affected assets add context beyond metric-value mismatch counts.
    if (
        affected_count is not None
        and asset_count is not None
        and affected_pct is not None
        and asset_count > 0
    ):
        if affected_pct <= 5.0:
            text = (
                f"The mismatches affect {int(affected_count)} of {int(asset_count)} "
                f"{asset} assets ({affected_pct:.2f}%), suggesting the issue may be "
                f"localized to a small subset of assets."
            )
            category = "affected_assets_localized"
        elif affected_pct >= 50.0:
            text = (
                f"The mismatches affect {int(affected_count)} of {int(asset_count)} "
                f"{asset} assets ({affected_pct:.2f}%), indicating that the issue "
                f"extends across a large share of this asset group."
            )
            category = "affected_assets_widespread"
        else:
            text = (
                f"The mismatches affect {int(affected_count)} of {int(asset_count)} "
                f"{asset} assets ({affected_pct:.2f}%). Use the asset list to check "
                f"whether affected elements share a location or configuration."
            )
            category = "affected_assets_distribution"

        add(
            category,
            text,
            {
                "asset_count": int(asset_count),
                "affected_asset_count": int(affected_count),
                "affected_asset_percent": affected_pct,
            },
        )

    # 10. Small samples make tail percentiles less stable.
    if comparable < 20:
        add(
            "small_sample_caution",
            (
                f"The {metric} distribution is based on only {comparable} valid "
                f"compared value(s). Treat P90/P95/P99 and maximum-based pattern "
                f"interpretations cautiously because a small sample may not represent "
                f"a stable population pattern."
            ),
            {"comparable_count": comparable},
        )

    return findings



def _compact_findings(findings: list[dict]) -> list[dict]:
    """
    Present findings as one organized block per metric without discarding
    the underlying evidence. Asset outliers are combined into one short list;
    all other distinct interpretations remain visible as separate bullets.
    Original categories and evidence are retained in ``details`` for downstream
    consumers such as Investigation Priorities.
    """
    grouped: dict[tuple, list[dict]] = {}
    ungrouped: list[dict] = []

    for finding in findings:
        parameter = finding.get("parameter_name")
        scope = finding.get("scope") or "unknown"
        asset = finding.get("asset_name")
        if parameter:
            key = (scope, asset, parameter)
            grouped.setdefault(key, []).append(finding)
        else:
            ungrouped.append(finding)

    compacted: list[dict] = []
    for (scope, asset, parameter), entries in grouped.items():
        # Preserve original finding order while suppressing exact duplicate text.
        unique_entries = []
        seen_messages = set()
        for entry in entries:
            message = str(entry.get("message") or "").strip()
            if not message or message in seen_messages:
                continue
            seen_messages.add(message)
            unique_entries.append(entry)

        outlier_entries = [
            entry for entry in unique_entries
            if entry.get("category") == "asset_level_outlier"
        ]
        regular_entries = [
            entry for entry in unique_entries
            if entry.get("category") != "asset_level_outlier"
        ]

        lines = []
        if regular_entries:
            lines.append(f"{parameter} — statistical interpretation")
            for entry in regular_entries:
                lines.append(f"  • {entry['message']}")

        if outlier_entries:
            outlier_bits = []
            for entry in outlier_entries:
                evidence = entry.get("evidence") or {}
                asset_id = evidence.get("asset_id", "unknown")
                delta = _finite_float(evidence.get("delta"))
                if delta is None:
                    outlier_bits.append(f"{asset} {asset_id}")
                else:
                    outlier_bits.append(f"{asset} {asset_id} (Δ={delta:.6g})")
            heading = f"{parameter} — key exceptions" if not lines else "  Key exceptions"
            if lines:
                lines.append(heading)
                lines.append("  • Largest mismatches: " + "; ".join(outlier_bits) + ".")
            else:
                lines.extend([heading, "  • Largest mismatches: " + "; ".join(outlier_bits) + "."])

        categories = list(dict.fromkeys(
            entry.get("category") for entry in unique_entries if entry.get("category")
        ))
        compacted.append({
            "category": "metric_statistical_summary",
            "message": "\n".join(lines),
            "scope": scope,
            "asset_name": asset,
            "parameter_name": parameter,
            "evidence": {
                "categories": categories,
                "finding_count": len(unique_entries),
                "details": [
                    {
                        "category": entry.get("category"),
                        "message": entry.get("message"),
                        "evidence": entry.get("evidence") or {},
                    }
                    for entry in unique_entries
                ],
            },
        })

    # Overall/group-level findings remain separate because they describe a
    # scope or cross-metric pattern rather than one specific metric.
    return compacted + ungrouped

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
            _asset_level_outliers(item)
        )

        findings["model"].extend(
            _distribution_interpretations(item)
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
                _asset_level_outliers(item)
            )

            findings["power_flow"].extend(
                _distribution_interpretations(item)
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

    return {key: _compact_findings(value) for key, value in findings.items()}
