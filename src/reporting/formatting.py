"""Shared text-formatting helpers for comparison reports."""
from __future__ import annotations

from src.reporting.metadata_report_adapter import (
    asset_summary_status,
    metadata_result_for_report,
)
from src.comparison.metadata_adapters import PARAMETER_ALIASES

REPORT_WIDTH = 100
ASSET_RULE_WIDTH = REPORT_WIDTH // 2


def format_bool(value):
    """Format a boolean-like value as YES, NO, or N/A."""
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "N/A"


def format_value(value, precision=6):
    """Format a finite numeric value, or N/A when unavailable."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number != number or number in (float("inf"), float("-inf")):
        return "N/A"
    return f"{number:.{precision}f}"


def format_delta(value, precision=6):
    """Format a signed numerical difference."""
    if value is None:
        return "N/A"

    return f"{value:+.{precision}f}"


def format_count(value):
    """Format a count, or N/A when unavailable."""
    return "N/A" if value is None else str(value)


def section(title):
    """Create a major report section."""
    return ["=" * REPORT_WIDTH, title, "=" * REPORT_WIDTH, ""]


def subsection(title):
    """Create a report subsection."""
    return [title, "-" * REPORT_WIDTH]


def format_ranked_results(
    title,
    ranked_results,
):
    """
    Format TopSevereResult objects into a ranked table.
    """

    lines = []

    lines.extend(
        subsection(title)
    )

    if not ranked_results:

        lines.append(
            "No severe differences found."
        )

        lines.append("")

        return lines

    lines.append(
        f"{'Rank':<8}"
        f"{'Asset':<24}"
        f"{'Case A':>16}"
        f"{'Case B':>16}"
        f"{'Delta':>16}"
        f"{'|Delta|':>16}"
    )

    lines.append("-" * REPORT_WIDTH)

    for result in ranked_results:

        lines.append(
            f"{result.rank:<8}"
            f"{result.asset:<24}"
            f"{format_value(result.value_a):>16}"
            f"{format_value(result.value_b):>16}"
            f"{format_delta(result.delta):>16}"
            f"{format_value(result.abs_delta):>16}"
        )

    lines.append("-" * REPORT_WIDTH)
    lines.append("")

    return lines


def format_comparison_result(metric_name, comparison_result):
    """Format an already-computed metric comparison for a text report."""
    lines = [f"Metric              : {metric_name}"]

    if comparison_result is None:
        lines.extend(["Status              : NOT PERFORMED", ""])
        return lines

    status_text = str(getattr(comparison_result, "status", "") or "").upper()
    if status_text in {"NOT PERFORMED", "SKIPPED"}:
        display_status = "NOT PERFORMED"
        is_match = None
    elif status_text in {"MATCH", "YES"}:
        display_status = "YES"
        is_match = True
    elif status_text in {"MISMATCH", "SEVERE MISMATCH", "NO"}:
        display_status = "NO"
        is_match = False
    else:
        is_match = getattr(comparison_result, "is_match", None)
        display_status = format_bool(is_match)

    lines.append(f"Match               : {display_status}")

    metric = getattr(comparison_result, "metric", None)
    unit = getattr(comparison_result, "unit", None)
    if metric:
        lines.append(f"Description         : {metric}")
    if unit:
        lines.append(f"Unit                : {unit}")

    for label, attr in (
        ("Compared count", "compared_count"),
        ("Mismatch count", "mismatch_count"),
        ("Severe count", "severe_count"),
    ):
        lines.append(f"{label:<20}: {format_count(getattr(comparison_result, attr, None))}")

    lines.append(f"{'Tolerance':<20}: {format_value(getattr(comparison_result, 'tolerance', None))}")
    lines.append(
        f"{'Severity threshold':<20}: "
        f"{format_value(getattr(comparison_result, 'severity_threshold', None))}"
    )

    if is_match is not True:
        # Maximum absolute delta, its asset, and its relative delta.
        max_abs = getattr(
            comparison_result,
            "max_abs_diff",
            None,
        )
        max_asset = getattr(
            comparison_result,
            "max_abs_diff_asset",
            None,
        )
        relative_at_max_abs = getattr(
            comparison_result,
            "max_abs_diff_relative_delta",
            None,
        )

        max_abs_line = (
            f"{'Max |delta|':<20}: {format_value(max_abs)}"
        )

        if max_asset is not None:
            max_abs_line += f" on {max_asset}"

        if relative_at_max_abs is not None:
            max_abs_line += (
                " (relative delta: "
                f"{format_value(relative_at_max_abs, precision=2)}%)"
            )

        lines.append(max_abs_line)

        # Maximum relative delta, its asset, and its absolute delta.
        max_relative = getattr(
            comparison_result,
            "max_relative_diff",
            None,
        )
        max_relative_asset = getattr(
            comparison_result,
            "max_relative_diff_asset",
            None,
        )
        max_relative_abs_delta = getattr(
            comparison_result,
            "max_relative_diff_abs_delta",
            None,
        )

        max_relative_line = (
            f"{'Max relative delta':<20}: "
            f"{format_value(max_relative, precision=2)}%"
        )

        if max_relative_asset is not None:
            max_relative_line += f" on {max_relative_asset}"

        if max_relative_abs_delta is not None:
            max_relative_line += (
                " (|delta|: "
                f"{format_value(max_relative_abs_delta)})"
            )

        lines.append(max_relative_line)

        # Print remaining statistics
        statistics = (
            ("Mean |delta|", "mean_abs_diff"),
            ("RMS delta", "rms_diff"),
            ("P50 |delta|", "p50_abs_diff"),
            ("P90 |delta|", "p90_abs_diff"),
            ("P95 |delta|", "p95_abs_diff"),
            ("P99 |delta|", "p99_abs_diff"),
        )
        for label, attr in statistics:
            lines.append(f"{label:<20}: {format_value(getattr(comparison_result, attr, None))}")

    reason = getattr(comparison_result, "reason", None)
    if reason:
        lines.append(f"{'Reason':<20}: {reason}")

    lines.append("")
    return lines


def format_metadata_asset_group(
    entries,
    asset_label,
    metric_name_keys,
    top_severe=None,
    asset_summary=[]
):
    """Render each metric and its severe rows immediately after its summary."""
    if not entries:
        return []

    top_severe = top_severe or {}

    status = asset_summary_status(entries)
    status = asset_summary_status(entries)
    asset_summary = asset_summary or {}

    mismatch_asset_count = asset_summary.get("mismatch_asset_count")
    mismatch_asset_percentage = asset_summary.get(
        "mismatch_asset_percentage"
    )

    lines = [
        "-" * ASSET_RULE_WIDTH,
        f"Asset               : {asset_label}",
        f"Status              : {status}",
    ]

    if mismatch_asset_count is not None:
        lines.append(
            f"Mismatch Asset Count: {mismatch_asset_count}"
        )

    if mismatch_asset_percentage is not None:
        lines.append(
            f"Mismatch Asset %    : {mismatch_asset_percentage:.2f}%"
        )

    lines.extend([
        "-" * ASSET_RULE_WIDTH,
        "",
    ])

    if status == "Skipped because Ybus matched.":
        return lines

    for entry in entries:
        metric_name = next(
            (entry.get(key) for key in metric_name_keys if entry.get(key)),
            asset_label,
        )

        # First print the normal metric comparison.
        lines.extend(
            format_comparison_result(
                metric_name,
                metadata_result_for_report(entry),
            )
        )

        # Print this metric's ranked severe differences, if any.
        asset_name = str(entry.get("asset_name", "Asset"))
        raw_parameter = str(entry.get("parameter_id") or entry.get("parameter_name")).lower()

        key = f"{asset_name}_{PARAMETER_ALIASES.get(raw_parameter, raw_parameter)}"
        ranked_rows = top_severe.get(key, [])

        if ranked_rows:
            title = f"Severe Differences (Top {len(ranked_rows)})"
            lines.extend(
                format_ranked_results(
                    title=title,
                    ranked_results=ranked_rows,
                )
            )

    return lines
