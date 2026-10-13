from src.comparison.result_structures import ActionResult
from src.reporting.metadata_report_adapter import metadata_groups
from src.reporting.formatting import (
    format_bool,
    format_value,
    format_count,
    section,
    subsection,
    format_metadata_asset_group,
)


def model_summary_report(
    ybus_comparison: ActionResult,
    model_comparison: ActionResult,
    model_match: bool,
    results_comparable: bool,
    metadata_results=None,
):
    """
    Generate the model/network comparison report section.

    Reports:
        - overall model status
        - Ybus comparison
        - bus comparison
        - branch comparison
        - load comparison
        - generation comparison

    Returns
    -------
    str
        Formatted model summary.
    """

    lines = []

    # --------------------------------------------------------
    # Section header
    # --------------------------------------------------------

    lines.extend(
        section("1. MODEL / NETWORK SUMMARY")
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    lines.extend(
        subsection("Overall Status")
    )

    lines.append(
        f"Model Match         : "
        f"{format_bool(model_match)}"
    )

    lines.append(
        f"Results Comparable  : "
        f"{format_bool(results_comparable)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Asset count comparison
    # --------------------------------------------------------

    model_result = getattr(model_comparison, "result", None)

    asset_summaries = {
        "Bus": getattr(model_result, "bus", None),
        "Branch": getattr(model_result, "branch", None),
        "Generation": getattr(model_result, "generation", None),
        "Load": getattr(model_result, "load", None),
    }

    header = "Asset Count Summary"
    header = header + ": Not computed" if asset_summaries['Load'] is None else header
    lines.extend(subsection(header))
    lines.append("")

    if asset_summaries['Bus']:
        lines.append(
            f"{'Asset':<24}"
            f"{'Case A':>16}"
            f"{'Case B':>16}"
        )

        lines.append("-" * 56)

        for asset in ("Bus", "Branch", "Generation", "Load"):
            summary = asset_summaries[asset]

            count_a = getattr(summary, "asset_count_a", "N/A")
            count_b = getattr(summary, "asset_count_b", "N/A")

            lines.append(
                f"{asset:<24}"
                f"{str(count_a):>16}"
                f"{str(count_b):>16}"
            )

    lines.append("")

    # --------------------------------------------------------
    # Ybus comparison
    # --------------------------------------------------------

    lines.extend(
        subsection("Ybus Comparison")
    )

    if ybus_comparison is None:

        lines.append(
            "Ybus comparison     : NOT PERFORMED"
        )

        lines.append("")

    elif getattr(ybus_comparison, "result", None) is None:

        lines.append("Ybus comparison     : NOT PERFORMED")
        lines.append("")

    else:

        lines.append(
            f"Ybus Match          : "
            f"{format_bool(ybus_comparison.result.is_match)}"
        )

        lines.append(
            f"Bus Count Match     : "
            f"{format_bool(ybus_comparison.result.bus_count_match)}"
        )

        lines.append(
            f"Topology Match      : "
            f"{format_bool(ybus_comparison.result.topology_match)}"
        )

        lines.append(
            f"Ybus Value Match    : "
            f"{format_bool(ybus_comparison.result.ybus_value_match)}"
        )

        lines.append(
            f"Pattern Differences : "
            f"{format_count(ybus_comparison.result.pattern_diff_count)}"
        )

        lines.append(
            f"Value Differences   : "
            f"{format_count(ybus_comparison.result.value_diff_count)}"
        )

        lines.append(
            f"Maximum Ybus |delta|: "
            f"{format_value(ybus_comparison.result.max_value_diff)}"
        )

        lines.append(f"Unit                : pu")

        lines.append(
            f"Bus count in Case A : "
            f"{ybus_comparison.result.shape_a[0] or 'N/A'}"
        )

        lines.append(
            f"Bus count in Case B : "
            f"{ybus_comparison.result.shape_b[0] or 'N/A'}"
        )

        if ybus_comparison.result.reason:
            lines.append(
                f"Reason              : "
                f"{ybus_comparison.result.reason}"
            )

        lines.append("")

    # --------------------------------------------------------
    # Detailed model comparisons
    # --------------------------------------------------------

    lines.extend(
        subsection("Detailed Model Comparisons")
    )

    lines.append("")

    if metadata_results:
        groups = metadata_groups(metadata_results)

        asset_order = [
            ("bus", "Bus"),
            ("branch", "Branch"),
            ("load", "Load"),
            ("generation", "Generation"),
            ("grid", "Grid"),
        ]

        model_result = getattr(model_comparison, "result", None)
        top_severe = getattr(model_result, "top_severe", {}) or {}

        # Asset-level mismatch statistics from the model comparison.
        model_asset_summaries = {}

        if model_result is not None:
            for asset_label in ("Bus", "Branch", "Generation", "Load", "Grid"):
                asset_result = getattr(
                    model_result,
                    asset_label.lower(),
                    None,
                )

                if asset_result is not None:
                    model_asset_summaries[asset_label] = {
                        "asset_count_a": getattr(
                            asset_result, "asset_count_a", None
                        ),
                        "asset_count_b": getattr(
                            asset_result, "asset_count_b", None
                        ),
                        "mismatch_asset_count": getattr(
                            asset_result, "mismatch_asset_count", None
                        ),
                        "mismatch_asset_percentage": getattr(
                            asset_result, "mismatch_asset_percentage", None
                        ),
                    }

        for asset_key, asset_label in asset_order:
            entries = groups.get(asset_key, [])

            if not entries:
                continue

            lines.extend(
                format_metadata_asset_group(
                    entries,
                    asset_label,
                    metric_name_keys=("parameter_name", "display_name", "parameter_id"),
                    top_severe=top_severe,
                    asset_summary=model_asset_summaries.get(asset_label),
                )
            )

    else:
        lines.append("No enabled model metadata metrics were configured.")
        lines.append("")

    return "\n".join(lines)
