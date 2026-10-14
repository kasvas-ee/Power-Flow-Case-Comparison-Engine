"""Render system-level solved-state metrics and differences for reports."""

from src.comparison.result_structures import PowerFlowComparisonResult
from src.reporting.formatting import (
    format_bool,
    format_delta,
    format_value,
    subsection,
    REPORT_WIDTH,
    format_metadata_asset_group,
    format_ranked_results
)


def _format_extreme(value, bus_id, unit):
    """Format an extreme value and its bus ID on separate lines."""
    if value is None or bus_id is None:
        return f"N/A @ bus N/A"

    return f"{value:.4f} {unit} @ bus {bus_id}"


def _format_system_summary(system):
    """
    Format SystemSummary as a Case A / Case B / Delta table.
    """

    lines = []

    lines.extend(
        subsection("System Summary")
    )

    lines.append("")

    lines.append(
        f"{'Metric':<32}"
        f"{'Case A':>18}"
        f"{'Case B':>18}"
        f"{'Delta':>18}"
    )

    lines.append("-" * REPORT_WIDTH)

    rows = [
        (
            "Generation P (MW)",
            system.pgen_a_mw,
            system.pgen_b_mw,
            system.delta_pgen_mw,
        ),
        (
            "Generation Q (MVAr)",
            system.qgen_a_mvar,
            system.qgen_b_mvar,
            system.delta_qgen_mvar,
        ),
        (
            "Load P (MW)",
            system.pload_a_mw,
            system.pload_b_mw,
            system.delta_pload_mw,
        ),
        (
            "Load Q (MVAr)",
            system.qload_a_mvar,
            system.qload_b_mvar,
            system.delta_qload_mvar,
        ),
        (
            "Active losses (MW)",
            system.ploss_a_mw,
            system.ploss_b_mw,
            system.delta_ploss_mw,
        ),
        (
            "Reactive losses (MVAr)",
            system.qloss_a_mvar,
            system.qloss_b_mvar,
            system.delta_qloss_mvar,
        ),
        (
            "P balance residual (MW)",
            system.p_balance_a_mw,
            system.p_balance_b_mw,
            system.delta_p_balance_mw,
        ),
        (
            "Q balance residual (MVAr)",
            system.q_balance_a_mvar,
            system.q_balance_b_mvar,
            system.delta_q_balance_mvar,
        ),
    ]

    for name, value_a, value_b, delta in rows:

        lines.append(
            f"{name:<32}"
            f"{format_value(value_a):>18}"
            f"{format_value(value_b):>18}"
            f"{format_delta(delta):>18}"
        )

    voltage_extremes = [
        (
            "Voltage Magnitude Min",
            _format_extreme(
                system.vm_min_a,
                system.vm_min_bus_a,
                "p.u.",
            ),
            _format_extreme(
                system.vm_min_b,
                system.vm_min_bus_b,
                "p.u.",
            ),
        ),
        (
            "Voltage Magnitude Max",
            _format_extreme(
                system.vm_max_a,
                system.vm_max_bus_a,
                "p.u.",
            ),
            _format_extreme(
                system.vm_max_b,
                system.vm_max_bus_b,
                "p.u.",
            ),
        ),
        (
            "Voltage Angle Min",
            _format_extreme(
                system.va_min_a,
                system.va_min_bus_a,
                "deg",
            ),
            _format_extreme(
                system.va_min_b,
                system.va_min_bus_b,
                "deg",
            ),
        ),
        (
            "Voltage Angle Max",
            _format_extreme(
                system.va_max_a,
                system.va_max_bus_a,
                "deg",
            ),
            _format_extreme(
                system.va_max_b,
                system.va_max_bus_b,
                "deg",
            ),
        ),
    ]

    lines.append("")
    lines.append("Voltage Extremes")
    lines.append(
        f"{'Metric':<28}"
        f"{'Case A':^30}"
        f"{'Case B':^30}"
    )
    lines.append("-" * REPORT_WIDTH)

    for metric, case_a, case_b in voltage_extremes:
        lines.append(
            f"{metric:<28}"
            f"{case_a:^30}"
            f"{case_b:^30}"
        )

    lines.append("")

    return lines


def state_summary_report(
    power_flow_comparison: PowerFlowComparisonResult,
    results_comparable: bool,
    results_match,
    metadata_results=None,
):
    """
    Generate the solved power-flow state report.

    Reports:
        - overall result status
        - metadata-driven bus, generation, load, and branch metrics
        - system totals
        - ranked severe differences
    """

    lines = []

    # --------------------------------------------------------
    # Section header
    # --------------------------------------------------------

    lines.extend(
        [
            "=" * REPORT_WIDTH,
            "2. POWER-FLOW STATE SUMMARY",
            "=" * REPORT_WIDTH,
            "",
        ]
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    lines.extend(
        subsection("Overall Result Status")
    )

    lines.append(
        f"Results Comparable  : "
        f"{format_bool(results_comparable)}"
    )

    lines.append(
        f"Results Match       : "
        f"{format_bool(results_match)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Results not comparable
    # --------------------------------------------------------

    if not results_comparable:

        lines.append(
            "Power-flow results were not evaluated because "
            "the two cases are not comparable."
        )

        lines.append("")

        return "\n".join(lines)

    # --------------------------------------------------------
    # Metric groups
    # --------------------------------------------------------

    if metadata_results:
        from src.reporting.metadata_report_adapter import metadata_groups

        grouped = metadata_groups(metadata_results)
        asset_groups = [
            ("bus", "Bus"),
            ("generation", "Generation"),
            ("load", "Load"),
            ("branch", "Branch"),
        ]

        # Asset-level mismatch statistics from the state comparison.
        asset_summary = getattr(
            power_flow_comparison,
            "asset_summaries",
            {},
        ) or {}

        for asset_key, asset_label in asset_groups:
            lines.extend(
                format_metadata_asset_group(
                    grouped.get(asset_key, []),
                    asset_label,
                    metric_name_keys=("parameter_name", "display_name", "parameter_id"),
                    top_severe=power_flow_comparison.top_severe,
                    asset_summary=vars(asset_summary.get(asset_key))
                )
            )

    else:
        lines.append("No metrics available.")
        lines.append("")

    # --------------------------------------------------------
    # System summary
    # --------------------------------------------------------

    if power_flow_comparison.system is not None:

        lines.extend(
            _format_system_summary(
                power_flow_comparison.system
            )
        )


    return "\n".join(lines)
