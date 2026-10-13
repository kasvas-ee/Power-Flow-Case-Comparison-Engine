from src.comparison.result_structures import ActionResult

from src.comparison.result_structures import (
    ModelComparisonResult,
    PowerFlowComparisonResult,
    SystemSummary,
    TopSevereResult,
    YbusComparisonResult,
)


# ============================================================
# Formatting helpers
# ============================================================

def format_value(value, precision=4):
    """Format a numerical value for an insight message."""
    if value is None:
        return "N/A"

    return f"{value:.{precision}f}"


def _format_signed(value, precision=4):
    """Format a signed numerical value."""
    if value is None:
        return "N/A"

    return f"{value:+.{precision}f}"


def _format_percentage(value, precision=1):
    """Format a percentage."""
    if value is None:
        return "N/A"

    return f"{value:.{precision}f}%"


# ============================================================
# Generic comparison helpers
# ============================================================

def _mismatch_rate(comparison):
    """
    Calculate mismatch percentage from an existing ComparisonResult.

    Returns None when the comparison was not performed or no
    assets were compared.
    """

    if comparison is None:
        return None

    if comparison.compared_count == 0:
        return None

    return (
        comparison.mismatch_count
        / comparison.compared_count
        * 100.0
    )


def _severe_rate(comparison):
    """
    Calculate severe-difference percentage from an existing
    ComparisonResult.
    """

    if comparison is None:
        return None

    if comparison.compared_count == 0:
        return None

    return (
        comparison.severe_count
        / comparison.compared_count
        * 100.0
    )


def _is_small_difference(
    delta,
    reference,
    relative_tolerance=0.001,
):
    """
    Determine whether a system-level difference is small
    relative to the reference quantity.

    This is intentionally used only for system-level
    interpretation, not for the primary metric comparison.
    """

    if delta is None:
        return None

    if reference is None:
        return None

    reference = abs(reference)

    if reference == 0:
        return abs(delta) == 0

    return abs(delta) / reference <= relative_tolerance


def _top_result(top_severe, key):
    """
    Return the highest-ranked severe result for a metric.
    """

    results = top_severe.get(key, [])

    if not results:
        return None

    return results[0]


# ============================================================
# Network insights
# ============================================================

def _generate_model_investigation_priorities(model_comparison):
    """
    Prioritize model assets whose configured parameters do not match.
    """

    priorities = []

    if model_comparison is None or model_comparison.result is None:
        return priorities

    metadata = getattr(
        model_comparison.result,
        "metadata_comparisons",
        [],
    ) or []

    seen_assets = set()

    for item in metadata:
        status = str(item.get("status", "")).strip().upper()

        # Only mismatching model parameters are relevant here.
        if status not in {"MISMATCH", "SEVERE MISMATCH", "NO"}:
            continue

        asset_name = item.get("asset_name", "Unknown asset")
        asset_id = item.get("asset_id", "")
        parameter_name = (
            item.get("display_name")
            or item.get("parameter_name")
            or item.get("parameter_id")
            or "unknown parameter"
        )

        # Identify the particular asset, not just its asset category.
        asset_label = str(asset_name)

        if item.get("max_abs_diff_asset"):
            asset_label = item["max_abs_diff_asset"]
        elif asset_id:
            asset_label = f"{asset_name} ({asset_id})"

        # Report each asset once, even if multiple parameters mismatch.
        asset_key = asset_label.lower()

        if asset_key in seen_assets:
            continue

        seen_assets.add(asset_key)

        priorities.append(
            (
                asset_label,
                f"model parameter mismatch — {parameter_name} in {item.get('unit')}",
                item.get("statistics")["max_delta"],
            )
        )

    return priorities


def _generate_network_insights(
    ybus_comparison: ActionResult,
    model_comparison: ActionResult,
):
    """
    Generate insights about network equivalence and model
    differences.
    """

    insights = []

    if ybus_comparison is None:
        insights.append(
            "Ybus comparison was not performed, so network "
            "equivalence could not be established."
        )
        return insights

    # --------------------------------------------------------
    # Bus count
    # --------------------------------------------------------

    if not ybus_comparison.result.bus_count_match:

        insights.append(
            "The two cases contain different numbers of buses. "
            "The solved power-flow states are therefore not "
            "directly comparable."
        )

        return insights

    # --------------------------------------------------------
    # Topology
    # --------------------------------------------------------

    if not ybus_comparison.result.topology_match:

        insights.append(
            "The Ybus sparsity patterns differ, indicating a "
            "network topology mismatch. Differences in solved "
            "states cannot be attributed solely to operating "
            "conditions."
        )

        return insights

    insights.append(
        "Network topology is unchanged between the two cases."
    )

    # --------------------------------------------------------
    # Ybus numerical values
    # --------------------------------------------------------

    if ybus_comparison.result.ybus_value_match:

        insights.append(
            "Ybus numerical values also match within tolerance, "
            "indicating equivalent network admittance parameters."
        )

    else:

        insights.append(
            "Ybus topology matches, but numerical admittance "
            "values differ, indicating differences in electrical "
            "network parameters."
        )

        if ybus_comparison.result.max_value_diff is not None:

            insights.append(
                "The maximum Ybus element difference is "
                f"{format_value(ybus_comparison.result.max_value_diff)}."
            )

    # --------------------------------------------------------
    # Branch model
    # --------------------------------------------------------

    if model_comparison is not None:

        branch = model_comparison.result.branch

        if branch is not None:

            if branch.mismatch_count > 0:

                insights.append(
                    "Branch-model comparison identifies "
                    f"{branch.mismatch_asset_count} branches with "
                    "parameter differences."
                )

            else:

                insights.append(
                    "No branch-model parameter differences "
                    "were detected within the configured tolerance."
                )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        load = model_comparison.result.load

        if load is not None and load.mismatch_count > 0:

            insights.append(
                "Load-model differences are present and may "
                "contribute to differences in the solved "
                "operating state."
            )

        # ----------------------------------------------------
        # Generation model
        # ----------------------------------------------------

        generation = model_comparison.result.generation

        if generation is not None and generation.mismatch_count > 0:

            insights.append(
                "Generation-model differences are present and "
                "may contribute to differences in the solved "
                "operating state."
            )

    return insights


# ============================================================
# System insights
# ============================================================

def _generate_system_insights(
    system: SystemSummary,
):
    """
    Generate system-level power balance, generation, load,
    and loss insights.
    """

    insights = []

    if system is None:
        return insights

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    if system.delta_pgen_mw is not None:

        if abs(system.delta_pgen_mw) > 0:

            insights.append(
                "Total active generation changes from "
                f"{format_value(system.pgen_a_mw, 2)} MW to "
                f"{format_value(system.pgen_b_mw, 2)} MW "
                f"({ _format_signed(system.delta_pgen_mw, 2)} MW)."
            )

        else:

            insights.append(
                "Total active generation is unchanged between "
                "the two cases."
            )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if system.delta_pload_mw is not None:

        if abs(system.delta_pload_mw) > 0:

            insights.append(
                "Total active load changes from "
                f"{format_value(system.pload_a_mw, 2)} MW to "
                f"{format_value(system.pload_b_mw, 2)} MW "
                f"({ _format_signed(system.delta_pload_mw, 2)} MW)."
            )

        else:

            insights.append(
                "Total active load is unchanged between "
                "the two cases."
            )

    # --------------------------------------------------------
    # Reactive load
    # --------------------------------------------------------

    if system.delta_qload_mvar is not None:

        if abs(system.delta_qload_mvar) != 0:

            insights.append(
                "Total reactive load changes by "
                f"{_format_signed(system.delta_qload_mvar, 2)} MVAr."
            )

        else:

            insights.append(
                "Total reactive load is unchanged between "
                "the two cases."
            )

    # --------------------------------------------------------
    # Active losses
    # --------------------------------------------------------

    if system.delta_ploss_mw is not None:

        if abs(system.delta_ploss_mw) > 0:

            insights.append(
                "Active-power losses change from "
                f"{format_value(system.ploss_a_mw, 2)} MW to "
                f"{format_value(system.ploss_b_mw, 2)} MW "
                f"({ _format_signed(system.delta_ploss_mw, 2)} MW)."
            )

        else:

            insights.append(
                "Active-power losses are unchanged between "
                "the two cases."
            )

    # --------------------------------------------------------
    # Loss versus generation relationship
    # --------------------------------------------------------

    if (
        system.delta_pgen_mw is not None
        and system.delta_pload_mw is not None
        and system.delta_ploss_mw is not None
    ):

        expected_generation_change = (
            system.delta_pload_mw
            + system.delta_ploss_mw
        )

        residual = (
            system.delta_pgen_mw
            - expected_generation_change
        )

        if abs(residual) < 1e-6:

            if (
                abs(system.delta_pload_mw) < 1e-9
                and abs(system.delta_ploss_mw) > 0
            ):

                insights.append(
                    "The change in total generation closely "
                    "matches the change in active-power losses "
                    "while total load remains unchanged. This "
                    "indicates that the generation difference is "
                    "primarily associated with increased or "
                    "decreased network losses."
                )

    # --------------------------------------------------------
    # Reactive losses
    # --------------------------------------------------------

    if system.delta_qloss_mvar is not None:

        if abs(system.delta_qloss_mvar) > 0:

            insights.append(
                "Reactive-power losses change by "
                f"{_format_signed(system.delta_qloss_mvar, 2)} MVAr."
            )

    # --------------------------------------------------------
    # Active balance
    # --------------------------------------------------------

    if (
        system.p_balance_a_mw is not None
        and system.p_balance_b_mw is not None
    ):

        max_balance = max(
            abs(system.p_balance_a_mw),
            abs(system.p_balance_b_mw),
        )

        if max_balance < 1e-6:

            insights.append(
                "Both cases maintain active-power balance "
                "within numerical tolerance."
            )

        else:

            insights.append(
                "At least one case has a non-negligible "
                "active-power balance residual; this should be "
                "investigated before interpreting all result "
                "differences as physical operating-point changes."
            )

    # --------------------------------------------------------
    # Reactive balance
    # --------------------------------------------------------

    if (
        system.q_balance_a_mvar is not None
        and system.q_balance_b_mvar is not None
    ):

        max_balance = max(
            abs(system.q_balance_a_mvar),
            abs(system.q_balance_b_mvar),
        )

        if max_balance < 1e-6:

            insights.append(
                "Both cases maintain reactive-power balance "
                "within numerical tolerance."
            )

        else:

            insights.append(
                "At least one case has a non-negligible "
                "reactive-power balance residual."
            )

    return insights


# ============================================================
# Bus-voltage insights
# ============================================================

def _generate_voltage_insights(
    power_flow_comparison: PowerFlowComparisonResult,
):
    """
    Generate voltage-magnitude and voltage-angle insights.
    """

    insights = []

    if power_flow_comparison is None:
        return insights

    bus_metrics = power_flow_comparison.bus

    if not bus_metrics:
        return insights

    # --------------------------------------------------------
    # Voltage magnitude
    # --------------------------------------------------------

    vm = bus_metrics.get("vm")

    if vm is not None:

        mismatch_rate = _mismatch_rate(vm)
        severe_rate = _severe_rate(vm)

        if mismatch_rate is not None:

            insights.append(
                "Voltage magnitudes differ at "
                f"{vm.mismatch_count} of "
                f"{vm.compared_count} buses "
                f"({_format_percentage(mismatch_rate)})."
            )

        if severe_rate is not None:

            insights.append(
                f"{vm.severe_count} buses exceed the configured "
                f"voltage-magnitude severity threshold "
                f"({_format_percentage(severe_rate)} of compared buses)."
            )

        # ----------------------------------------------------
        # Localized versus widespread
        # ----------------------------------------------------

        if (
            vm.compared_count > 0
            and vm.severe_count > 0
        ):

            if severe_rate <= 5.0:

                insights.append(
                    "Voltage differences are concentrated in a "
                    "small number of buses, indicating a localized "
                    "voltage-state discrepancy."
                )

            else:

                insights.append(
                    "Voltage differences are widespread across "
                    "the network rather than being confined to "
                    "a small number of buses."
                )

        # ----------------------------------------------------
        # Largest voltage difference
        # ----------------------------------------------------

        if vm.max_abs_diff_asset is not None:

            insights.append(
                f"{vm.max_abs_diff_asset} has the largest "
                f"voltage-magnitude difference of "
                f"{format_value(vm.max_abs_diff)} p.u."
            )

        # ----------------------------------------------------
        # Direction of largest difference
        # ----------------------------------------------------

        top_vm = _top_result(
            power_flow_comparison.top_severe,
            "bus_vm",
        )

        if top_vm is not None:

            direction = (
                "higher"
                if top_vm.delta > 0
                else "lower"
            )

            insights.append(
                f"The voltage magnitude at {top_vm.asset} is "
                f"{format_value(abs(top_vm.delta))} p.u. "
                f"{direction} in Case B than in Case A."
            )

    # --------------------------------------------------------
    # Voltage angle
    # --------------------------------------------------------

    va = bus_metrics.get("va")

    if va is not None:

        mismatch_rate = _mismatch_rate(va)

        if mismatch_rate is not None:

            insights.append(
                "Voltage-angle differences occur at "
                f"{va.mismatch_count} of "
                f"{va.compared_count} buses "
                f"({_format_percentage(mismatch_rate)})."
            )

        if va.max_abs_diff_asset is not None:

            insights.append(
                f"{va.max_abs_diff_asset} has the largest "
                f"voltage-angle difference of "
                f"{format_value(va.max_abs_diff)} degrees."
            )

    return insights


# ============================================================
# Generation and load insights
# ============================================================

def _generate_generation_load_insights(
    power_flow_comparison: PowerFlowComparisonResult,
):
    """
    Generate insights about generator and load distributions.
    """

    insights = []

    if power_flow_comparison is None:
        return insights

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation = power_flow_comparison.generation

    if generation:

        for metric_name, comparison in generation.items():

            if comparison is None:
                continue

            mismatch_rate = _mismatch_rate(comparison)

            if mismatch_rate is not None:

                insights.append(
                    f"Generation metric '{metric_name}' differs "
                    f"for {comparison.mismatch_count} of "
                    f"{comparison.compared_count} compared "
                    f"generators ({_format_percentage(mismatch_rate)})."
                )

            top_result = _top_result(
                power_flow_comparison.top_severe,
                f"generation_{metric_name}",
            )

            if top_result is not None:

                insights.append(
                    f"{top_result.asset} has the largest "
                    f"generation '{metric_name}' difference "
                    f"of {format_value(top_result.abs_delta)}."
                )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    load = power_flow_comparison.load

    if load:

        for metric_name, comparison in load.items():

            if comparison is None:
                continue

            mismatch_rate = _mismatch_rate(comparison)

            if mismatch_rate is not None:

                insights.append(
                    f"Load metric '{metric_name}' differs for "
                    f"{comparison.mismatch_count} of "
                    f"{comparison.compared_count} compared "
                    f"loads ({_format_percentage(mismatch_rate)})."
                )

    return insights


# ============================================================
# Branch-flow insights
# ============================================================

def _generate_branch_insights(
    power_flow_comparison: PowerFlowComparisonResult,
):
    """
    Generate insights about branch flows and branch losses.
    """

    insights = []

    if power_flow_comparison is None:
        return insights

    branch = power_flow_comparison.branch

    if not branch:
        return insights

    # --------------------------------------------------------
    # Active power flow
    # --------------------------------------------------------

    for metric_name in (
        "p_from",
        "p_to",
    ):

        comparison = branch.get(metric_name)

        if comparison is None:
            continue

        mismatch_rate = _mismatch_rate(comparison)

        if mismatch_rate is not None:

            insights.append(
                f"Branch {metric_name} active-power flows differ "
                f"for {comparison.mismatch_count} of "
                f"{comparison.compared_count} branches "
                f"({_format_percentage(mismatch_rate)})."
            )

    # --------------------------------------------------------
    # Reactive power flow
    # --------------------------------------------------------

    for metric_name in (
        "q_from",
        "q_to",
    ):

        comparison = branch.get(metric_name)

        if comparison is None:
            continue

        mismatch_rate = _mismatch_rate(comparison)

        if mismatch_rate is not None:

            insights.append(
                f"Branch {metric_name} reactive-power flows differ "
                f"for {comparison.mismatch_count} of "
                f"{comparison.compared_count} branches "
                f"({_format_percentage(mismatch_rate)})."
            )

    # --------------------------------------------------------
    # Active losses
    # --------------------------------------------------------

    p_loss = branch.get("p_loss")

    if p_loss is not None:

        if p_loss.severe_count > 0:

            insights.append(
                f"{p_loss.severe_count} branches exceed the "
                "configured active-loss severity threshold."
            )

        top_result = _top_result(
            power_flow_comparison.top_severe,
            "branch_p_loss",
        )

        if top_result is not None:

            insights.append(
                f"{top_result.asset} has the largest active-loss "
                f"difference of "
                f"{format_value(top_result.abs_delta)} MW."
            )

    # --------------------------------------------------------
    # Reactive losses
    # --------------------------------------------------------

    q_loss = branch.get("q_loss")

    if q_loss is not None:

        if q_loss.severe_count > 0:

            insights.append(
                f"{q_loss.severe_count} branches exceed the "
                "configured reactive-loss severity threshold."
            )

        top_result = _top_result(
            power_flow_comparison.top_severe,
            "branch_q_loss",
        )

        if top_result is not None:

            insights.append(
                f"{top_result.asset} has the largest reactive-loss "
                f"difference of "
                f"{format_value(top_result.abs_delta)} MVAr."
            )

    # --------------------------------------------------------
    # Branch redistribution interpretation
    # --------------------------------------------------------

    p_from = branch.get("p_from")
    p_to = branch.get("p_to")

    if (
        p_from is not None
        and p_to is not None
        and p_from.severe_count > 0
        and p_to.severe_count > 0
    ):

        insights.append(
            "The observed branch-flow differences indicate "
            "redistribution of active power through the network "
            "rather than a difference confined to a single branch."
        )

    return insights


# ============================================================
# Investigation priorities
# ============================================================
def generate_investigation_priorities(
    power_flow_comparison: PowerFlowComparisonResult,
    model_comparison=None,
    model_match=True,
    statistical_insights=None,
):
    """
    Prioritize model mismatches before solved-state differences.

    Existing top-severe priorities remain the foundation.
    Statistical findings supplement them when they identify
    additional assets worth investigating.
    """

    # --------------------------------------------------------
    # Model mismatches take precedence
    # --------------------------------------------------------

    if not model_match:
        return _generate_model_investigation_priorities(
            model_comparison
        )

    priorities = []
    seen_assets = set()

    # --------------------------------------------------------
    # Existing top-severe priorities
    # --------------------------------------------------------

    if power_flow_comparison is not None:

        top_severe = power_flow_comparison.top_severe

        priority_keys = [
            ("bus_vm", "voltage-magnitude"),
            ("bus_va", "voltage-angle"),
            ("generation_p", "generation active-power"),
            ("generation_q", "generation reactive-power"),
            ("load_p", "load active-power"),
            ("load_q", "load reactive-power"),
            ("branch_p_from", "branch active-power flow"),
            ("branch_p_to", "branch active-power flow"),
            ("branch_q_from", "branch reactive-power flow"),
            ("branch_q_to", "branch reactive-power flow"),
            ("branch_p_loss", "branch active-power loss"),
            ("branch_q_loss", "branch reactive-power loss"),
        ]

        for metric_key, description in priority_keys:

            result = _top_result(top_severe, metric_key)

            if result is None:
                continue

            asset_key = str(result.asset).strip().casefold()

            if asset_key in seen_assets:
                continue

            seen_assets.add(asset_key)

            priorities.append(
                (
                    result.asset,
                    description,
                    result.abs_delta,
                )
            )

    # --------------------------------------------------------
    # Supplement with statistical findings
    # --------------------------------------------------------

    if not statistical_insights:
        return priorities

    for section_name in ("overall", "model", "power_flow"):

        findings = statistical_insights.get(section_name, []) or []

        for finding in findings:

            if not isinstance(finding, dict):
                continue

            asset = (
                finding.get("asset")
                or finding.get("asset_name")
            )

            # Keep network-wide findings in the Statistical
            # Insights section; priorities need a specific asset.
            if not asset:
                continue

            asset_key = str(asset).strip().casefold()

            if asset_key in seen_assets:
                continue

            description = (
                finding.get("description")
                or finding.get("message")
                or finding.get("title")
                or "statistical comparison finding"
            )

            delta = finding.get(
                "max_delta",
                finding.get("max_abs_delta"),
            )

            priorities.append(
                (
                    str(asset),
                    str(description),
                    delta,
                )
            )

            seen_assets.add(asset_key)

    return priorities


# ============================================================
# Main insight generator
# ============================================================

def generate_insights(
    ybus_comparison: ActionResult,
    model_comparison: ActionResult,
    power_flow_comparison: PowerFlowComparisonResult,
    model_match: bool,
    results_comparable: bool,
    results_match,
):
    """
    Generate structured power-system insights.

    The function does not calculate or modify comparison metrics.
    It interprets the already-computed comparison results.

    Returns
    -------
    dict
        Structured insights suitable for TXT/JSON/report output.
    """

    result = {
        "network": [],
        "system": [],
        "voltage": [],
        "generation_load": [],
        "branch_flows": [],
    }

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    result["network"] = _generate_network_insights(
        ybus_comparison=ybus_comparison,
        model_comparison=model_comparison,
    )

    # --------------------------------------------------------
    # Stop state interpretation if results aren't comparable
    # --------------------------------------------------------
    if results_match:
        result["system"].append("Solved power-flow results were are an exact match.")

        return result

    if not results_comparable:

        result["system"].append(
            "Solved power-flow results were not interpreted "
            "because the cases are not comparable."
        )

        return result

    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

    if power_flow_comparison is not None:

        result["system"] = _generate_system_insights(
            system=power_flow_comparison.system,
        )

        # ----------------------------------------------------
        # Voltage
        # ----------------------------------------------------

        result["voltage"] = _generate_voltage_insights(
            power_flow_comparison=power_flow_comparison,
        )

        # ----------------------------------------------------
        # Generation / load
        # ----------------------------------------------------

        result["generation_load"] = (
            _generate_generation_load_insights(
                power_flow_comparison=power_flow_comparison,
            )
        )

        # ----------------------------------------------------
        # Branches
        # ----------------------------------------------------

        result["branch_flows"] = (
            _generate_branch_insights(
                power_flow_comparison=power_flow_comparison,
            )
        )

    return result
