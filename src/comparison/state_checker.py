"""System-level aggregate calculations retained alongside metadata comparisons.

Per-asset numerical comparisons live exclusively in metadata_comparator.py.
"""
import numpy as np
from src.comparison.metadata_adapters import run_metadata_comparisons, rank_metadata
from src.comparison.result_structures import ActionResult, PowerFlowComparisonResult, SystemSummary
from src.logger import logger


def _voltage_extremes(net):
    """
    Calculate voltage magnitude and angle extrema using vectorized
    NumPy operations. No Python-level iteration over buses.
    """

    res_bus = net.res_bus

    vm = res_bus["vm_pu"].to_numpy(copy=False)
    va = res_bus["va_degree"].to_numpy(copy=False)

    bus_ids = np.asarray(res_bus.index)

    def extrema(values):
        valid = np.isfinite(values)

        if not valid.any():
            return None, None, None, None

        valid_positions = np.flatnonzero(valid)
        valid_values = values[valid]

        min_position = valid_positions[np.argmin(valid_values)]
        max_position = valid_positions[np.argmax(valid_values)]

        return (
            float(values[min_position]),
            bus_ids[min_position].item()
            if hasattr(bus_ids[min_position], "item")
            else bus_ids[min_position],
            float(values[max_position]),
            bus_ids[max_position].item()
            if hasattr(bus_ids[max_position], "item")
            else bus_ids[max_position],
        )

    vm_min, vm_min_bus, vm_max, vm_max_bus = extrema(vm)
    va_min, va_min_bus, va_max, va_max_bus = extrema(va)

    return {
        "vm_min": vm_min,
        "vm_min_bus": vm_min_bus,
        "vm_max": vm_max,
        "vm_max_bus": vm_max_bus,
        "va_min": va_min,
        "va_min_bus": va_min_bus,
        "va_max": va_max,
        "va_max_bus": va_max_bus,
    }


def _calculate_generation(net):
    """
    Return total active/reactive generation.
    """

    pgen = (
        net.res_gen.p_mw.sum()
        + net.res_ext_grid.p_mw.sum()
    )

    qgen = (
        net.res_gen.q_mvar.sum()
        + net.res_ext_grid.q_mvar.sum()
    )

    return float(pgen), float(qgen)


def _calculate_load(net):
    """
    Return total active/reactive load.
    """

    pload = net.res_load.p_mw.sum()
    qload = net.res_load.q_mvar.sum()

    return float(pload), float(qload)


def _calculate_branch_losses(net):
    """
    Return explicit branch/impedance losses calculated from
    the solved branch flows.
    """

    branch = net.res_impedance
    p_loss = branch.p_from_mw.to_numpy(copy=False) + branch.p_to_mw.to_numpy(copy=False)
    q_loss = branch.q_from_mvar.to_numpy(copy=False) + branch.q_to_mvar.to_numpy(copy=False)

    return (
        float(np.nansum(p_loss)),
        float(np.nansum(q_loss)),
    )


def compare_system_metrics(net_a, net_b):
    """
    Calculate system-level solved-state quantities.

    Explicit branch losses are calculated independently from
    generation/load balance.

    The balance residual is:

        generation - load - explicit branch loss

    This gives us a useful power-flow consistency check.
    """

    # ---------------------------------------------------------
    # Case A
    # ---------------------------------------------------------

    pgen_a, qgen_a = _calculate_generation(net_a)
    pload_a, qload_a = _calculate_load(net_a)

    ploss_a, qloss_a = _calculate_branch_losses(net_a)

    p_balance_a = (
        pgen_a
        - pload_a
        - ploss_a
    )

    q_balance_a = (
        qgen_a
        - qload_a
        - qloss_a
    )

    vm_extremes_a = _voltage_extremes(net_a)

    # ---------------------------------------------------------
    # Case B
    # ---------------------------------------------------------

    pgen_b, qgen_b = _calculate_generation(net_b)
    pload_b, qload_b = _calculate_load(net_b)

    ploss_b, qloss_b = _calculate_branch_losses(net_b)

    p_balance_b = (
        pgen_b
        - pload_b
        - ploss_b
    )

    q_balance_b = (
        qgen_b
        - qload_b
        - qloss_b
    )

    vm_extremes_b = _voltage_extremes(net_b)

    # ---------------------------------------------------------
    # Differences
    # ---------------------------------------------------------

    return SystemSummary(
        pgen_a_mw=pgen_a,
        pgen_b_mw=pgen_b,
        delta_pgen_mw=pgen_b - pgen_a,

        qgen_a_mvar=qgen_a,
        qgen_b_mvar=qgen_b,
        delta_qgen_mvar=qgen_b - qgen_a,

        pload_a_mw=pload_a,
        pload_b_mw=pload_b,
        delta_pload_mw=pload_b - pload_a,

        qload_a_mvar=qload_a,
        qload_b_mvar=qload_b,
        delta_qload_mvar=qload_b - qload_a,

        ploss_a_mw=ploss_a,
        ploss_b_mw=ploss_b,
        delta_ploss_mw=ploss_b - ploss_a,

        qloss_a_mvar=qloss_a,
        qloss_b_mvar=qloss_b,
        delta_qloss_mvar=qloss_b - qloss_a,

        p_balance_a_mw=p_balance_a,
        p_balance_b_mw=p_balance_b,
        delta_p_balance_mw=(
            p_balance_b - p_balance_a
        ),

        q_balance_a_mvar=q_balance_a,
        q_balance_b_mvar=q_balance_b,
        delta_q_balance_mvar=(
            q_balance_b - q_balance_a
        ),

        vm_min_a=vm_extremes_a["vm_min"],
        vm_min_bus_a=vm_extremes_a["vm_min_bus"],
        vm_max_a=vm_extremes_a["vm_max"],
        vm_max_bus_a=vm_extremes_a["vm_max_bus"],

        vm_min_b=vm_extremes_b["vm_min"],
        vm_min_bus_b=vm_extremes_b["vm_min_bus"],
        vm_max_b=vm_extremes_b["vm_max"],
        vm_max_bus_b=vm_extremes_b["vm_max_bus"],

        va_min_a=vm_extremes_a["va_min"],
        va_min_bus_a=vm_extremes_a["va_min_bus"],
        va_max_a=vm_extremes_a["va_max"],
        va_max_bus_a=vm_extremes_a["va_max_bus"],

        va_min_b=vm_extremes_b["va_min"],
        va_min_bus_b=vm_extremes_b["va_min_bus"],
        va_max_b=vm_extremes_b["va_max"],
        va_max_bus_b=vm_extremes_b["va_max_bus"],
    )


def _metadata_summary(item):
    """Convert metadata statistics to the small result shape used by insights."""
    from src.reporting.metadata_report_adapter import metadata_result_for_report
    return metadata_result_for_report(item)


def _populate_insight_views(result):
    """Build compatibility views from metadata results without recomparing values."""
    aliases = {
        "bus": {"vm_pu": "vm", "vm": "vm", "va_degree": "va", "va": "va"},
        "generation": {"p_mw": "p", "q_mvar": "q", "p": "p", "q": "q", "p_share": "p_share"},
        "load": {"p_mw": "p", "q_mvar": "q", "p": "p", "q": "q"},
        "branch": {"p_from_mw": "p_from", "q_from_mvar": "q_from", "p_to_mw": "p_to", "q_to_mvar": "q_to",
                    "p_from": "p_from", "q_from": "q_from", "p_to": "p_to", "q_to": "q_to",
                    "p_loss_mw": "p_loss", "q_loss_mvar": "q_loss", "p_loss_percent": "p_loss_pct", "q_loss_percent": "q_loss_pct",
                    "p_loss": "p_loss", "q_loss": "q_loss", "p_loss_pct": "p_loss_pct", "q_loss_pct": "q_loss_pct",
                    "s_from_mva": "s_from", "s_to_mva": "s_to", "s_from": "s_from", "s_to": "s_to"},
    }
    views = {"bus": {}, "generation": {}, "load": {}, "branch": {}}
    for item in result.metadata_comparisons:
        asset = str(item.get("asset_name", "")).lower()
        parameter = str(item.get("parameter_id") or "").lower()
        group = "generation" if asset in {"gen", "generation", "grid"} else asset
        if group not in views:
            continue
        key = aliases.get(group, {}).get(parameter, parameter)
        if item.get("status") != "NOT PERFORMED":
            views[group][key] = _metadata_summary(item)
    result.bus, result.generation, result.load, result.branch = (
        views["bus"], views["generation"], views["load"], views["branch"]
    )


def run_power_flow_comparison(net_a, net_b, config):
    """Run metadata state comparisons and independent system-level aggregation."""
    try:
        result = PowerFlowComparisonResult()
        result.metadata_comparisons = run_metadata_comparisons(
            (net_a, net_b),
            (None, None),
            metrics=config.metric_definitions,
            scope="state",
            report_all_mismatches=config.report_all_mismatches,
            enable_multiprocessing=config.enable_multiprocessing
        )
        # System summary is an aggregate component, not a duplicate per-asset comparator.
        result.system = compare_system_metrics(net_a, net_b)
        result.top_severe = rank_metadata(result.metadata_comparisons, int(getattr(config, "top_n_severe", 5)))
        _populate_insight_views(result)
        return ActionResult(action="Power Flow Comparison", success=True, result=result)
    except Exception as exc:
        logger.exception("Metadata-driven state comparison failed")
        return ActionResult(action="Power Flow Comparison", success=False,
                            error_type=type(exc).__name__, error_reason=str(exc))
