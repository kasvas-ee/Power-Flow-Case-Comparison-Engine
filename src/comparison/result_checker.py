"""Metadata-driven solved-state comparison orchestration.

Per-asset comparison logic is exclusively metadata-driven. This module keeps
only orchestration, system-level aggregation, and report/insight adapters.
"""
from pathlib import Path

from src.comparison.metadata_adapters import run_metadata_comparisons, rank_metadata
from src.comparison.result_structures import ActionResult, PowerFlowComparisonResult
from src.comparison.state_checker import compare_system_metrics
from src.logger import logger


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


def results_match_from_metadata(results):
    """Return overall match based on enabled, performed metadata comparisons."""
    performed = [r for r in (results or []) if r.get("status") != "NOT PERFORMED"]
    return all(r.get("status") == "MATCH" for r in performed) and bool(performed)


def run_power_flow_comparison(net_a, net_b, config):
    """Run metadata state comparisons and independent system-level aggregation."""
    try:
        result = PowerFlowComparisonResult()
        metrics_path = Path(__file__).resolve().parents[2] / "config/metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metrics_path}")
        result.metadata_comparisons = run_metadata_comparisons(
            (net_a, net_b),
            (None, None),
            metrics_path,
            scope="state",
            report_all_mismatches=config.report_all_mismatches,
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
