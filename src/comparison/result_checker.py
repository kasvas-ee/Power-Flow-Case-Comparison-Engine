"""Metadata-driven solved-state comparison orchestration.

Per-asset comparison logic is exclusively metadata-driven. This module keeps
only orchestration, system-level aggregation, and report/insight adapters.
"""
from types import SimpleNamespace
from src.comparison.model_checker import count_mismatching_assets, get_asset_counts


def populate_state_asset_summaries(result, asset_counts_a, asset_counts_b):
    """
    Populate per-asset-type counts and unique mismatch statistics
    for solved-state comparisons.

    Mismatch statistics are skipped when the asset counts differ.
    """
    asset_types = {
        "bus": "Bus",
        "branch": "Branch",
        "load": "Load",
        "generation": "Generation",
    }

    for asset_name, count_key in asset_types.items():
        count_a = asset_counts_a.get(count_key, 0)
        count_b = asset_counts_b.get(count_key, 0)
        reason = None
        if count_a != count_b:
            mismatch_asset_count = None
            mismatch_asset_percentage = None
            reason="Mismatch in asset count"

        else:
            mismatch_asset_count = count_mismatching_assets(
                result.metadata_comparisons,
                asset_name,
            )

            # Distinguish a valid zero-mismatch result from missing
            # per-asset comparison data.
            if mismatch_asset_count is None:
                performed_results = [
                    item
                    for item in result.metadata_comparisons
                    if str(item.get("asset_name", "")).strip().lower()
                    == asset_name
                    and str(item.get("status", "")).strip().upper()
                    != "NOT PERFORMED"
                    and item.get("asset_ids") is not None
                    and item.get("mismatch_mask") is not None
                ]

                if performed_results:
                    mismatch_asset_count = 0

            if mismatch_asset_count is not None:
                mismatch_asset_percentage = (
                    100.0 * mismatch_asset_count / count_a
                    if count_a > 0
                    else 0.0
                )
            else:
                mismatch_asset_percentage = None

        result.asset_summaries[asset_name] = SimpleNamespace(
            mismatch_asset_count=mismatch_asset_count,
            mismatch_asset_percentage=mismatch_asset_percentage,
            reason=reason
        )


def results_match_from_metadata(results):
    """Return overall match based on enabled, performed metadata comparisons."""
    performed = [r for r in (results or []) if r.get("status") != "NOT PERFORMED"]
    return all(r.get("status") == "MATCH" for r in performed) and bool(performed)
