# #####################################################################
# Check whether the network models are identical or comparable
# #####################################################################

import numpy as np
import pandapower as pp
from pathlib import Path
from types import SimpleNamespace

from pandapower.pypower.makeYbus import makeYbus

from src.comparison.ybus_comparator import compare_ybus
from src.comparison.metadata_adapters import run_metadata_comparisons, rank_metadata
from src.comparison.result_structures import (
    ActionResult,
    ModelComparisonResult,
    ValidationResult,
)
from src.logger import logger


def convert_to_ppc(net):
    """
    Convert a canonical pandapower network to PPC format.
    """

    action = "PPC conversion"

    try:
        ppc = pp.converter.to_ppc(net)

        return ActionResult(
            action=action,
            success=True,
            result=ppc,
        )

    except Exception as exc:
        return ActionResult(
            action=action,
            success=False,
            error_type=type(exc).__name__,
            error_reason=str(exc),
        )


def compute_ybus_matric(ppc):
    """
    Compute the Ybus matrix for a PPC case.

    The branch matrix is padded only when required by makeYbus.
    The original PPC branch matrix is not modified.
    """

    action = "Ybus creation"

    try:
        branch = ppc["branch"]

        required_columns = 26

        if branch.shape[1] < required_columns:
            branch = np.pad(
                branch,
                (
                    (0, 0),
                    (0, required_columns - branch.shape[1]),
                ),
            )

        Ybus, _, _ = makeYbus(
            ppc["baseMVA"],
            ppc["bus"],
            branch,
        )

        return ActionResult(
            action=action,
            success=True,
            result=Ybus,
        )

    except Exception as exc:
        return ActionResult(
            action=action,
            success=False,
            error_type=type(exc).__name__,
            error_reason=str(exc),
        )


def count_mismatching_assets(metadata_comparisons, asset_name):
    """
    Count unique assets with at least one parameter mismatch.

    The raw metadata comparison dictionaries contain asset_ids and
    mismatch_mask. Multiple parameter comparisons can refer to the
    same asset, so asset IDs are deduplicated before counting.

    Returns None when no usable per-asset comparison data is available.
    """
    import numpy as np
    import pandas as pd

    mismatching_ids = []

    for item in metadata_comparisons:
        if str(item.get("asset_name", "")).strip().lower() != asset_name:
            continue

        if str(item.get("status", "")).strip().upper() == "NOT PERFORMED":
            continue

        asset_ids = item.get("asset_ids")
        mismatch_mask = item.get("mismatch_mask")

        if asset_ids is None or mismatch_mask is None:
            continue

        ids = np.asarray(asset_ids).reshape(-1)
        mask = np.asarray(mismatch_mask, dtype=bool).reshape(-1)

        if len(ids) != len(mask):
            continue

        # Select all mismatching IDs at once; no per-asset Python loop.
        affected_ids = ids[mask]

        # Ignore missing IDs.
        if affected_ids.size:
            affected_ids = affected_ids[~pd.isna(affected_ids)]

            if affected_ids.size:
                mismatching_ids.append(affected_ids)

    if not mismatching_ids:
        return None

    # Deduplicate assets that mismatch on multiple parameters.
    unique_ids = np.unique(np.concatenate(mismatching_ids))

    return int(unique_ids.size)


def _populate_model_summary_from_metadata(result, asset_counts_a, asset_counts_b):
    """
    Create report/insight summaries from metadata results and attach
    unique asset-level mismatch statistics.
    """
    from src.reporting.metadata_report_adapter import metadata_result_for_report

    by_asset = {}

    for item in result.metadata_comparisons:
        key = str(item.get("asset_name", "")).strip().lower()

        by_asset.setdefault(key, {})[
            str(item.get("parameter_id") or item.get("parameter_name"))
        ] = metadata_result_for_report(item)

    asset_groups = {
        "bus": "Bus",
        "branch": "Branch",
        "load": "Load",
        "generation": "Generation",
    }

    for summary_name, count_key in asset_groups.items():
        metrics = by_asset.get(summary_name, {})

        count_a = asset_counts_a.get(count_key, 0)
        count_b = asset_counts_b.get(count_key, 0)

        if count_a != count_b:
            mismatch_asset_count = None
        else:
            mismatch_asset_count = count_mismatching_assets(
                result.metadata_comparisons,
                summary_name,
            )

            if mismatch_asset_count is None:
                performed_results = [
                    item
                    for item in result.metadata_comparisons
                    if str(item.get("asset_name", "")).strip().lower()
                    == summary_name
                    and str(item.get("status", "")).strip().upper()
                    != "NOT PERFORMED"
                    and item.get("asset_ids") is not None
                    and item.get("mismatch_mask") is not None
                ]

                if performed_results:
                    mismatch_asset_count = 0

        summary = _combine_metadata_group(
            metrics=metrics,
            asset_count_a=count_a,
            asset_count_b=count_b,
            mismatch_asset_count=mismatch_asset_count,
        )

        setattr(result, summary_name, summary)

    return


def _combine_metadata_group(
    metrics,
    asset_count_a,
    asset_count_b,
    mismatch_asset_count,
):
    """
    Combine parameter summaries and attach asset-level statistics.

    Asset-level mismatch statistics are not relevant when Case A and
    Case B contain different numbers of assets of this type.
    """
    from types import SimpleNamespace

    counts_match = asset_count_a == asset_count_b

    if counts_match:
        mismatch_percentage = (
            100.0 * mismatch_asset_count / asset_count_a
            if asset_count_a > 0 and mismatch_asset_count is not None
            else (
                0.0
                if asset_count_a == 0 and mismatch_asset_count == 0
                else None
            )
        )
    else:
        mismatch_asset_count = None
        mismatch_percentage = None

    return SimpleNamespace(
        is_match=(
            all(getattr(v, "is_match", False) for v in metrics.values())
            if metrics
            else None
        ),
        compared_count=sum(
            getattr(v, "compared_count", 0)
            for v in metrics.values()
        ),
        mismatch_count=sum(
            getattr(v, "mismatch_count", 0)
            for v in metrics.values()
        ),
        severe_count=sum(
            getattr(v, "severe_count", 0)
            for v in metrics.values()
        ),
        metrics=metrics,

        # Asset-level statistics
        asset_count_a=asset_count_a,
        asset_count_b=asset_count_b,
        mismatch_asset_count=mismatch_asset_count,
        mismatch_asset_percentage=mismatch_percentage,
    )


def get_asset_counts(ppc):
    """Return asset counts without iterating through individual assets."""

    def table_count(table_name, non_zero_column_ids=[]):
        table = ppc.get(table_name)
        if table is None:
            return 0
        if non_zero_column_ids:
            return np.count_nonzero(np.any(table[:, non_zero_column_ids] != 0, axis=1))
        else:
            return len(table)

    return {
        "Bus": table_count("bus"),
        "Branch": table_count("branch"),
        "Generation": table_count("gen"),
        "Load": table_count("bus", [2, 3]),
    }


def match_model(case_a, case_b, config=None):
    """
    Perform the model-comparison pipeline.

    Pipeline:

        pandapower net
            ↓
        PPC conversion
            ↓
        Ybus construction
            ↓
        Ybus comparison
            ↓
        detailed model comparison
    """
    results_comparability = ValidationResult(False)  # Placeholder to capture state comparability and reason if False

    cases = {
        "Case A": case_a,
        "Case B": case_b,
    }

    ppcs = {}
    ybus = {}

    # ---------------------------------------------------------
    # 1. Convert both cases to PPC and generate YBus matrix.
    # ---------------------------------------------------------

    for case_name, case in cases.items():

        ppcs[case_name] = convert_to_ppc(case)

        if not ppcs[case_name].success:
            logger.error(
                "%s PPC conversion failed: %s",
                case_name,
                ppcs[case_name].error_reason,
            )

            ybus[case_name] = ActionResult(
                action="Ybus creation",
                success=False,
                error_type="PrerequisiteError",
                error_reason="PPC conversion failed; Ybus cannot be constructed.",
            )

            continue

        # Generate Ybus matrix
        ybus[case_name] = compute_ybus_matric(ppcs[case_name].result)

    # ---------------------------------------------------------
    # 2. Ybus comparison.
    # ---------------------------------------------------------
    ybus_comparison = ActionResult(action="Ybus Comparison")
    if ybus["Case A"].success and ybus["Case B"].success:
        ybus_comparison.result = compare_ybus(ybus["Case A"].result, ybus["Case B"].result, tol=config.y_pu_tolerance)
        ybus_comparison.success = True
    else:
        failed_cases = []

        if not ppcs["Case A"].success : failed_cases.append("Case A PPC conversion")
        if not ppcs["Case B"].success : failed_cases.append("Case B PPC conversion")
        if not ybus["Case A"].success : failed_cases.append("Case A Ybus construction")
        if not ybus["Case B"].success : failed_cases.append("Case B Ybus construction")

        ybus_comparison.success=False
        ybus_comparison.error_type = "PrerequisiteError"
        ybus_comparison.error_reason = ("Ybus comparison could not be performed. "
                                        "Failed prerequisites: " + ", ".join(failed_cases))
    logger.info(ybus_comparison)

    # ---------------------------------------------------------
    # 3. Detailed assetwise model comparison.
    # ---------------------------------------------------------
    model_comparison = ActionResult(action="Model Comparison")
    model_comparison.result = ModelComparisonResult()

    # Get asset counts
    asset_counts_a = get_asset_counts(ppcs["Case A"].result)
    asset_counts_b = get_asset_counts(ppcs["Case B"].result)
    counts_match = asset_counts_a==asset_counts_b
    _populate_model_summary_from_metadata(model_comparison.result, asset_counts_a, asset_counts_b)

    # Check if all pre-checks have passed
    reason_prefix = "Precheck Failed: Model comparison cannot proceed because "
    if ybus_comparison.success and counts_match:
        if ybus_comparison.result.topology_match and ybus_comparison.result.bus_count_match:
            model_comparison.success = True
        else:
            model_comparison.success = False
            model_comparison.error_type = "PrerequisiteError"
            reason = "number of buses in the system do not match" if not ybus_comparison.result.bus_count_match \
                else "the Ybus sparsity patterns differ, indicating a network topology mismatch."
            model_comparison.error_reason = reason_prefix + reason
            results_comparability.is_valid = False
            results_comparability.reason = reason_prefix + reason
    else:
        model_comparison.success=False
        model_comparison.error_type = "PrerequisiteError"
        reason = "Ybus comparison was not completed." if not ybus_comparison.success else "assets count do not match."
        model_comparison.error_reason = reason_prefix + reason
        results_comparability.is_valid = False
        results_comparability.reason = reason_prefix + reason

    # Initiate comparison of asset parameters
    if not model_comparison.success:
        results_comparability.is_valid = False
        reason = "Results are not comparable: the model-comparison stage did not complete."
        results_comparability.reason = reason if results_comparability.reason is None else results_comparability.reason
    else:  # Initiate comparison of asset parameters
        # If Ybus has matched, the bus and branch data are bound to match.
        skip_assets = ({"bus", "branch"} if ybus_comparison.result.is_match else {})
        try:
            model_comparison.result.metadata_comparisons = run_metadata_comparisons(
                (case_a, case_b),
                (ppcs["Case A"].result, ppcs["Case B"].result),
                metrics=config.metric_definitions,
                scope="model",
                skip_assets=skip_assets,
                report_all_mismatches=config.report_all_mismatches,
                enable_multiprocessing=config.enable_multiprocessing
            )

            model_comparison.result.top_severe = rank_metadata(
                model_comparison.result.metadata_comparisons,
                config.top_n_severe,
            )

            _populate_model_summary_from_metadata(model_comparison.result, asset_counts_a, asset_counts_b)

            logger.info(
                "Comparison of all assets in the model is completed: %d parameter comparisons",
                len(model_comparison.result.metadata_comparisons),
            )
        except Exception as exc:
            # Diagnostics are deliberately non-blocking during this integration step.
            logger.warning("Metadata model diagnostics were not performed: %s", exc)
            results_comparability.is_valid = False
            results_comparability.reason = "The model-comparison stage did not complete."

        # logger.info(model_comparison)
        results_comparability.is_valid = True
        results_comparability.reason = "Power-flow results are comparable: Asset counts and network topology match."

    return (
        ppcs,
        ybus,
        ybus_comparison,
        model_comparison,
        results_comparability,
        asset_counts_a,
        asset_counts_b,
    )


def determine_model_match(ybus_comparison, model_comparison):
    """Ybus remains the primary equivalence test; enabled model metadata adds diagnostics."""
    if ybus_comparison is None or not ybus_comparison.success:
        return False
    if model_comparison is None or not model_comparison.success:
        return False
    if not ybus_comparison.result.is_match:
        return False
    metadata = getattr(model_comparison.result, "metadata_comparisons", []) or []
    performed = [r for r in metadata if r.get("status") != "NOT PERFORMED"]
    return all(r.get("status") == "MATCH" for r in performed)
