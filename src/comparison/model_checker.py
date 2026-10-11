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


def compare_models(ppcs, ybus_result):
    """Create the model-comparison stage after the primary Ybus test.

    Asset-level model checks are metadata-driven; this function only enforces
    the Ybus prerequisites and returns a container for those results.
    PPC conversion, Ybus construction, and Ybus comparison remain unchanged.
    """
    result = ModelComparisonResult()
    if not ybus_result.bus_count_match:
        return ActionResult(
            action="Model Comparison", success=False, result=result,
            error_type="PrerequisiteError",
            error_reason="Number of buses differs between cases.",
        )
    if not ybus_result.topology_match:
        return ActionResult(
            action="Model Comparison", success=False, result=result,
            error_type="PrerequisiteError",
            error_reason="Network topology differs between cases.",
        )
    return ActionResult(action="Model Comparison", success=True, result=result)


def _populate_model_summary_from_metadata(result):
    """Create report/insight summary objects from metadata results (no re-comparison)."""
    from src.reporting.metadata_report_adapter import metadata_result_for_report
    from types import SimpleNamespace
    by_asset = {}
    for item in result.metadata_comparisons:
        key = str(item.get("asset_name", "")).lower()
        by_asset.setdefault(key, {})[str(item.get("parameter_id") or item.get("parameter_name"))] = metadata_result_for_report(item)
    result.bus = _combine_metadata_group(by_asset.get("bus", {}))
    result.branch = _combine_metadata_group(by_asset.get("branch", {}))
    result.load = _combine_metadata_group(by_asset.get("load", {}))
    result.generation = _combine_metadata_group(by_asset.get("generation", {}))


def _combine_metadata_group(metrics):
    if not metrics:
        return None
    # Preserve individual metadata results; the report uses raw metadata records.
    return SimpleNamespace(
        is_match=all(getattr(v, "is_match", False) for v in metrics.values()),
        compared_count=sum(getattr(v, "compared_count", 0) for v in metrics.values()),
        mismatch_count=sum(getattr(v, "mismatch_count", 0) for v in metrics.values()),
        severe_count=sum(getattr(v, "severe_count", 0) for v in metrics.values()),
        metrics=metrics,
    )


def _get_asset_counts(net):
    """Return asset counts without iterating through individual assets."""

    def table_count(table_name):
        table = net.get(table_name)
        return 0 if table is None else len(table)

    return {
        "Bus": table_count("bus"),
        "Branch": table_count("impedance"),
        "Generation": (
            table_count("gen")
            + table_count("ext_grid")
        ),
        "Load": table_count("load"),
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

    cases = {
        "Case A": case_a,
        "Case B": case_b,
    }

    ppcs = {}
    ybus = {}

    # ---------------------------------------------------------
    # 1. Convert both cases to PPC.
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
                error_reason=(
                    "PPC conversion failed; "
                    "Ybus cannot be constructed."
                ),
            )

            continue

        ybus[case_name] = compute_ybus_matric(
            ppcs[case_name].result
        )

    # ---------------------------------------------------------
    # 2. Ybus comparison.
    # ---------------------------------------------------------

    if (
        ppcs["Case A"].success
        and ppcs["Case B"].success
        and ybus["Case A"].success
        and ybus["Case B"].success
    ):

        ybus_comparison_result = compare_ybus(
            ybus["Case A"].result,
            ybus["Case B"].result,
            tol=1e-6,
        )

        ybus_comparison = ActionResult(
            action="Ybus Comparison",
            success=True,
            result=ybus_comparison_result,
        )

    else:

        failed_cases = []

        if not ppcs["Case A"].success:
            failed_cases.append("Case A PPC conversion")

        if not ppcs["Case B"].success:
            failed_cases.append("Case B PPC conversion")

        if not ybus["Case A"].success:
            failed_cases.append("Case A Ybus construction")

        if not ybus["Case B"].success:
            failed_cases.append("Case B Ybus construction")

        ybus_comparison = ActionResult(
            action="Ybus Comparison",
            success=False,
            error_type="PrerequisiteError",
            error_reason=(
                "Ybus comparison could not be performed. "
                "Failed prerequisites: "
                + ", ".join(failed_cases)
            ),
        )

    logger.info(ybus_comparison)

    # ---------------------------------------------------------
    # 3. Detailed model comparison.
    # ---------------------------------------------------------

    if not ybus_comparison.success:

        model_comparison = ActionResult(
            action="Model Comparison",
            success=False,
            result=ModelComparisonResult(),
            error_type="PrerequisiteError",
            error_reason=(
                "Model comparison cannot proceed because "
                "Ybus comparison was not completed."
            ),
        )

    else:

        model_comparison = compare_models(
            [
                ppcs["Case A"].result,
                ppcs["Case B"].result,
            ],
            ybus_comparison.result,
        )

    model_comparison.result.asset_counts_a = _get_asset_counts(case_a)
    model_comparison.result.asset_counts_b = _get_asset_counts(case_b)

    metrics_path = Path(__file__).resolve().parents[2] / "config/metrics.csv"
    if model_comparison.success and metrics_path.exists():
        try:
            skip_assets = (
                {"bus", "branch"}
                if ybus_comparison.result.is_match
                else set()
            )

            model_comparison.result.metadata_comparisons = run_metadata_comparisons(
                (case_a, case_b),
                (ppcs["Case A"].result, ppcs["Case B"].result),
                metrics_path,
                scope="model",
                skip_assets=skip_assets,
                report_all_mismatches=config.report_all_mismatches,
            )

            model_comparison.result.top_severe = rank_metadata(
                model_comparison.result.metadata_comparisons,
                int(getattr(config, "top_n_severe", 5)),
            )

            _populate_model_summary_from_metadata(model_comparison.result)
            logger.info(
                "Metadata model comparisons completed: %d parameter comparisons",
                len(model_comparison.result.metadata_comparisons),
            )
        except Exception as exc:
            # Diagnostics are deliberately non-blocking during this integration step.
            logger.warning("Metadata model diagnostics were not performed: %s", exc)

    return (
        ppcs,
        ybus,
        ybus_comparison,
        model_comparison,
    )


def determine_results_comparability(ybus_comparison, model_comparison):
    """Results are comparable when Ybus dimensions/topology align."""
    if ybus_comparison is None or not ybus_comparison.success:
        return ValidationResult(False, "Results are not comparable: Ybus comparison was not performed.")
    ybus_result = ybus_comparison.result
    if not ybus_result.bus_count_match:
        return ValidationResult(False, "Results are not comparable: the two cases have different numbers of buses.")
    if not ybus_result.topology_match:
        return ValidationResult(False, "Results are not comparable: the Ybus sparsity patterns differ, indicating a network topology mismatch.")
    if model_comparison is None or not model_comparison.success:
        return ValidationResult(False, "Results are not comparable: the model-comparison stage did not complete.")
    return ValidationResult(True, "Power-flow results are comparable: bus count and network topology match.")


def validate_model_comparison(model_comparison, ybus_comparison, results_comparable):
    """Validate that Ybus and metadata model-comparison stages completed."""
    if ybus_comparison is None or not ybus_comparison.success:
        return ValidationResult(False, "Model comparison is invalid because Ybus comparison did not complete.")
    if model_comparison is None or not model_comparison.success:
        return ValidationResult(False, f"Model comparison failed: {getattr(model_comparison, 'error_reason', 'not performed')}" )
    if not results_comparable.is_valid:
        return ValidationResult(True, "Model comparison completed, but solved-state results are not comparable. " + str(results_comparable.reason))
    return ValidationResult(True, "Model comparison completed and solved-state results are comparable.")


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
