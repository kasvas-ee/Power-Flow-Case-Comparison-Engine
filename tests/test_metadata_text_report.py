from pathlib import Path

from src.comparison.result_structures import (
    ActionResult, ComparisonReport, ModelComparisonResult,
    PowerFlowComparisonResult, ValidationResult, YbusComparisonResult,
)
from src.reporting.report import write_comparison_report


def _metadata(scope, asset, parameter, display, values_a, values_b, delta,
              status="MISMATCH", tol=0.01, severe=0.1):
    import numpy as np
    d = np.asarray(delta, dtype=float)
    mismatch = np.abs(d) > tol
    severe_mask = np.abs(d) > severe
    return {
        "scope": scope, "asset_name": asset, "asset_id": asset,
        "parameter_name": parameter, "parameter_id": parameter,
        "display_name": display, "unit": "pu", "tolerance": tol,
        "severe_threshold": severe, "status": status, "reason": None,
        "case_a": np.asarray(values_a), "case_b": np.asarray(values_b),
        "delta": d, "asset_ids": np.array([1, 2]),
        "mismatch_mask": mismatch, "severe_mask": severe_mask,
        "statistics": {"comparable_count": 2, "mismatch_count": int(mismatch.sum()),
                       "severe_mismatch_count": int(severe_mask.sum())},
    }


def test_report_uses_ybus_and_metadata_in_original_sections(tmp_path):
    ybus = ActionResult("Ybus Comparison", True, YbusComparisonResult(
        is_match=True, bus_count_match=True, topology_match=True,
        ybus_value_match=True, pattern_diff_count=0, value_diff_count=0,
        max_value_diff=0.0, shape_a=(2, 2), shape_b=(2, 2),
        reason="Ybus matrices match within the specified tolerance."))
    model = ActionResult("Model Comparison", True, ModelComparisonResult(
        metadata_comparisons=[_metadata("model", "branch", "r_pu", "Branch Resistance",
                                        [0.1, 0.2], [0.1, 0.3], [0.0, 0.1])]))
    state = PowerFlowComparisonResult(metadata_comparisons=[
        _metadata("state", "bus", "vm_pu", "Bus Voltage Magnitude",
                  [1.0, 1.0], [1.0, 0.95], [0.0, 0.05])])
    report = ComparisonReport(
        ppcs=None, ybus=None, ybus_comparison=ybus, model_comparison=model,
        model_match=False, results_comparable=ValidationResult(True, "comparable"),
        results_match=False, validation=ValidationResult(True, "validation passed"),
        power_flow_comparison=state, insights=None)
    path = tmp_path / "comparison_report.txt"
    text = write_comparison_report(report, path)
    assert path.read_text(encoding="utf-8") == text
    assert "1. MODEL / NETWORK SUMMARY" in text
    assert "Ybus Match          : YES" in text
    assert "Detailed Model Comparisons" in text
    assert "Branch Resistance" in text
    assert "2. POWER-FLOW STATE SUMMARY" in text
    assert "Bus Voltage Magnitude" in text
    assert "METADATA-DRIVEN" not in text
