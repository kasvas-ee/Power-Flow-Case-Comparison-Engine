from types import SimpleNamespace
import json
import numpy as np

from src.reporting import ai_insight_report as ai


def _report():
    entry = {
        "scope": "model", "asset_name": "load", "parameter_name": "p_mw",
        "display_name": "Active Power", "parameter_id": "p_mw",
        "status": "SEVERE MISMATCH", "unit": "MW", "tolerance": 0.0001,
        "severe_threshold": 1.0, "reason": None,
        "statistics": {"mismatch_count": 1, "severe_mismatch_count": 1, "comparable_count": 3},
        "mismatch_mask": np.array([False, True, False]),
        "severe_mask": np.array([False, True, False]),
        "delta": np.array([0.0, 94.2, 0.0]),
        "case_a": np.array([50.0, 120.0, 40.0]),
        "case_b": np.array([50.0, 214.2, 40.0]),
        "asset_ids": np.array([0, 2, 3]),
    }
    return SimpleNamespace(
        ybus_comparison=SimpleNamespace(is_match=False, bus_count_match=True, topology_match=True,
            ybus_value_match=False, pattern_diff_count=0, value_diff_count=1,
            max_value_diff=0.01, reason=None),
        model_match=False, results_comparable=True, results_match=False,
        validation=SimpleNamespace(is_valid=True, reason=None),
        model_comparison=SimpleNamespace(metadata_comparisons=[entry]),
        power_flow_comparison=SimpleNamespace(metadata_comparisons=[], system=None),
    )


def test_evidence_package_is_compact_and_has_mismatch_examples():
    package = ai.build_evidence_package(_report())
    assert package["evidence"]
    assert any(item["facts"].get("mismatch_examples") for item in package["evidence"])
    assert len(package["evidence"]) <= ai.MAX_EVIDENCE_ITEMS


def test_invalid_evidence_references_are_removed():
    package = ai.build_evidence_package(_report())
    valid_id = package["evidence"][-1]["id"]
    payload = {
        "executive_assessment": "Potential model-related difference.",
        "diagnostic_narratives": [{
            "title": "Load hypothesis", "hypothesis": "Check the load source data.",
            "supporting_evidence_ids": [valid_id, "E999"],
            "counter_evidence_or_unknowns": "Causation is not established.",
            "confidence": "medium", "recommended_checks": ["Verify the input values."],
        }],
        "consistency_questions": [], "evidence_gaps": [],
        "overall_assessment": "likely_model_related",
    }
    cleaned = ai._validate_insights(payload, package["evidence"])
    assert cleaned["diagnostic_narratives"][0]["supporting_evidence_ids"] == [valid_id]


def test_html_is_self_contained_and_print_friendly():
    package = ai.build_evidence_package(_report())
    valid_id = package["evidence"][-1]["id"]
    payload = {
        "executive_assessment": "A mismatch warrants checking the source model.",
        "diagnostic_narratives": [{
            "title": "Check the load model", "hypothesis": "The load difference may be upstream.",
            "supporting_evidence_ids": [valid_id],
            "counter_evidence_or_unknowns": "Causation is unproven.",
            "confidence": "low", "recommended_checks": ["Verify scaling."],
        }],
        "consistency_questions": [], "evidence_gaps": [],
        "overall_assessment": "likely_model_related",
    }
    rendered = ai._render_html(payload, package, "local-test-model")
    assert "<!doctype html>" in rendered.lower()
    assert "@media print" in rendered
    assert "Check the load model" in rendered
    assert "cdn." not in rendered.lower()
