import numpy as np

from src.insights.statistical_insights import (
    generate_statistical_insights,
)


def _make_comparison(
    scope="model",
    asset_name="line",
    parameter_name="resistance",
    mismatch_count=0,
    comparable_count=4,
    mismatch_percent=0.0,
):
    mismatch_mask = np.array(
        [True] * mismatch_count
        + [False] * (comparable_count - mismatch_count),
        dtype=bool,
    )

    delta = np.array(
        [1.0] * mismatch_count
        + [0.0] * (comparable_count - mismatch_count),
        dtype=float,
    )

    return {
        "scope": scope,
        "asset_name": asset_name,
        "parameter_name": parameter_name,
        "display_name": parameter_name,
        "asset_ids": np.arange(comparable_count),
        "delta": delta,
        "mismatch_mask": mismatch_mask,
        "status": "MISMATCH" if mismatch_count else "MATCH",
        "statistics": {
            "comparable_count": comparable_count,
            "mismatch_count": mismatch_count,
            "severe_mismatch_count": 0,
            "mismatch_percent": mismatch_percent,
            "median_delta": float(np.median(delta)),
            "p95_delta": float(np.percentile(delta, 95)),
            "p99_delta": float(np.percentile(delta, 99)),
            "max_delta": float(np.max(delta)),
            "median_relative_delta": None,
            "p95_relative_delta": None,
            "max_relative_delta": None,
        },
    }


def test_empty_comparisons_return_empty_findings():
    result = generate_statistical_insights(
        model_comparison=None,
        power_flow_comparison=None,
    )

    assert result == {
        "model": [],
        "power_flow": [],
        "overall": [],
    }


def test_model_mismatch_produces_input_discrepancy():
    item = _make_comparison(
        mismatch_count=1,
        comparable_count=4,
        mismatch_percent=25.0,
    )

    class ModelResult:
        metadata_comparisons = [item]

    class ModelComparison:
        result = ModelResult()

    result = generate_statistical_insights(
        model_comparison=ModelComparison(),
    )

    input_findings = [
        finding
        for finding in result["model"]
        if finding["category"] == "input_discrepancy"
    ]

    assert len(input_findings) == 1
    assert "resistance" in input_findings[0]["message"]


def test_matching_values_produce_no_input_discrepancy():
    item = _make_comparison(
        mismatch_count=0,
        comparable_count=4,
        mismatch_percent=0.0,
    )

    class ModelResult:
        metadata_comparisons = [item]

    class ModelComparison:
        result = ModelResult()

    result = generate_statistical_insights(
        model_comparison=ModelComparison(),
    )

    input_findings = [
        finding
        for finding in result["model"]
        if finding["category"] == "input_discrepancy"
    ]

    assert input_findings == []


def test_uncomparable_power_flow_is_not_interpreted():
    model_item = _make_comparison(
        mismatch_count=1,
        comparable_count=4,
        mismatch_percent=25.0,
    )

    power_flow_item = _make_comparison(
        scope="state",
        mismatch_count=4,
        comparable_count=4,
        mismatch_percent=100.0,
    )

    class ModelResult:
        metadata_comparisons = [model_item]

    class ModelComparison:
        result = ModelResult()

    class PowerFlowResult:
        metadata_comparisons = [power_flow_item]

    result = generate_statistical_insights(
        model_comparison=ModelComparison(),
        power_flow_comparison=PowerFlowResult(),
        results_comparable=False,
    )

    assert result["power_flow"] == []