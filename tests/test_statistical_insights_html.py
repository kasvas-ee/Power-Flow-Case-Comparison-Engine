from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np

from src.reporting.statistical_insights_html import generate_statistical_insights_html


def test_html_report_contains_metrics_and_legend():
    item = {
        "scope": "model", "asset_name": "branch", "display_name": "Branch Resistance",
        "parameter_name": "branch_r", "unit": "pu", "tolerance": 1e-7,
        "severe_threshold": 1e-5,
        "asset_ids": np.array([101, 102, 103]),
        "delta": np.array([0.0, 1e-6, 2e-5]),
        "mismatch_mask": np.array([False, True, True]),
        "statistics": {
            "comparable_count": 3, "mismatch_count": 2, "mismatch_percent": 66.6667,
            "severe_mismatch_count": 1, "mean_delta": 7e-6, "median_delta": 1e-6,
            "mad_delta": 1e-6, "p95_delta": 1.8e-5, "p99_delta": 1.96e-5,
            "max_delta": 2e-5, "positive_delta_percent": 100.0,
            "negative_delta_percent": 0.0, "asset_count": 3,
            "affected_asset_count": 2, "affected_asset_percent": 66.6667,
        },
    }
    comparison = SimpleNamespace(metadata_comparisons=[item])
    report = SimpleNamespace(
        model_comparison=comparison, power_flow_comparison=None,
        statistical_insights={
            "model": [{"category": "percentile_outlier_pattern", "message": "Maximum delta is much larger than P95.", "parameter_name": "branch_r"}],
            "power_flow": [], "overall": [],
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "report.html"
        result = generate_statistical_insights_html(report, target)
        text = result.read_text(encoding="utf-8")
        assert "Branch Resistance" in text
        assert "Largest differences" in text
        assert "Extreme outlier pattern" in text
        assert "Legend: insight terms and how to investigate" in text
        assert "Expand all" in text
        assert "Branch 102" in text or "branch 102" in text
