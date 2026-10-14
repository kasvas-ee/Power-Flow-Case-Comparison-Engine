"""Define structured dataclass containers for validation, comparison, summary, and reporting stages."""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActionResult:
    """
    Result of an executable comparison-engine action.

    Used for operations that can succeed or fail independently,
    such as PPC conversion or Ybus construction.
    """

    action: str
    success: Optional[bool] = False
    result: Optional[Any] = None
    error_type: Optional[str] = None
    error_reason: Optional[str] = None


@dataclass
class ComparisonResult:
    """
    Generic numerical comparison result.

    Used for both model-data comparisons and solved-state
    power-flow metric comparisons.
    """

    metric: Optional[str] = ""
    unit: Optional[str] = ""

    is_match: bool = False

    compared_count: int = 0
    mismatch_count: int = 0
    severe_count: int = 0

    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    rms_diff: Optional[float] = None

    p50_abs_diff: Optional[float] = None
    p90_abs_diff: Optional[float] = None
    p95_abs_diff: Optional[float] = None
    p99_abs_diff: Optional[float] = None

    max_abs_diff_index: Optional[int] = None
    max_abs_diff_asset: Optional[str] = None

    severe_rows: Optional[List[str]] = None

    tolerance: float = 0.0
    severity_threshold: Optional[float] = None

    reason: Optional[str] = None


@dataclass
class TopSevereResult:
    """
    One ranked severe mismatch.

    Signed delta is retained because it provides useful
    directionality for reporting and insight generation.
    """

    rank: int
    asset: str

    value_a: Optional[float] = None
    value_b: Optional[float] = None

    delta: Optional[float] = None
    abs_delta: Optional[float] = None


@dataclass
class ModelComparisonResult:
    """Metadata-driven model diagnostics after the primary Ybus comparison.

    The component summary fields are lightweight views used by insight text;
    all numerical comparisons are owned by metadata_comparator.py.
    """

    bus: Optional[ComparisonResult] = None
    branch: Optional[ComparisonResult] = None
    load: Optional[ComparisonResult] = None
    generation: Optional[ComparisonResult] = None

    metadata_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    top_severe: Dict[str, List[TopSevereResult]] = field(default_factory=dict)


@dataclass
class YbusComparisonResult:
    """
    Result of comparing the two Ybus matrices.

    Ybus comparison is deliberately decomposed into:
        1. bus count / matrix shape
        2. topology / sparsity pattern
        3. numerical admittance values
    """

    is_match: bool = False

    bus_count_match: bool = False
    topology_match: bool = False
    ybus_value_match: bool = False

    pattern_diff_count: Optional[int] = None
    value_diff_count: Optional[int] = None
    max_value_diff: Optional[float] = None

    shape_a: Optional[Tuple[int, int]] = None
    shape_b: Optional[Tuple[int, int]] = None

    reason: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Final validation state of the comparison prerequisites.
    """

    is_valid: bool
    reason: Optional[str] = None


@dataclass
class SystemSummary:
    """
    System-level P/Q summary for the two solved cases.

    Values are expressed separately for Case A and Case B,
    together with their signed differences.
    """

    pgen_a_mw: Optional[float] = None
    pgen_b_mw: Optional[float] = None
    delta_pgen_mw: Optional[float] = None

    qgen_a_mvar: Optional[float] = None
    qgen_b_mvar: Optional[float] = None
    delta_qgen_mvar: Optional[float] = None

    pload_a_mw: Optional[float] = None
    pload_b_mw: Optional[float] = None
    delta_pload_mw: Optional[float] = None

    qload_a_mvar: Optional[float] = None
    qload_b_mvar: Optional[float] = None
    delta_qload_mvar: Optional[float] = None

    ploss_a_mw: Optional[float] = None
    ploss_b_mw: Optional[float] = None
    delta_ploss_mw: Optional[float] = None

    qloss_a_mvar: Optional[float] = None
    qloss_b_mvar: Optional[float] = None
    delta_qloss_mvar: Optional[float] = None

    p_balance_a_mw: Optional[float] = None
    p_balance_b_mw: Optional[float] = None
    delta_p_balance_mw: Optional[float] = None

    q_balance_a_mvar: Optional[float] = None
    q_balance_b_mvar: Optional[float] = None
    delta_q_balance_mvar: Optional[float] = None

    vm_min_a: Optional[float] = None
    vm_min_bus_a: Optional[Any] = None
    vm_max_a: Optional[float] = None
    vm_max_bus_a: Optional[Any] = None

    vm_min_b: Optional[float] = None
    vm_min_bus_b: Optional[Any] = None
    vm_max_b: Optional[float] = None
    vm_max_bus_b: Optional[Any] = None

    va_min_a: Optional[float] = None
    va_min_bus_a: Optional[Any] = None
    va_max_a: Optional[float] = None
    va_max_bus_a: Optional[Any] = None

    va_min_b: Optional[float] = None
    va_min_bus_b: Optional[Any] = None
    va_max_b: Optional[float] = None
    va_max_bus_b: Optional[Any] = None


@dataclass
class PowerFlowComparisonResult:
    """
    Solved-state comparison results.

    Each dictionary is keyed by metric name, for example:
        bus["Bus voltage magnitude"]
        generation["Generation P"]
        load["Load P"]
        branch["Branch P loss"]

    System is a dedicated summary because system-level values
    are not row-wise metric comparisons.
    """

    bus: Dict[str, ComparisonResult] = field(
        default_factory=dict
    )

    generation: Dict[str, ComparisonResult] = field(
        default_factory=dict
    )

    load: Dict[str, ComparisonResult] = field(
        default_factory=dict
    )

    branch: Dict[str, ComparisonResult] = field(
        default_factory=dict
    )

    # Per-asset-type summaries, separate from individual metric summaries.
    asset_summaries: Dict[str, Any] = field(default_factory=dict)

    system: Optional[SystemSummary] = None

    # Enabled solved-state parameter results from the metadata comparator.
    metadata_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    top_severe: Dict[str, List[TopSevereResult]] = field(
        default_factory=dict
    )


@dataclass
class ComparisonReport:
    """
    Complete comparison-engine output.

    Status semantics:

        model_match
            Did the canonical network models match?

        results_comparable
            Is it valid to compare the solved-state results?

        results_match
            Did the compared solved-state results match?

    results_match is None when results were not comparable
    or were not evaluated.
    """

    ppcs: Any
    ybus: Any

    ybus_comparison: YbusComparisonResult
    model_comparison: ModelComparisonResult

    model_match: bool
    results_comparable: bool
    results_match: Optional[bool]

    validation: ValidationResult

    power_flow_comparison: Optional[
        PowerFlowComparisonResult
    ] = None

    insights: Optional[Any] = None
    statistical_insights: Optional[Any] = None
    investigation_priorities: Optional[Any] = None


def build_report(
    ppcs,
    ybus,
    ybus_comparison,
    model_comparison,
    model_match,
    results_comparable,
    results_match,
    validation,
    power_flow_comparison=None,
    insights=None,
    statistical_insights=None,
    investigation_priorities=None
):
    """
    Construct the final comparison report.
    """

    return ComparisonReport(
        ppcs=ppcs,
        ybus=ybus,
        ybus_comparison=ybus_comparison,
        model_comparison=model_comparison,
        model_match=model_match,
        results_comparable=results_comparable,
        results_match=results_match,
        validation=validation,
        power_flow_comparison=power_flow_comparison,
        insights=insights,
        statistical_insights=statistical_insights,
        investigation_priorities=investigation_priorities
    )


@dataclass(frozen=True)
class MetricDefinition:
    scope: str
    source: str
    asset_name: str
    asset_id: str
    parameter_name: str
    parameter_id: str
    enabled: bool
    comparison: str
    tolerance: float
    severe_threshold: float | None
    unit: str | None
    display_name: str
    match_strategy: str | None

    @property
    def is_derived(self) -> bool:
        """Indicate whether the metric is defined by a formula instead of a direct field."""
        return self.source == "derived"
