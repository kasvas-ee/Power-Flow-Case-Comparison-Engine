from pathlib import Path
from src.comparison.result_structures import ComparisonReport
from src.reporting.model_summary_report import model_summary_report
from src.reporting.formatting import format_bool, section, REPORT_WIDTH
from src.reporting.state_summary_report import state_summary_report
from src.reporting.insights_report import insights_report, statistical_insights_report, investigation_priorities_report
from src.reporting.statistical_insights_html import generate_statistical_insights_html
from src.reporting.side_by_side_comparator import export_side_by_side_comparator
from src.reporting.ai_insight_report import generate_ai_insight_report
from src.logger import logger


def write_reports(config, engine, report):
    summary, full_text = write_comparison_report(report, str(config.output_path) + "/comparison_report.txt")

    # Optional HTML statistical report; never invalidate the base TXT report.
    try:
        html_path = generate_statistical_insights_html(
            report=report,
            output_path=config.output_path / "statistical_insights_report.html",
        )
        logger.info("Statistical insights HTML report exported to %s", html_path)
    except Exception as exc:
        logger.exception("Statistical insights HTML report generation failed: %s", exc)

    if config.generate_side_by_side_comparator and getattr(engine, "canonical_cases", None):
        export_side_by_side_comparator(engine.canonical_cases, report,
                                       config.output_path / "side_by_side_comparator.xlsx")

    if config.generate_ai_insight_report:
        try:
            html_path, json_path = generate_ai_insight_report(
                report=report,
                settings=config,
                output_dir=config.output_path,
            )
            logger.info("AI engineering insight report exported to %s", html_path)
            logger.info("AI insight evidence JSON exported to %s", json_path)
        except Exception as exc:
            # AI is an optional interpretation layer: never invalidate deterministic outputs.
            logger.exception("AI insight generation failed; deterministic reports remain available: %s", exc)

    return summary, full_text


def write_comparison_report(
    report: ComparisonReport,
    filename,
):
    """
    Build and write the complete power-flow comparison report.

    Parameters
    ----------
    report : ComparisonReport
        Final comparison report object produced by build_report().

    filename : str or Path
        Output TXT filename.

    Returns
    -------
    str
        Complete report text.
    """

    lines = []

    # ========================================================
    # Report header
    # ========================================================

    lines.extend(
        section(
            "POWER FLOW CASE COMPARISON REPORT"
        )
    )

    # ========================================================
    # Executive status
    # ========================================================

    lines.extend(
        [
            "Executive Status",
            "-" * REPORT_WIDTH,
            "",
        ]
    )

    lines.append(
        f"Model Match         : "
        f"{format_bool(report.model_match)}"
    )

    lines.append(
        f"Results Comparable  : "
        f"{format_bool(report.results_comparable.is_valid)}"
    )

    lines.append(
        f"Validation Passed   : "
        f"{format_bool(report.validation.is_valid)}"
    )

    if report.validation.reason:
        lines.append(
            f"Validation Reason   : "
            f"{report.validation.reason}"
        )

    lines.append(
        f"Results Match       : "
        f"{format_bool(report.results_match)}"
    )

    lines.append("")

    summary = lines.copy()

    # ========================================================
    # Model summary
    # ========================================================

    lines.append(
        model_summary_report(
            ybus_comparison=report.ybus_comparison,
            model_comparison=report.model_comparison,
            model_match=report.model_match,
            results_comparable=report.results_comparable.is_valid,
            metadata_results=getattr(
                getattr(report.model_comparison, "result", None),
                "metadata_comparisons",
                [],
            ),
        )
    )

    # ========================================================
    # State summary
    # ========================================================

    if report.power_flow_comparison is not None:

        lines.append(
            state_summary_report(
                power_flow_comparison=report.power_flow_comparison,
                results_comparable=report.results_comparable.is_valid,
                results_match=report.results_match,
                metadata_results=getattr(report.power_flow_comparison, "metadata_comparisons", []),
            )
        )

    # ========================================================
    # Insights
    # ========================================================

    if report.insights is not None:

        lines.append(insights_report(report.insights))

    # ========================================================
    # Statistical insights
    # ========================================================

    if report.statistical_insights is not None:
        lines.append(statistical_insights_report(report.statistical_insights))

    # ========================================================
    # Investigation Priorities
    # ========================================================

    if report.investigation_priorities is not None:
        lines.append(investigation_priorities_report(report.investigation_priorities))


    # ========================================================
    # Build final report
    # ========================================================

    report_text = "\n".join(lines)

    # ========================================================
    # Write UTF-8 TXT file
    # ========================================================

    path = Path(filename)

    path.write_text(
        report_text,
        encoding="utf-8",
    )

    return summary, report_text
