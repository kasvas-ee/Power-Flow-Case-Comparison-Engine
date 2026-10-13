# ###################################################
# THE MAIN FUNCTION TO RUN THE ENTIRE APPLICATION
# ###################################################

from src.logger import logger, setup_logging
from src.config import process_config
from src.canonicalization.parser import canonicalize_cases
from src.comparison.model_checker import match_model, determine_model_match
from src.comparison.state_checker import run_power_flow_comparison
from src.comparison.result_structures import ValidationResult, build_report
from src.comparison.result_checker import results_match_from_metadata, populate_state_asset_summaries
from src.insights.comparison_summary_insights import generate_insights, generate_investigation_priorities
from src.insights.statistical_insights import generate_statistical_insights
from src.reporting.report import write_reports


class CaseDiffEngine:

    def __init__(self, config):
        self.config = config

    def compare(self):
        """
        Engine Orchistrator
        """

        # -----------------------------------------------------
        # 1. Read Cases A & B -> Canonical format
        # -----------------------------------------------------

        canonical_cases = canonicalize_cases(self.config)
        self.canonical_cases = canonical_cases

        # -----------------------------------------------------
        # 2. Compare network/model
        # -----------------------------------------------------

        ppcs, ybus, ybus_comparison, model_comparison, results_comparability, asset_counts_a, asset_counts_b = (
            match_model(*canonical_cases, config=self.config)
        )

        # -----------------------------------------------------
        # 3. Exit if results cannot be compared
        # -----------------------------------------------------

        if results_comparability.is_valid:
            validation = ValidationResult(True, "Model comparison completed and solved-state results are comparable.")
        else:
            validation = ValidationResult(False, "Solved-state results are not comparable. "
                                                 "Reason: " + str(results_comparability.reason))

            return build_report(
                ppcs=ppcs,
                ybus=ybus,
                ybus_comparison=ybus_comparison,
                model_comparison=model_comparison,
                model_match=False,
                results_comparable=results_comparability,
                results_match=None,
                validation=validation,
            )

        # -----------------------------------------------------
        # 4. Compare solved power-flow states
        # -----------------------------------------------------

        power_flow_action = run_power_flow_comparison(
            *canonical_cases,
            self.config,
        )

        # Generate asset-level summary from parameter-level summary
        populate_state_asset_summaries(
            power_flow_action.result,
            asset_counts_a,
            asset_counts_b,
        )

        if not power_flow_action.success:
            validation = ValidationResult(
                is_valid=False,
                reason=(
                    "Power-flow comparison failed: "
                    f"{power_flow_action.error_reason}"
                ),
            )

            return build_report(
                ppcs=ppcs,
                ybus=ybus,
                ybus_comparison=ybus_comparison,
                model_comparison=model_comparison,
                model_match=False,
                results_comparable=results_comparability,
                results_match=None,
                validation=validation,
            )

        power_flow_comparison = power_flow_action.result

        # -----------------------------------------------------
        # 5. Determine overall result match
        # -----------------------------------------------------

        results_match = results_match_from_metadata(
            power_flow_comparison.metadata_comparisons
        )

        # Generate Insights
        insights = generate_insights(
            ybus_comparison=ybus_comparison,
            model_comparison=model_comparison,
            power_flow_comparison=power_flow_comparison,
            model_match=determine_model_match(ybus_comparison, model_comparison),
            results_comparable=results_comparability,
            results_match=results_match,
        )

        statistical_insights = generate_statistical_insights(
            model_comparison=model_comparison,
            power_flow_comparison=power_flow_comparison,
            results_comparable=results_comparability.is_valid,
        )

        investigation_priorities = generate_investigation_priorities(
            power_flow_comparison=power_flow_comparison,
            model_comparison=model_comparison,
            model_match=determine_model_match(ybus_comparison, model_comparison),
            statistical_insights=statistical_insights,
        )

        # -----------------------------------------------------
        # 6. TODO: Export / Visualize insights
        # -----------------------------------------------------

        # visual_insights = generate_insights()

        # -----------------------------------------------------
        # 7. Build final report
        # -----------------------------------------------------

        report = build_report(
            ppcs=ppcs,
            ybus=ybus,
            ybus_comparison=ybus_comparison,
            model_comparison=model_comparison,
            model_match=determine_model_match(ybus_comparison, model_comparison),
            results_comparable=results_comparability,
            results_match=results_match,
            validation=validation,
            power_flow_comparison=power_flow_comparison,
            insights=insights,
            statistical_insights=statistical_insights,
            investigation_priorities=investigation_priorities
        )

        return report


def main(config_file):
    """
    Application entry point
    """
    # Setup logger
    setup_logging()
    logger.info("Application started")

    # Read user-defined settings
    config = process_config(config_file)

    # Setup object for comparison engine
    engine = CaseDiffEngine(config)

    report = engine.compare()

    write_reports(config, engine, report)

    logger.info("Application completed")
    return report


if __name__ == "__main__":
    settings_file = "./config/settings.yaml"
    result = main(settings_file)
