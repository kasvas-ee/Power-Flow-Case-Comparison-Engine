# ###################################################
# THE MAIN FUNCTION TO RUN THE ENTIRE APPLICATION
# ###################################################

from src.logging_config import logger, setup_logging
from src.settings_config import process_config
from src.parsers.parser import import_and_convert
from src.comparators.model_checker import match_model

if __name__ == "__main__":
    setup_logging()
    logger.info("Application started")

    settings_file = "./config/settings.yaml"
    # Read user-defined settings
    config = process_config(settings_file)

    # Read Cases A & B -> Canonical format
    case_a, case_b = import_and_convert(config)

    # Pre-check: Match model (topology and input parameters)
    is_match = match_model(config, case_a, case_b)
    if config.abort_if_network_mismatch and not is_match:
        raise ReferenceError("Network model is not identical. Aborting the comparison.")

    # TODO: Match state (powerflow)

    # TODO: Generate Statistics

    # TODO: Export / Visualize insights

    logger.info("Application completed")
