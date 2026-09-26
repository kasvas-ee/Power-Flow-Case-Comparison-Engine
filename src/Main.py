# ###################################################
# THE MAIN FUNCTION TO RUN THE ENTIRE APPLICATION
# ###################################################

from src.logging_config import logger, setup_logging
from src.settings_config import process_config
from src.importers.importer import import_and_convert

if __name__ == "__main__":
    setup_logging()
    logger.info("Application started")

    settings_file = "./config/settings.yaml"
    # Read user-defined settings
    config = process_config(settings_file)

    # Read Cases A & B -> Canonical format
    case_a, case_b = import_and_convert(config)

    # TODO: Pre-check cases A & B

    # TODO: Match Topology

    # TODO: Match powerflow results

    # TODO: Generate Statistics

    # TODO: Export / Visualize insights

    logger.info("Application completed")
