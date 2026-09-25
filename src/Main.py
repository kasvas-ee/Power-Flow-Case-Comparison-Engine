# ###################################################
# THE MAIN FUNCTION TO RUN THE ENTIRE APPLICATION
# ###################################################

from src.logging_config import logger, setup_logging
from src.settings_config import process_config

if __name__ == "__main__":
    setup_logging()
    logger.info("Application started")

    settings_file = "./config/settings.yaml"
    # Read user-defined settings
    config = process_config(settings_file)

    # TODO: Read Cases A & B -> Canonical format

    # TODO: Pre-check cases A & B

    # TODO: Match Topology

    # TODO: Match powerflow results

    # TODO: Generate Statistics

    # TODO: Export / Visualize insights

    logger.info("Application completed")
