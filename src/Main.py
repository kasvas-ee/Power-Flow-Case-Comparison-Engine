# ###################################################
# THE MAIN FUNCTION TO RUN THE ENTIRE APPLICATION
# ###################################################

from src.logging_config import logger, setup_logging

if __name__ == "__main__":
    setup_logging()
    logger.info("Application started")

    settings_file = "./config/settings.yaml"
    # TODO: Read user-defined settings

    # TODO: Read Cases A & B -> Canonical format

    # TODO: Pre-check cases A & B

    # TODO: Match Topology

    # TODO: Match powerflow results

    # TODO: Generate Statistics

    # TODO: Export / Visualize insights

    logger.info("Application completed")
