# #######################################################################################
# Create powerflow case files or predefined examples in 2 formats, that can be used as inputs to the main engine
# #######################################################################################
from datetime import datetime
from Examples.common import create_config, create_inputs, CASES
from src.Main import (
    setup_logging,
    logger,
    process_config,
    CaseDiffEngine,
    write_reports)



def list_examples():
    return sorted(CASES)


class Example:

    @staticmethod
    def create(example_name: str, m_file_path=None, **options):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        # Setup logger
        setup_logging()
        logger.info("Example started")

        config_file = f"config/{example_name}_settings.yaml"  # _{timestamp}
        input_dir = f"IOfiles/{example_name}/Inputs"
        output_dir = f"IOfiles/{example_name}/Outputs"

        create_config(config_file, input_dir, output_dir)
        create_inputs(input_dir, example_name, m_file_path, **options)

        # Read user-defined settings
        config = process_config(config_file)

        # Setup object for comparison engine
        engine = CaseDiffEngine(config)

        report = engine.compare()

        write_reports(config, engine, report)

        logger.info("Application completed")
        return report


if __name__ == "__main__":
    Example.create("IEEE14bus")  # , scale_impedance_by=1.1)  #, "C:/Users/KCH/PycharmProjects/Power-Flow-Case-Comparison-Engine/IOfiles/case6515rte.m")