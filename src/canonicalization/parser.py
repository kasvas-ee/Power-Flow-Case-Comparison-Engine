"""Select and run the canonicalization adapter for each configured input case."""

from src.canonicalization.matpower import mp_to_canonical
from src.canonicalization.pandapower import pp_to_canonical
from src.canonicalization.powsybl import ppsb_to_canonical
from src.canonicalization.psse import psse_to_canonical
from src.canonicalization.pypsa import pypsa_to_canonical
from src.logger import logger


def canonicalize_cases(config):
    """
    Calls the appropriate adapter based on the data format
    to convert to the canonical format and export to the output path
    return case A and case B in canonical format
    """
    parsers = {
        "matpower": mp_to_canonical,
        "pandapower": pp_to_canonical,
        "pypowsybl": ppsb_to_canonical,
        "psse": psse_to_canonical,
        "pypsa": pypsa_to_canonical,
    }

    # Create the output folder
    (config.output_path / "converted_data").mkdir(exist_ok=True)

    cases = {"case_a":[config.case_a_format, config.case_a_path], "case_b":[config.case_b_format, config.case_b_path]}
    converted_cases = ()
    for case_title, case_details in cases.items():
        parser_class = parsers.get(case_details[0])
        if parser_class is None:
            raise ValueError(f"Unsupported format {case_details[0]}")

        # Create the output folder
        output_path = config.output_path / "converted_data" / case_title
        output_path.mkdir(exist_ok=True)

        # Convert the data
        converted_cases += (parser_class(case_details[1], output_path, config.export_net),)

        # TODO: Handle topological differences. A fixed 1-to-1 asset map is assumed for now.

        logger.info("Read & converted data for {}".format(case_title))

    return converted_cases
