# ################################################################
# Adapter to convert Panda Power format to canonical format
# ################################################################

import pandapower as pp
from src.logging_config import logger

def pp_to_canonical(case_path, output_path):
    """
    Read the case file in Panda Power's native (JSON) format, convert it to the canonical CSV formats
    and export it to the output path
    """
    try:
        net = pp.from_json(case_path)

        for table_name, table in net.items():
            if hasattr(table, "to_csv"):
                table.to_csv(output_path / f"{table_name}.csv", index=True)
    except Exception as e:
        msg = f"Failed to load data from {case_path}: {e}"
        logger.error(msg)
        raise IOError(msg)
