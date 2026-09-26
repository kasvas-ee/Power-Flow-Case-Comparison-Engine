# ################################################################
# Adapter to convert Panda Power format to canonical format
# ################################################################
import pandapower as pp
from src.logging_config import logger


def net_exporter(net, output_path):
    export_list = ['bus', "ext_grid", "gen", "load", "line", "trafo", "impedance", "res_bus", "res_ext_grid", "res_gen", "res_load", "res_line", "res_trafo"]

    for name in export_list:
        net[name].to_csv(output_path / f"{name}.csv", index=True)

    # # Uncomment to export all tables
    # for name, table in net.items():
    #     if isinstance(table, pd.DataFrame):
    #         table.to_csv(output_path / f"{name}.csv", index=True)

    return


def pp_to_canonical(case_path, output_path, export_net):
    """
    Read the case file in Panda Power's native (JSON) format, convert it to the canonical CSV formats
    and export it to the output path
    """
    try:
        net = pp.from_json(case_path)

        if export_net:
            net_exporter(net, output_path)

        return net
    except Exception as e:
        msg = f"Failed to load data from {case_path}: {e}"
        logger.error(msg)
        raise IOError(msg)
