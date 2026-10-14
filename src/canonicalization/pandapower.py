# ################################################################
# Adapter to convert Panda Power format to canonical format
# ################################################################
import sys
import numpy as np
import pandas as pd
import pandapower as pp
from src.logger import logger


def validate_net(net):
    """Check that the canonical pandapower network has the required tables and columns."""
    # TODO: Include 'in_service' status
    required_columns = {
        "bus":   {},
        "line":  {"from_bus", "to_bus", "length_km"},
        "trafo": {"hv_bus", "lv_bus"},
        "load":  {"bus", "p_mw", "q_mvar"},
        "gen":   {"bus", "p_mw"},
        "ext_grid": {"bus"},
        "res_bus": {"vm_pu", "va_degree"},
        "res_line": {"p_from_mw", "q_from_mvar", "loading_percent"},
        "res_trafo": {"p_hv_mw", "q_hv_mvar", "loading_percent"},
        "res_load": {"p_mw", "q_mvar"},
        "res_gen": {"p_mw", "q_mvar"},
        "res_ext_grid": {"p_mw", "q_mvar"},
    }

    errors = []

    for table, columns in required_columns.items():
        # Check that the table exists
        if not hasattr(net, table):
            errors.append(f"Missing table: net.{table}")
            continue

        df = getattr(net, table)

        # Check that the table is a DataFrame
        if not hasattr(df, "columns"):
            errors.append(f"net.{table} is not a DataFrame")
            continue

        # Check required columns
        missing = set(columns) - set(df.columns)
        if missing:
            errors.append(
                f"net.{table} is missing columns: {sorted(missing)}"
            )

    if errors:
        for error in errors:
            logger.error(error)
        sys.exit(1)

    logger.info("Canonicalization passed.")

    return


def net_exporter(net, output_path):
    """Export selected pandapower element and result tables as CSV files."""
    export_list = ['bus', "ext_grid", "gen", "load", "impedance", "shunt", "poly_cost",
                   "res_bus", "res_ext_grid", "res_gen", "res_load", "res_impedance", "res_shunt"]

    for name in export_list:
        net[name].to_csv(output_path / f"{name}.csv", index=True)

    # # Uncomment to export all tables
    # import pandas as pd
    # for name, table in net.items():
    #     if isinstance(table, pd.DataFrame):
    #         table.to_csv(output_path / f"{name}.csv", index=True)

    return


def canonicalize_impedance(net):
    """
    Convert PandaPower line and transformer elements into a single canonical impedance / res_impedance representation.

    The resulting net contains:
        net.impedance
        net.res_impedance

    in place of:
        net.line, net.trafo
        net.res_line, net.res_trafo
    """

    # Convert to PPC format to get R, X B in PU for Lines and Trafo
    ppc = pp.converter.to_ppc(net)

    # Join branch + trafo results
    branch_cols = ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
    branch_res = pd.concat(
        [net.res_line[branch_cols],
         net.res_trafo[["p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar"]].set_axis(branch_cols, axis=1)],
        axis=0
    )

    # Swap Fbus & Tbus of branches and their results where FbusID>TbusID
    mask = ppc["branch"][:, 0] > ppc["branch"][:, 1]

    if mask.any():
        # Swap from/to bus IDs
        ppc["branch"][np.ix_(mask, [0, 1])] = ppc["branch"][np.ix_(mask, [1, 0])]

        # Swap Pf <-> Pt and Qf <-> Qt
        branch_res.iloc[mask, :] = branch_res.to_numpy()[np.ix_(mask, [2, 3, 0, 1])]

    # Sort data by the From-To bus to use this index to reorder both data and results
    # A stable terminal ordering keeps matching branch rows aligned across cases.
    sorted_idx = np.lexsort((ppc["branch"][:, 1], ppc["branch"][:, 0]))  # Sorted order

    # TODO: Expand to other asset-types beyond Bus, Branch, Load and Gen

    # Create impedance from sorted branch
    branch_array = ppc["branch"][sorted_idx]
    net.impedance = pd.DataFrame({
        "from_bus": (branch_array[:, 0]).astype(int),
        "to_bus": (branch_array[:, 1]).astype(int),
        "rft_pu": branch_array[:, 2],
        "xft_pu": branch_array[:, 3],
        "rtf_pu": branch_array[:, 2],
        "xtf_pu": branch_array[:, 3],
        "sn_mva": branch_array[:, 5],
        "in_service": True,
        "gf_pu": 0.0,
        "gt_pu": 0.0,
        "bf_pu": branch_array[:, 4] / 2,
        "bt_pu": branch_array[:, 4] / 2,
    })

    # Create impedance res from sorted branch results
    net.res_impedance = branch_res.iloc[sorted_idx].reset_index(drop=True)

    # Delete branch and trafo DFs
    net["line"] = net["line"].iloc[0:0].copy()
    net["trafo"] = net["trafo"].iloc[0:0].copy()
    net["res_line"] = net["res_line"].iloc[0:0].copy()
    net["res_trafo"] = net["res_trafo"].iloc[0:0].copy()

    return net


def update_res_gen(net):
    """
    Merge res_ext_gen into res_gen and ensure the results are sorted by bus ID
    """
    net["res_gen"] = pd.concat([
        net.res_ext_grid.assign(bus=net.ext_grid["bus"]),
        net.res_gen.assign(bus=net.gen["bus"]),
    ]).sort_values("bus").drop(columns="bus").reset_index(drop=True)

    # Delete branch and trafo DFs
    net["res_ext_grid"] = net["res_ext_grid"].iloc[0:0].copy()
    return net


def pp_to_canonical(case_path, output_path, export_net):
    """
    Read the case file in Panda Power's native (JSON) format, convert it to the canonical CSV formats
    and export it to the output path
    """
    try:
        net = pp.from_json(case_path)

        # Convert Branch and transformers to impedance for uniformity with other formats
        net = canonicalize_impedance(net)

        # Identify slack bus and join ext_gen to the gen list in result tables
        net = update_res_gen(net)

        # TODO: Placeholder to handle topological reductions here (eg: bus reduction).

        if export_net:
            net_exporter(net, output_path)

        validate_net(net)

        return net
    except Exception as e:
        msg = f"Failed to load data from {case_path}: {e}"
        logger.error(msg)
        raise IOError(msg)
