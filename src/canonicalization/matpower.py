# #########################################################################
# TODO: Adapter to convert MATPOWER case file (.m file) to canonical format
# #########################################################################
from scipy.io import loadmat
import pandapower as pp
import pandas as pd
import numpy as np
from src.canonicalization.pandapower import net_exporter
from src.logger import logger


def mat_to_pp(mat, net):
    """
    The data  and results (if available) are copied from MATPOWER format (mat) to PandaPower format (net)
    """
    # ## DATA ##
    # Base MVA
    net.sn_mva = mat["baseMVA"].max()

    # bus
    bus_array = mat["bus"]
    vn_kv = 1 if max(bus_array[:, 9]) == 0 else bus_array[:, 9] # Nominal voltages (column 9 in MATPOWER bus table)
    pp.create_buses(net, len(bus_array), vn_kv=vn_kv,)  # Load all buses into pp

    # shunts
    bus_indices = np.arange(len(bus_array))

    gs_mw = bus_array[:, 4]  # GS
    bs_mvar = bus_array[:, 5] * -1  # BS

    shunt_mask = (gs_mw != 0) | (bs_mvar != 0)

    pp.create_shunts(
        net,
        buses=bus_indices[shunt_mask],
        p_mw=gs_mw[shunt_mask],
        q_mvar=bs_mvar[shunt_mask],
    )

    # Generators
    gen_array = mat["gen"]

    # --Identify slack bus indices (MATPOWER bus type = 3)
    slack_buses = bus_array[bus_array[:, 1] == 3][:, 0].astype(int)  # bus numbers (1-based)
    slack_mask = np.isin(gen_array[:, 0].astype(int), slack_buses)

    # --Non-slack generators
    gen_indices = pp.create_gens(
        net,
        buses=(gen_array[~slack_mask, 0] - 1).astype(int),  # bus numbers (MATPOWER is 1-based)
        min_q_mvar=gen_array[~slack_mask, 4],
        max_q_mvar=gen_array[~slack_mask, 3],
        max_p_mw=gen_array[~slack_mask, 8],
        min_p_mw=gen_array[~slack_mask, 9],
        p_mw=gen_array[~slack_mask, 1],
        vm_pu=gen_array[~slack_mask, 5]
    )

    # --Slack generator(s) → ext_grid
    # Note: Assuming only 1 slack bus
    ext_grid_index = pp.create_ext_grid(
        net,
        bus=(gen_array[slack_mask, 0] - 1).astype(int)[0],  # usually one slack gen
        vm_pu=gen_array[slack_mask, 5][0],
        va_degree=bus_array[bus_array[:, 1] == 3][:, 8][0],  # bus voltage angle (MATPOWER bus col 8)
        min_q_mvar=gen_array[slack_mask, 4][0],
        max_q_mvar=gen_array[slack_mask, 3][0],
        max_p_mw=gen_array[slack_mask, 8][0],
        min_p_mw=gen_array[slack_mask, 9][0],
        # slack=True
    )

    # --Gen Poly Cost
    gencost = mat["gencost"]
    non_slack_cost = gencost[~slack_mask]
    slack_cost = gencost[slack_mask]

    ext_grid_cost_df = pd.DataFrame({
        "element": [ext_grid_index],
        "et": "ext_grid",
        "cp0_eur": slack_cost[:, -1] ,
        "cp1_eur_per_mw": slack_cost[:, -2] if slack_cost.shape[1]>5 else 0,
        "cp2_eur_per_mw2": slack_cost[:, -3] if slack_cost.shape[1]>6 else 0,
    })

    gen_cost_df = pd.DataFrame({
        "element": gen_indices,
        "et": "gen",
        "cp0_eur": non_slack_cost[:, -1] ,
        "cp1_eur_per_mw": non_slack_cost[:, -2] if non_slack_cost.shape[1]>5 else 0,
        "cp2_eur_per_mw2": non_slack_cost[:, -3] if non_slack_cost.shape[1]>6 else 0,
    })

    net.poly_cost = pd.concat(
        [net.poly_cost, ext_grid_cost_df, gen_cost_df],
        ignore_index=True
    )

    # load
    pq_buses = np.where((bus_array[:, 2] != 0) | (bus_array[:, 3] != 0))[0]
    pp.create_loads(
        net,
        buses=pq_buses,
        p_mw=bus_array[pq_buses, 2],
        q_mvar=bus_array[pq_buses, 3]
    )

    # branch - Creating impedance instead of lines & trafo to avoid converting r,x,b from p.u. to actual R, X, C & G
    branch_array = mat["branch"]

    # Sort data (and result) by the From-To bus to use this index to reorder both data and results
    sorted_idx = np.lexsort((branch_array[:, 1], branch_array[:, 0]))  # Sorted order
    branch_array = branch_array[sorted_idx]

    net.impedance = pd.DataFrame({
        "from_bus": (branch_array[:, 0] - 1).astype(int),
        "to_bus": (branch_array[:, 1] - 1).astype(int),
        "rft_pu": branch_array[:, 2],
        "xft_pu": branch_array[:, 3],
        "rtf_pu": branch_array[:, 2],
        "xtf_pu": branch_array[:, 3],
        "sn_mva": branch_array[:, 5],
        "in_service": True,
        "gf_pu": 0.0,
        "gt_pu": 0.0,
        "bf_pu": branch_array[:, 4]/2,
        "bt_pu": branch_array[:, 4]/2,
    })

    # ## RESULTS ##
    # --- Bus results ---
    bus = pd.DataFrame(bus_array[:, :13], columns=[
        "BUS_I", "BUS_TYPE", "Pd", "Qd", "Gs", "Bs", "BUS_AREA", "Vm", "Va", "BASE_KV", "ZONE", "Vmax", "Vmin"
    ])
    net.res_bus = pd.DataFrame({
        "vm_pu": bus["Vm"],
        "va_degree": bus["Va"],
        "p_mw": bus["Pd"],
        "q_mvar": bus["Qd"]
    })

    # --- Shunt results ---
    vm_pu = net.res_bus.loc[net.shunt["bus"], "vm_pu"].to_numpy()
    res = net.shunt[["p_mw", "q_mvar"]].to_numpy() * vm_pu[:, None] ** 2

    net.res_shunt = pd.DataFrame(
        res,
        columns=["p_mw", "q_mvar"],
        index=net.shunt.index,
    )

    # --- Generator results ---
    gen = pd.DataFrame(gen_array[:, :10], columns=[
        "GEN_BUS", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin"
    ])

    # --- --Non-slack generator results
    net.res_gen = pd.DataFrame({
        "p_mw": gen.loc[:, "Pg"].values,
        "q_mvar": gen.loc[:, "Qg"].values,
        "vm_pu": gen.loc[:, "Vg"].values
    })

    # --- Load results (from bus Pd/Qd) ---
    load_mask = (bus["Pd"] != 0) | (bus["Qd"] != 0)
    net.res_load = pd.DataFrame({
        "p_mw": bus.loc[load_mask, "Pd"],
        "q_mvar": bus.loc[load_mask, "Qd"]
    })

    # --- Branch results (as impedance) ---
    flows = pd.DataFrame(branch_array[:, 13:17], columns=["Pf", "Qf", "Pt", "Qt"])

    net.res_impedance = pd.DataFrame({
        "p_from_mw": flows.loc[:, "Pf"],
        "q_from_mvar": flows.loc[:, "Qf"],
        "p_to_mw": flows.loc[:, "Pt"],
        "q_to_mvar": flows.loc[:, "Qt"]
    })

    return net


def reorder_bus_ids(mat):
    if max(mat["bus"][:, 0]) == mat["bus"].shape[0]:
        # Bus numbering is sequential
        return mat
    else:
        old = mat["bus"][:, 0].astype(np.int64)
        new = np.arange(1, len(old) + 1, dtype=np.int64)

        lookup = np.empty(old.max() + 1, dtype=np.int64)
        lookup[old] = new

        mat["bus"][:, 0] = new
        mat["gen"][:, 0] = lookup[mat["gen"][:, 0].astype(np.int64)]
        mat["branch"][:, :2] = lookup[mat["branch"][:, :2].astype(np.int64)]
        return mat


def swap_branch_buses(mat):
    mask = mat["branch"][:, 0] > mat["branch"][:, 1]
    if mask.any():
        # Swap from/to bus IDs, Pf <-> Pt and Qf <-> Qt
        mat["branch"][np.ix_(mask, [0, 1, 13, 14, 15, 16])] = \
            mat["branch"][np.ix_(mask, [1, 0, 15, 16, 13, 14])]
    return mat


def mp_to_canonical(case_path, output_path, export_net):
    """
    Read the .mat case file generated by MATPOWER, convert it to the canonical CSV formats
    and export it to the output path
    """
    try:
        # Reader
        mat = loadmat(case_path)

        # Fix busIDs if they are non-sequential (eg: IEEE 300 bus system)
        mat = reorder_bus_ids(mat)

        # Swap Fbus & Tbus of branches and their results where FbusID>TbusID
        mat = swap_branch_buses(mat)

        # TODO: Validate if the mat dictionary has necessary keys and corresponding numpy arrays of expected width.

        # Converter
        net = pp.create_empty_network()
        net = mat_to_pp(mat, net)

        # Exporter
        if export_net:
            net_exporter(net, output_path)

        return net
    except Exception as e:
        msg = f"Failed to load data from {case_path}: {e}"
        logger.error(msg)
        raise IOError(msg)
