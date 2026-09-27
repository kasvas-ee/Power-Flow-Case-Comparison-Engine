# #####################################################################
# Check if network details used for powerflow calculation are identical
# #####################################################################
import numpy as np
import pandapower as pp
from pandapower.pypower.makeYbus import makeYbus
from src.comparators.comparators import compare_ybus, compare_np
from src.logging_config import logger


def compute_ybus_matric(ppc):
    """
    Compute the Ybus matrix for a given bus and branch data
    """
    ppc["branch"] = np.pad(ppc["branch"], ((0, 0), (0, 26 - ppc["branch"].shape[1])))  # This must always be 20 cols

    # Build Ybus, Yf, Yt
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    return Ybus


def match_model(config, case_a, case_b):
    """
    Step1: Check size of Bus, branch, Gen and load. If mismatch in drastic, abort.
    Step2: From Ybus, check topology mismatch and impedance/shunt mismatch.
        If there is a mismatch, match each asset type in detail.
    Step3: Check if gen and load data are matching in the 2 models
    case_a and case_b are in the canonical format i.e. PandaPower's pandapowerNet datatype, which contain data as DFs
    """
    # Convert pandapower net to PYPOWER case (ppc)
    ppcs = ()
    for case in [case_a, case_b]:
        # Convert pandapower net to MATPOWER/PYPOWER case
        ppcs += (pp.converter.to_ppc(case),)

    # Match Ybus for topology and impedance/shunt match
    ybus_matrices = ()
    for ppc in ppcs:
        ppc = pp.converter.to_ppc(case)

        # Call function to compute Ybus
        ybus_matrices += (compute_ybus_matric(ppc),)
    summary = compare_ybus(*ybus_matrices, tol=1e-6)
    # TODO: Set correct tolerances
    logger.info(summary)

    if not summary["bus_count_match"]:
        msg = "Number of busses are different. Aborting comparison."
        logger.error(msg)
        raise ReferenceError(msg)

    if summary["topology_match"] and summary["impedance_match"]:
        load_columns = [0,2,3]
        summary["load_match"] = compare_np(ppcs[0]["bus"][:, load_columns], ppcs[1]["bus"][:, load_columns], filename="Load data",
                         check_row_size=True, match_columns=[0])
    else:
        summary["bus_match"] = compare_np(ppcs[0]["bus"], ppcs[1]["bus"], filename="Bus data",
                         check_row_size=True, match_columns=[0])
        summary["branch_match"] = compare_np(ppcs[0]["branch"], ppcs[1]["branch"], filename="Branch data", tol=10e-6,
                         check_row_size=True, sort_columns=[0, 1], match_columns=[range(5)])
        summary["load_match"] = summary["branch_match"]

    # Match gen model
    summary["gen_match"] = compare_np(ppcs[0]["gen"], ppcs[1]["gen"], filename="Gen data", tol = 10e-6,
                     check_row_size=True,  sort_columns=[0], match_columns=[range(9)])
    # TODO: Set correct tolerances

    return all(summary[k] for k in ["topology_match", "impedance_match", "load_match", "gen_match"])
