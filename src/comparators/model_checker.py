# #####################################################################
# Check if network details used for powerflow calculation are identical
# #####################################################################
import numpy as np
import pandapower as pp
from pandapower.pypower.makeYbus import makeYbus
from src.comparators.comparators import compare_np
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
    Check if topology, impedances, gen and load data are matching in the 2 models
    case_a and case_b are in the canonical format i.e. PandaPower's pandapowerNet datatype, which contain data as DFs
    """
    # Convert pandapower net to PYPOWER case (ppc)
    ppcs = ()
    for case in [case_a, case_b]:
        # Convert pandapower net to MATPOWER/PYPOWER case
        ppcs += (pp.converter.to_ppc(case),)

    # Match network model
    res = compare_np(ppcs[0]["bus"], ppcs[1]["bus"], filename="Bus data", check_row_size=True, match_columns=[0])
    res = compare_np(ppcs[0]["branch"], ppcs[1]["branch"], filename="Branch data", tol = 10e-6,
                     check_row_size=True,  sort_columns=[0,1], match_columns=[range(5)])
    res = compare_np(ppcs[0]["gen"], ppcs[1]["gen"], filename="Gen data", tol = 10e-6,
                     check_row_size=True,  sort_columns=[0], match_columns=[range(9)])

    # Match connectivity & impedences
    ybus_matrices = ()
    for ppc in ppcs:

        ppc = pp.converter.to_ppc(case)

        # Call function to compute Ybus
        ybus_matrices += (compute_ybus_matric(ppc),)

    res = match_scalar_matrices(config, ybus_matrices)

    return