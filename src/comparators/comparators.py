import numpy as np
from scipy.sparse import issparse
from src.logging_config import logger


def compare_ybus(Ybus_a, Ybus_b, tol=1e-9):
    """
    Compare two Ybus matrices from pandapower.

    Parameters
    ----------
    Ybus_a, Ybus_b : np.ndarray or scipy.sparse matrix
        Bus admittance matrices to compare.
    tol : float
        Numerical tolerance for impedance mismatch.

    Returns
    -------
    dict
        Summary of mismatches:
        {
            "topology_match": bool,
            "impedance_match": bool,
            "pattern_diff_count": int,
            "value_diff_count": int,
            "max_value_diff": float
        }
    """

    logger.info("Starting Ybus comparison")

    # Convert sparse to dense for comparison
    if issparse(Ybus_a):
        Ybus_a = Ybus_a.toarray()
    if issparse(Ybus_b):
        Ybus_b = Ybus_b.toarray()

    bus_count_match = Ybus_a.shape == Ybus_b.shape
    # Check shape
    if not bus_count_match:
        logger.error(f"Shape mismatch: A={Ybus_a.shape}, B={Ybus_b.shape}")
        return {
            "bus_count_match":False,
            "topology_match": False,
            "impedance_match": False,
            "pattern_diff_count": None,
            "value_diff_count": None,
            "max_value_diff": None
        }

    # Sparsity pattern (nonzero locations)
    pattern_a = (np.abs(Ybus_a) > tol)
    pattern_b = (np.abs(Ybus_b) > tol)
    pattern_diff = np.logical_xor(pattern_a, pattern_b)
    pattern_diff_count = np.sum(pattern_diff)

    topology_match = pattern_diff_count == 0
    if topology_match:
        logger.info("Topology (sparsity pattern) matches")
    else:
        logger.warning(f"Topology mismatch: {pattern_diff_count} entries differ")

    # Value comparison
    value_diff = np.abs(Ybus_a - Ybus_b)
    value_diff_mask = value_diff > tol
    value_diff_count = np.sum(value_diff_mask)
    max_value_diff = np.max(value_diff) if value_diff_count > 0 else 0.0

    impedance_match = value_diff_count == 0
    if impedance_match:
        logger.info("Impedance values match within tolerance")
    else:
        logger.warning(f"Impedance mismatch: {value_diff_count} entries differ, "
                       f"max difference = {max_value_diff}")

    return {
        "bus_count_match": bus_count_match,
        "topology_match": topology_match,
        "impedance_match": impedance_match,
        "pattern_diff_count": int(pattern_diff_count),
        "value_diff_count": int(value_diff_count),
        "max_value_diff": float(max_value_diff)
    }


def compare_np(np_a, np_b, filename, tol=0, check_row_size=False, check_col_size=False,
               sort_columns=None, match_columns=None):
    """
    Compare two numpy arrays within a given tolerance:
    if check_row_size is True, the number of rows have to match exactly
    if check_col_size is True, the number of columns have to match exactly
    if sort_columns is a list of int, the numpy array is sorted over these columns before matching contents for exact match
    if match_columns is a list of int, these columns need to match exactly
    """
    match = True
    # Ensure inputs are numpy arrays
    np_a = np.asarray(np_a)
    np_b = np.asarray(np_b)

    logger.debug(f"{filename}: Array A shape: {np_a.shape}, Array B shape: {np_b.shape}")

    # Check row size
    if check_row_size and np_a.shape[0] != np_b.shape[0]:
        logger.error(f"Row size mismatch: {filename} has {np_a.shape[0]} rows in A but {np_b.shape[0]} rows in B")
        match = False

    # Check column size
    if check_col_size and np_a.shape[1] != np_b.shape[1]:
        logger.error(f"Column size mismatch: {filename} has {np_a.shape[1]} columns in A but {np_b.shape[1]} columns in B")
        match = False

    # Sort arrays if requested
    if sort_columns is not None:
        np_a = np_a[np.lexsort([np_a[:, col] for col in reversed(sort_columns)])]
        np_b = np_b[np.lexsort([np_b[:, col] for col in reversed(sort_columns)])]

    # Check match_columns exactly
    if match_columns is not None:
        diffs = np.abs(np_a[:, match_columns] - np_b[:, match_columns])
        max_diff = np.max(diffs)
        logger.debug(f"Max difference in match_columns = {max_diff}")
        if max_diff > tol:
            logger.error(f"Mismatch in {filename}. Max difference {max_diff} exceeds tolerance {tol}")
            mismatches = np.argwhere(diffs > tol)
            logger.error(f"Arrays differ at {len(mismatches)} positions beyond tolerance")
            for idx in mismatches[:10]:  # log first 10 mismatches
                logger.debug(
                    f"Mismatch at {tuple(idx)}: A={np_a[tuple(idx)]}, B={np_b[tuple(idx)]}, diff={diffs[tuple(idx)]}")
            match = False
        else:
            logger.debug(f"Max difference {max_diff} within tolerance {tol}")
    if match:
        logger.info(f"{filename} is a good match")
    return match

