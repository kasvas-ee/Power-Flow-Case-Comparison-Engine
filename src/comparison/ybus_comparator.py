import numpy as np
from scipy.sparse import csr_matrix, issparse

from src.logger import logger
from src.comparison.result_structures import YbusComparisonResult


def _as_csr(matrix):
    """
    Convert a Ybus matrix to CSR format without densifying it.
    """

    if issparse(matrix):
        return matrix.tocsr()

    return csr_matrix(np.asarray(matrix))


def _count_pattern_differences(matrix_a, matrix_b, tol=0):
    """
    Count locations where one Ybus has a value above tolerance
    and the other does not.
    """

    pattern_a = matrix_a.copy()
    pattern_b = matrix_b.copy()

    # Eliminate intentionally stored zeros
    pattern_a.eliminate_zeros()
    pattern_b.eliminate_zeros()

    difference = pattern_a.astype(bool) != pattern_b.astype(bool)

    return difference.nnz


def _compare_values(matrix_a, matrix_b, tol):
    """
    Compare numerical Ybus values without densifying the matrices.

    Returns:
        value_diff_count
        max_value_diff
    """

    difference = matrix_a - matrix_b

    if difference.nnz == 0:
        return 0, 0.0

    abs_difference = np.abs(difference.data)

    value_diff_mask = abs_difference > tol

    value_diff_count = np.sum(value_diff_mask)

    if value_diff_count == 0:
        return 0, 0.0

    max_value_diff = float(np.max(abs_difference))

    return value_diff_count, max_value_diff


def compare_ybus(ybus_a, ybus_b, tol=1e-9):
    """
    Compare two Ybus sparse matrices.

    Comparison is performed in three stages:

    1. Matrix dimensions / bus count
    2. Sparsity pattern / topology
    3. Numerical admittance values

    Memory optim: Sparse matrices remain sparse throughout the comparison.
    """

    logger.info("Starting Ybus comparison")
    ybus_outcome = YbusComparisonResult()
    ybus_a = _as_csr(ybus_a)
    ybus_b = _as_csr(ybus_b)

    # ---------------------------------------------------------
    # 1. Bus count / matrix dimensions
    # ---------------------------------------------------------

    ybus_outcome.shape_a = ybus_a.shape
    ybus_outcome.shape_b = ybus_b.shape

    bus_count_match = ybus_a.shape == ybus_b.shape
    if not bus_count_match:
        logger.error("Bus count mismatch: Ybus shape mismatch: A={}, B={}".format(ybus_a.shape, ybus_b.shape))
        ybus_outcome.reason = "Bus count mismatch: Ybus matrix shapes are different."
        return ybus_outcome
    ybus_outcome.bus_count_match = True

    # ---------------------------------------------------------
    # 2. Topology / sparsity pattern
    # ---------------------------------------------------------

    pattern_diff_count = _count_pattern_differences(ybus_a, ybus_b)
    ybus_outcome.pattern_diff_count = pattern_diff_count

    topology_match = pattern_diff_count == 0
    if not topology_match:
        logger.error("Topology mismatch: {} entries in Ybus sparcity patterns differ.".format(pattern_diff_count))
        ybus_outcome.reason = "Topology mismatch: Ybus sparsity patterns differ."
        return ybus_outcome
    ybus_outcome.topology_match = True

    # ---------------------------------------------------------
    # 3. Numerical Ybus values
    # ---------------------------------------------------------

    value_diff_count, max_value_diff = _compare_values(ybus_a, ybus_b, tol)
    ybus_outcome.value_diff_count = value_diff_count
    ybus_outcome.max_value_diff = max_value_diff

    ybus_value_match = value_diff_count == 0
    if not ybus_value_match:
        logger.warning("Ybus value mismatch: {} admittance values differ, "
                       "maximum difference = {}".format(value_diff_count, max_value_diff))
        ybus_outcome.reason = "Ybus value mismatch: admittance values differ beyond tolerance."
        return ybus_outcome
    ybus_outcome.ybus_value_match = True
    ybus_outcome.is_match=True
    msg = "Ybus matrices match within the specified tolerance."
    logger.warning(msg)
    ybus_outcome.reason = msg
    return ybus_outcome
