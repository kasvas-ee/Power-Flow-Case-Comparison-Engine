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


def _count_pattern_differences(matrix_a, matrix_b, tol):
    """
    Count locations where one Ybus has a value above tolerance
    and the other does not.
    """

    pattern_a = matrix_a.copy()
    pattern_b = matrix_b.copy()

    pattern_a.data = (
        np.abs(pattern_a.data) > tol
    ).astype(np.int8)

    pattern_b.data = (
        np.abs(pattern_b.data) > tol
    ).astype(np.int8)

    pattern_a.eliminate_zeros()
    pattern_b.eliminate_zeros()

    difference = pattern_a != pattern_b

    return int(difference.nnz)


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

    value_diff_count = int(
        np.sum(value_diff_mask)
    )

    if value_diff_count == 0:
        return 0, 0.0

    max_value_diff = float(
        np.max(abs_difference)
    )

    return value_diff_count, max_value_diff


def compare_ybus(Ybus_a, Ybus_b, tol=1e-9):
    """
    Compare two Ybus matrices.

    Comparison is performed in three stages:

    1. Matrix dimensions / bus count
    2. Sparsity pattern / topology
    3. Numerical admittance values

    Sparse matrices remain sparse throughout the comparison.
    """

    logger.info("Starting Ybus comparison")

    Ybus_a = _as_csr(Ybus_a)
    Ybus_b = _as_csr(Ybus_b)

    shape_a = Ybus_a.shape
    shape_b = Ybus_b.shape

    # ---------------------------------------------------------
    # 1. Bus count / matrix dimensions
    # ---------------------------------------------------------

    bus_count_match = shape_a == shape_b

    if not bus_count_match:
        logger.error(
            "Ybus shape mismatch: A=%s, B=%s",
            shape_a,
            shape_b,
        )

        return YbusComparisonResult(
            is_match=False,
            bus_count_match=False,
            topology_match=False,
            ybus_value_match=False,
            shape_a=shape_a,
            shape_b=shape_b,
            reason=(
                "Bus count mismatch: "
                "Ybus matrix shapes are different."
            ),
        )

    # ---------------------------------------------------------
    # 2. Topology / sparsity pattern
    # ---------------------------------------------------------

    pattern_diff_count = _count_pattern_differences(
        Ybus_a,
        Ybus_b,
        tol,
    )

    topology_match = pattern_diff_count == 0

    if not topology_match:
        logger.warning(
            "Ybus topology mismatch: %d entries differ",
            pattern_diff_count,
        )

        return YbusComparisonResult(
            is_match=False,
            bus_count_match=True,
            topology_match=False,
            ybus_value_match=False,
            pattern_diff_count=pattern_diff_count,
            value_diff_count=None,
            max_value_diff=None,
            shape_a=shape_a,
            shape_b=shape_b,
            reason=(
                "Topology mismatch: "
                "Ybus sparsity patterns differ."
            ),
        )

    # ---------------------------------------------------------
    # 3. Numerical Ybus values
    # ---------------------------------------------------------

    value_diff_count, max_value_diff = _compare_values(
        Ybus_a,
        Ybus_b,
        tol,
    )

    ybus_value_match = value_diff_count == 0

    if not ybus_value_match:
        logger.warning(
            "Ybus value mismatch: %d entries differ, "
            "maximum difference = %s",
            value_diff_count,
            max_value_diff,
        )

        return YbusComparisonResult(
            is_match=False,
            bus_count_match=True,
            topology_match=True,
            ybus_value_match=False,
            pattern_diff_count=pattern_diff_count,
            value_diff_count=value_diff_count,
            max_value_diff=max_value_diff,
            shape_a=shape_a,
            shape_b=shape_b,
            reason=(
                "Ybus value mismatch: "
                "admittance values differ beyond tolerance."
            ),
        )

    # ---------------------------------------------------------
    # 4. Full Ybus match
    # ---------------------------------------------------------

    logger.info(
        "Ybus matrices match within tolerance."
    )

    return YbusComparisonResult(
        is_match=True,
        bus_count_match=True,
        topology_match=True,
        ybus_value_match=True,
        pattern_diff_count=0,
        value_diff_count=0,
        max_value_diff=0.0,
        shape_a=shape_a,
        shape_b=shape_b,
        reason=(
            "Ybus matrices match within the "
            "specified tolerance."
        ),
    )