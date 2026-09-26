import numpy as np
from src.logging_config import logger


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
