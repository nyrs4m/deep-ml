import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """

    # 1. Input Validation for dimensions
    if arr.ndim not in (1, 2):
        raise ValueError("Input array must be 1D or 2D.")
        
    if norm_type == 'frobenius' and arr.ndim != 2:
        raise ValueError("The 'frobenius' norm is a matrix norm and requires a 2D array.")

    # 2. Compute Entrywise Norms
    if norm_type == 'l1':
        # Sum of absolute values of all elements
        return float(np.sum(np.abs(arr)))
        
    elif norm_type == 'l2':
        # Square root of the sum of squares of all elements
        return float(np.sqrt(np.sum(np.square(arr))))
        
    elif norm_type == 'linf':
        # Maximum absolute value among all elements
        return float(np.max(np.abs(arr)))
        
    elif norm_type == 'frobenius':
        # Square root of sum of absolute squares (for 2D)
        return float(np.sqrt(np.sum(np.square(arr))))
        
    else:
        raise ValueError(f"Unknown norm type: '{norm_type}'. Expected 'l1', 'l2', 'linf', or 'frobenius'.")
