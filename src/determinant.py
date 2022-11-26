"""Pure Python/NumPy generic determinant calculator

Tools to calculate determinants, including of minor-matrix determinants.
The main goal is to work with NumPy array of any 'dtype', like:
  fractions, decimal, numbers, numpy.polynomial.Polynomial(), etc.
The determinant.det() should work just like numpy.linalg.det(), with preserved 'dtype'.
"""
import itertools
import numpy as np


# Limit the size of a determinants to be calculated at once, in order to:
# - Avoid precision loss, due to huge intermediate values, leading to incorrect result,
#   like: "1e6**3 + 1 - 1e6**3 == 0"
# - Better performance
MAX_DET_SIZE = 3

ASSERT_LEVEL = 1

# Precalculated permutations to avoid 'itertools' for small sizes
PRECALC_PERMS = list(np.array(vals, np.int8) for vals in (
        [[]],               # size=0
        [[0]],              # size=1
        [[0, 1], [1, 0]],   # size=2
    ))

def _ref_perm_parity(perm_idxs: np.array) -> np.array:
    """Obtain the parity of permulations from the number of inversions

    Uses nested for-loops, as a reference for pytests
    See:
        https://statlect.com/matrix-algebra/sign-of-a-permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation
    """
    odd_mask = False
    # Plain nested for-loops implementation
    for i in range(perm_idxs.shape[-1] - 1):
        for j in range(i + 1, perm_idxs.shape[-1]):
            odd_mask = odd_mask ^ (perm_idxs[..., i] > perm_idxs[..., j])
    return odd_mask

def permutation_parity(perm_idxs: np.array) -> np.array:
    """Obtain the parity of permulations from the number of inversions"""
    # Indices of 'perm_idxs' elements, to calculate inversions/parity at once:
    # the combinations of their number over 2
    idxs = np.fromiter(itertools.combinations(range(perm_idxs.shape[-1]), 2), dtype=(int, 2))

    # Regroup permutation indices in couples to check for inversions
    idxs = perm_idxs[...,idxs]
    odd_mask = np.logical_xor.reduce(idxs[...,0] > idxs[...,1], axis=-1)

    if ASSERT_LEVEL > 3:
        # Use a reference result
        assert (odd_mask == _ref_perm_parity(perm_idxs)).all(), 'Wrong optimized permutation parity'
    return odd_mask

def permutations_indices(size: int) -> np.array:
    """Obtain all permulations for specific number range"""
    if size < len(PRECALC_PERMS):
        return PRECALC_PERMS[size]
    # Use int8 as allocation of more than "factorial(128)" elements is impossible anyway
    return np.fromiter(itertools.permutations(range(size)),
            dtype=(np.int8, [size]))

def combinations_parity(comb_mask) -> np.array:
    """Obtain the parity of combinations from the number of inversions"""
    odd_mask = np.logical_xor.accumulate(~comb_mask, axis=-1)
    odd_mask[~comb_mask] = False
    odd_mask = np.logical_xor.reduce(odd_mask, axis=-1)

    if ASSERT_LEVEL > 3:
        # Combine permutation indices for the reference result
        combs = take_by_masks(np.arange(comb_mask.shape[-1]), comb_mask)
        rems = take_by_masks(np.arange(comb_mask.shape[-1]), ~comb_mask)
        perm = np.concatenate((combs, rems), axis=-1)
        # Use a reference result
        assert (odd_mask == _ref_perm_parity(perm)).all(), 'Wrong optimized combinations parity'
    return odd_mask

def combinations_masks(size: int, comb_size: int) -> np.array:
    """Obtain all combinations for specific number range

    Note: The returned mask can be inverted to get the remainder from combination"""
    if comb_size == 1:
        return np.identity(size, dtype=bool)
    if comb_size == size - 1:
        return ~np.identity(size, dtype=bool)[::-1]
    idxs = np.fromiter(itertools.combinations(range(size), comb_size),
            dtype=(np.int8, [comb_size]))
    masks = np.zeros(shape=(idxs.shape[0], size), dtype=bool)
    masks[np.arange(idxs.shape[0])[...,np.newaxis], idxs] = True
    return masks

def take_by_masks(data: np.array, masks: np.array):
    """Extract elements by masking the last dimension, keeping the mask's shape"""
    # Broadcast the last dimension to match 'masks'
    data = np.broadcast_to(data[...,np.newaxis,:],
            shape=data.shape[:-1] + masks.shape)
    # The reshape() below expects, the masking to extract
    # equal number of elements from each row
    if ASSERT_LEVEL > 1:
        mask_dims = np.count_nonzero(masks, axis=-1)
        assert (mask_dims == mask_dims[0]).all(), 'Mask is not uniform'
    return data[..., masks].reshape(data.shape[:-1] + (-1,))

def take_data_matrix(data: np.array, col_idxs: np.array, row_base: int) -> np.array:
    """Callback to extract matrix data for determinant calculation

    Parameters
    ----------
    data : (..., M, M) array_like
        The source to retrieve data from
    col_idxs : (..., P, N) array_like
        The columns to be read, grouped in the last dimension
    row_base : int
        The index of row to start reading, i.e. row for col_idxs[..., 0],
        others are increasing by one

    Returns
    -------
    data : (..., P, N) array_like
        The data at corresponding `col_idxs` starting at `row_base`
    """
    return data[..., np.arange(col_idxs.shape[-1]) + row_base, col_idxs]

def take_data_numwall(data: np.array, num_rows: int, num_dets: int or None,
        col_idx: np.array, row_base: int) -> np.array:
    """Callback to convert data to number-wall style matrix

    Parameters
    ----------
    num_rows : int
        Number of rows in the topmost number-wall matrix
    num_dets : int or None
        Number of sequential determinants to be returned (for parallel processing)
    - Others: see take_data_matrix()
    """
    # Convert indices using the number-wall style data-order
    col_idx += num_rows - 1 - (np.arange(col_idx.shape[-1]) + row_base)
    if num_dets is not None:
        # Retrieve data for multiple determinants starting at each possible index
        col_idx = col_idx + np.arange(num_dets).reshape((-1,) + (1,) * col_idx.ndim)
    return data[..., col_idx]

def det_sum_simple(products: np.array, odd_masks: np.array) -> np.array:
    """Simple determinat sum implementation

    Sum the products by negating the odd-permutations
    """
    res = np.negative(products, out=products, where=odd_masks)
    return res.sum(-1)

def det_sum_split(products: np.array, odd_masks: np.array) -> np.array:
    """Split determinat sum implementation

    This should reduce the float precision loss in intermediate sum() results
    """
    even_res = res[..., ~odd_masks]
    odd_res = res[..., odd_masks]
    # First sum the common pairs of even and odd permutations
    comm_size = min(even_res.shape[-1], odd_res.shape[-1])
    res = (even_res[..., :comm_size] - odd_res[..., :comm_size]).sum(-1)
    # Then add the remainder from even or odd ones
    return res + even_res[..., comm_size:].sum(-1) - odd_res[..., comm_size:].sum(-1)

def det_minors_of_columns(take_data: callable, col_idxs: np.array, row_base: int, *,
        minor_size: int, left_only=False,
        max_det_size: int = MAX_DET_SIZE, det_sum: callable = det_sum_simple) -> np.array:
    """Minor determinants of all combinations of sub-matrices, return left and/or right ones"""
    # Represent the sub-matrix combinations as masks to easily switch from left to right side
    masks = combinations_masks(col_idxs.shape[-1], minor_size)
    # Calculate left-side minors
    res = det_of_columns(take_data, take_by_masks(col_idxs, masks),
            row_base, max_det_size=max_det_size, det_sum=det_sum)
    if left_only:
        res = res, None
    else:
        # Calculate right-side minors (optional)
        res = res, det_of_columns(take_data, take_by_masks(col_idxs, ~masks),
                row_base + minor_size, max_det_size=max_det_size, det_sum=det_sum)

    # Third element is the parity of combination
    res = *res, combinations_parity(masks)
    return res

def det_of_columns(take_data: callable, col_idxs: np.array, row_base: int, *,
        max_det_size: int = MAX_DET_SIZE, det_sum: callable = det_sum_simple) -> np.array:
    """Matrix determinant calculation on given combinations of indices

    Parameters
    ----------
    take_data : function(I, i) -> (V)
        Callback to retrieve actual data, see take_data_matrix()
    col_idxs : array_like
        Indices of matrix columns to calculate determinant
    row_base : int
        Index of fist row to calculate determinant, the number of rows is
        `col_idxs.shape[-1]`
    max_det_size : int, optional
        Max size of intermediate sub-matrices to avoid overflows
    det_sum : function(V, M) -> (S)
        Callback to do the final determinant sum, see det_sum_simple()
    """
    # Calculate only the determinants, that are small enough
    if col_idxs.shape[-1] <= max_det_size:
        # Combine all permutations in a single array
        idxs = permutations_indices(col_idxs.shape[-1])
        perms = col_idxs[..., idxs]

        res = take_data(perms, row_base)
        del perms
        # Apply determinant rule: products from permutations
        res = res.prod(-1)

        # Get the permutation parity
        odd_masks = permutation_parity(idxs)
        del idxs
    else:
        # Split each determinant into two minor-determinants (from sub-matrices):
        # main (left-side) minor and remainder (right-size) minor
        split = (col_idxs.shape[-1] + 1) // 2
        minors, r_minors, odd_masks = det_minors_of_columns(take_data, col_idxs, row_base,
                minor_size = split, left_only=False,
                max_det_size=max_det_size, det_sum=det_sum)
        # Apply determinant rule: products from sub-determinants
        res = minors * r_minors
        del minors, r_minors

    # Apply determinant rule: sum the products by negating the odd-permutations
    return det_sum(res, odd_masks)

def det(data: np.array, *, max_det_size: int = MAX_DET_SIZE) -> np.array:
    """Compute the determinant of an array - our version of numpy.linalg.det()

    Parameters
    ----------
    data : (..., M, M) array_like
        Input array to compute determinants for.
    max_det_size : int, optional
        Max size of intermediate sub-matrices to avoid overflows

    Returns
    -------
    det : (...) array_like
        Determinant of `data`.
    """
    data = np.asarray(data)
    assert data.shape[-1] == data.shape[-2], 'Non-square data matrix'

    def take_data(col_idxs, row_base):
        return take_data_matrix(data, col_idxs, row_base)
    return det_of_columns(take_data, np.arange(data.shape[-1]), 0,
            max_det_size=max_det_size)

def det_minors(data: np.array, *, max_det_size: int = MAX_DET_SIZE) -> np.array:
    """Compute the minor determinants of an array

    Parameters
    ----------
    data : (..., M, N) array_like
        Input array to compute minor determinants for (`M < N`).

    - Others: see det()
    """
    data = np.asarray(data)
    assert data.shape[-1] >= data.shape[-2], 'Unexpected shape of data matrix'

    def take_data(col_idxs, row_base):
        return take_data_matrix(data, col_idxs, row_base)
    res, _, odd_masks = det_minors_of_columns(take_data, np.arange(data.shape[-1]), 0,
            minor_size=data.shape[-2], left_only=True,
            max_det_size=max_det_size)
    return np.negative(res, out=res, where=odd_masks)
