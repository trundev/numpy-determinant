"""Pure Python/NumPy generic determinant calculator

Tools to calculate determinants, including of minor-matrix determinants.
The main goal is to work with NumPy array of any 'dtype', like:
  fractions, decimal, numbers, numpy.polynomial.Polynomial(), etc.
The determinant.det() should work just like numpy.linalg.det(), with preserved 'dtype'.
"""
import numpy as np
import combinatorics


# Limit the size of a determinants to be calculated at once, in order to:
# - Avoid precision loss, due to huge intermediate values, leading to incorrect result,
#   like: "1e6**3 + 1 - 1e6**3 == 0"
# - Better performance
MAX_DET_SIZE = 3

ASSERT_LEVEL = 1

# Try to avoid too large intermediate arrays
ARR_THRESHOLD = int(50e6)

def take_by_masks(data: np.array, masks: np.array):
    """Extract elements by masking the last dimension, keeping the mask's shape"""
    # Expand/broadcast the last dimension to match 'masks'
    shape = list(data.shape)
    shape[-1:-1] = (1,)*(masks.ndim - 1)
    data = np.broadcast_to(data.reshape(shape), shape=data.shape[:-1] + masks.shape)
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

def det_minors_of_masks(det_of_masks: callable, start_comb: combinatorics.combination, row_base: int, *,
        num_rows: int, minors: np.array or None = None,
        max_det_size: int = MAX_DET_SIZE, det_sum: callable = det_sum_simple) -> np.array:
    """Minor determinants built on top of combinations of sub-matrices"""
    ###HACK:
    if row_base == 0:
        start_comb._bool_mask = start_comb._bool_mask[0]
    ###
    # Split each determinant into multiple minor-determinants (from sub-matrices):
    row_step = max_det_size
    while row_base < num_rows:
        if row_step >= num_rows - row_base:
            row_step = num_rows - row_base
        # Limit row step, to avoid too large intermediate arrays
        if ARR_THRESHOLD < start_comb.count * np.math.comb(start_comb.size - row_base, row_step):
            row_step = 1
        # Next combinations, based on the remainders from current ones
        start_comb, rem_comb = start_comb.remainder_combination(row_step)

        # Calculate the next minors
        r_minors = det_of_masks(rem_comb.to_bool_mask(), row_base,
                max_det_size=max_det_size, det_sum=det_sum)
        if minors is not None:
            minors = (minors[...,np.newaxis] * r_minors).reshape(minors.shape[:-1] + (-1,))
        else:
            minors = r_minors
        del r_minors

        # Calculate the next combination masks and parity
        odd_masks = start_comb.combinations_parity(rem_comb).flatten()
        start_comb = start_comb.merge(rem_comb)
        del rem_comb

        # Build remap-xform matrix to sum the duplicate combinations (identify overlaps)
        if not start_comb.is_singular():
            comb_idxs = start_comb.to_comb_index()
            expected_dups = np.math.comb(row_base + row_step, row_step)
            remap_idxs = np.argsort(comb_idxs, axis=None).reshape(-1, expected_dups)
            del comb_idxs
        else:
            # Final row(s), all combinations overlap
            remap_idxs = np.arange(start_comb.count)

        # Drop the duplicated combimations - the remap places them in the pre-last dimension
        comb_masks = start_comb.to_bool_mask()
        comb_masks = comb_masks.reshape(-1, comb_masks.shape[-1])
        if ASSERT_LEVEL > 3:
            np.testing.assert_equal(comb_masks[remap_idxs[...,1:]], comb_masks[remap_idxs[...,:-1]],
                    err_msg='Unable to isolate "comb_masks" duplicates')
        start_comb = combinatorics.combination_bools.from_bool_mask(comb_masks[remap_idxs[...,0]])
        del comb_masks

        # Apply determinant rule: sum the products by negating the odd-permutations
        minors = minors[...,remap_idxs]
        odd_masks = odd_masks[remap_idxs]
        minors = det_sum(minors, odd_masks)
        del remap_idxs, odd_masks

        row_base += row_step

    return minors, start_comb

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
        perm = combinatorics.permutation(col_idxs.shape[-1])
        col_idxs = col_idxs[..., perm.to_indices()]

        res = take_data(col_idxs, row_base)
        del col_idxs
        # Apply determinant rule: products from permutations
        res = res.prod(-1)

        # Get the permutation parity
        odd_masks = perm.permutation_parity()
        del perm

        # Apply determinant rule: sum the products by negating the odd-permutations
        return det_sum(res, odd_masks)

    def det_of_masks(masks, row_base, *, max_det_size, det_sum):
        """Wrapper of det_of_columns() to use combination masks instead of column indices"""
        return det_of_columns(take_data, take_by_masks(col_idxs, masks), row_base,
                max_det_size=max_det_size, det_sum=det_sum)
    # Build on top of empty initial (left-side) minors
    num_rows = col_idxs.shape[-1]
    minors, _ = det_minors_of_masks(det_of_masks, combinatorics.combination_bools(num_rows, 0), row_base,
            num_rows=num_rows, max_det_size=max_det_size, det_sum=det_sum)
    return minors

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
    col_idxs = np.arange(data.shape[-1])
    def det_of_masks(masks, row_base, *, max_det_size, det_sum):
        """Wrapper of det_of_columns() to use combination masks instead of column indices"""
        return det_of_columns(take_data, take_by_masks(col_idxs, masks), row_base,
                max_det_size=max_det_size, det_sum=det_sum)

    # Build on top of empty initial (left-side) minors
    minors, combs = det_minors_of_masks(det_of_masks, combinatorics.combination_bools(data.shape[-1], 0), 0,
            num_rows=data.shape[-2], max_det_size=max_det_size)
    odd_masks = combs.combinations_parity()
    return np.negative(minors, out=minors, where=odd_masks)
