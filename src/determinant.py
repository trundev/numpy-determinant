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

# Try to avoid too large intermediate arrays
ARR_THRESHOLD = int(50e6)

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
    """Obtain the parity of permulations from the number of inversions

    Parameters
    ----------
    perm_idxs : (..., N) array_like
        Array of permulations represented as indices, individual permulations are in the last dimension

    Returns
    -------
    odd_mask : (...) array_like
        Array of booleans for each permulation, value is 'True' where the parity is odd
    """
    # Indices of 'perm_idxs' elements, to calculate inversions/parity at once:
    # the combinations of their number over 2
    idxs = np.fromiter(itertools.combinations(range(perm_idxs.shape[-1]), 2), dtype=(int, 2))

    # Regroup permutation indices in couples to check for inversions
    idxs = perm_idxs[...,idxs]
    odd_mask = np.logical_xor.reduce(idxs[...,0] > idxs[...,1], axis=-1)

    if ASSERT_LEVEL > 3:
        # Use a reference result
        np.testing.assert_equal(odd_mask, _ref_perm_parity(perm_idxs), err_msg='Wrong optimized permutation parity')
    return odd_mask

def permutations_indices(size: int) -> np.array:
    """Obtain all permulations for specific number range"""
    if size < len(PRECALC_PERMS):
        return PRECALC_PERMS[size]
    # Use int8 as allocation of more than "factorial(128)" elements is impossible anyway
    return np.fromiter(itertools.permutations(range(size)),
            dtype=(np.int8, [size]))

def combinations_parity(comb_mask: np.array, rem_mask: np.array or None = None) -> np.array:
    """Obtain the parity of combinations from the number of inversions

    Parameters
    ----------
    comb_mask : (..., N) array_like
        Array of combinations represented as mask, individual combinations are in the last dimension
    rem_mask : (..., N) array_like, optional
        Array of combination remainders to compare with. The number of inversions is the swaps between two
        adjacent elements, in order to move all the `rem_mask` elements after `comb_mask`.
        When `None`, the inverted `comb_mask` is used instead, which counts the swaps to move all `comb_mask`
        elements in front.

    Returns
    -------
    odd_mask : (...) array_like
        Array of booleans for each combination, value is 'True' where the parity is odd
    """
    if rem_mask is None:
        rem_mask = ~comb_mask
    else:
        comb_mask, rem_mask = np.broadcast_arrays(comb_mask, rem_mask)
    odd_mask = np.logical_xor.accumulate(rem_mask, axis=-1)
    odd_mask[~comb_mask] = False
    odd_mask = np.logical_xor.reduce(odd_mask, axis=-1)

    if ASSERT_LEVEL > 3:
        # Combine permutation indices for the reference result
        combs = take_by_masks(np.arange(comb_mask.shape[-1]), comb_mask)
        rems = take_by_masks(np.arange(comb_mask.shape[-1]), rem_mask)
        perm = np.concatenate((combs, rems), axis=-1)
        # Use a reference result
        np.testing.assert_equal(odd_mask, _ref_perm_parity(perm), err_msg='Wrong optimized combinations parity')
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

def combinations_to_indices(comb_masks: np.array) -> np.array:
    """Convert combination mask to unique index"""
    comb_sizes = comb_masks.sum(-1) - 1
    indices = np.zeros(shape=comb_masks.shape[:-1], dtype=int)
    comb_vectorized = np.vectorize(np.math.comb, otypes=indices.dtype.char)
    for idx in range(comb_masks.shape[-1]):
        masks = comb_masks[...,idx]
        comb_sizes[masks] -= 1
        masks = ~masks & (comb_sizes >= 0)
        if masks.any():
            indices[masks] += comb_vectorized(comb_masks.shape[-1] - idx - 1, comb_sizes[masks])

    if ASSERT_LEVEL > 3:
        np.testing.assert_equal(combinations_from_indices(indices, comb_masks.shape[-1], comb_masks.sum(-1)), comb_masks,
            err_msg='Combinations to indices conversion is irreversible')
    return indices

def combinations_from_indices(indices: np.array, size: int, comb_size: int or np.array) -> np.array:
    """Convert combination unique index to mask"""
    indices, comb_size = np.broadcast_arrays(indices, comb_size - 1)
    indices = indices.copy()
    comb_size = comb_size.copy()
    comb_masks = np.zeros(shape=indices.shape + (size,), dtype=bool)
    comb_vectorized = np.vectorize(np.math.comb, otypes=comb_size.dtype.char)
    val_masks = comb_size >= 0      # Elements, that are being processed
    comb_size = comb_size[val_masks]
    indices = indices[val_masks]
    for idx in range(size):
        # Drop elements where 'comb_size' is exhausted
        masks = comb_size >= 0
        if not masks.any():
            break   # All elements are processed
        if not masks.all():
            val_masks[val_masks] = masks
            comb_size = comb_size[masks]
            indices = indices[masks]

        split_idx = comb_vectorized(size - idx - 1, comb_size)
        masks = indices < split_idx
        comb_masks[...,idx][val_masks] = masks
        comb_size[masks] -= 1
        masks = ~masks
        indices[masks] -= split_idx[masks]

    if ASSERT_LEVEL > 1:
        np.testing.assert_equal(comb_size, -1, err_msg='Unprocessed combinations left')
    return comb_masks

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

def det_minors_of_masks(det_of_masks: callable, comb_masks: np.array, row_base: int, *,
        num_rows: int, minors: np.array or None = None,
        max_det_size: int = MAX_DET_SIZE, det_sum: callable = det_sum_simple) -> np.array:
    """Minor determinants built on top of combinations of sub-matrices"""
    # Split each determinant into multiple minor-determinants (from sub-matrices):
    row_step = max_det_size
    while row_base < num_rows:
        if row_step >= num_rows - row_base:
            row_step = num_rows - row_base
        # Limit row step, to avoid too large intermediate arrays
        if ARR_THRESHOLD < comb_masks.size * np.math.comb(comb_masks.shape[-1] - row_base, row_step):
            row_step = 1
        # Optimization: simplified right-side masks selection for the final row(s)
        if row_base + row_step < comb_masks.shape[-1]:
            # Next combinations, based on the remainders from current ones
            masks = combinations_masks(comb_masks.shape[-1] - row_base, row_step)
            if minors is not None:
                masks = np.broadcast_to(masks[np.newaxis,...], minors.shape[-1:] + masks.shape)

            # Masks for the right-side elements:
            # the next combinations spread over the current remainders
            r_masks = ~comb_masks
            # Extra pre-last dimension for each right-side combination
            r_masks = np.stack((r_masks,) * masks.shape[-2], axis=-2)
            r_masks[r_masks] = masks.flat
            del masks
        else:
            # Final row(s), right-side masks are just the left-overs
            r_masks = ~comb_masks[...,np.newaxis,:]

        # Calculate the next minors
        r_minors = det_of_masks(r_masks, row_base,
                max_det_size=max_det_size, det_sum=det_sum)
        if minors is not None:
            minors = (minors[...,np.newaxis] * r_minors).reshape(minors.shape[:-1] + (-1,))
        else:
            minors = r_minors
        del r_minors

        # Calculate the next combination masks and parity
        comb_masks = comb_masks[...,np.newaxis,:]
        odd_masks = combinations_parity(comb_masks, r_masks).flatten()
        comb_masks = (comb_masks | r_masks).reshape(-1, comb_masks.shape[-1])
        del r_masks

        # Build remap-xform matrix to sum the duplicate combinations (identify overlaps)
        if not comb_masks.all():
            comb_idxs = combinations_to_indices(comb_masks)
            expected_dups = np.math.comb(row_base + row_step, row_step)
            remap_idxs = np.argsort(comb_idxs).reshape(-1, expected_dups)
            del comb_idxs
        else:
            # Final row(s), all combinations overlap
            remap_idxs = np.arange(comb_masks.shape[-2])

        # Drop the duplicated combimations - the remap places them in the pre-last dimension
        if ASSERT_LEVEL > 3:
            np.testing.assert_equal(comb_masks[remap_idxs[...,1:]], comb_masks[remap_idxs[...,:-1]],
                    err_msg='Unable to isolate "comb_masks" duplicates')
        comb_masks = comb_masks[remap_idxs[...,0]]

        # Apply determinant rule: sum the products by negating the odd-permutations
        minors = minors[...,remap_idxs]
        odd_masks = odd_masks[remap_idxs]
        minors = det_sum(minors, odd_masks)
        del remap_idxs, odd_masks

        row_base += row_step

    return minors, comb_masks

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

        # Apply determinant rule: sum the products by negating the odd-permutations
        return det_sum(res, odd_masks)

    def det_of_masks(masks, row_base, *, max_det_size, det_sum):
        """Wrapper of det_of_columns() to use combination masks instead of column indices"""
        return det_of_columns(take_data, take_by_masks(col_idxs, masks), row_base,
                max_det_size=max_det_size, det_sum=det_sum)
    # Build on top of empty initial (left-side) minors
    num_rows = col_idxs.shape[-1]
    minors, _ = det_minors_of_masks(det_of_masks, np.full(num_rows, False), row_base,
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
    minors, masks = det_minors_of_masks(det_of_masks, np.full(data.shape[-1], False), 0,
            num_rows=data.shape[-2], max_det_size=max_det_size)
    odd_masks = combinations_parity(masks)
    return np.negative(minors, out=minors, where=odd_masks)
