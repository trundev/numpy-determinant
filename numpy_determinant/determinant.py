"""Pure Python/NumPy generic determinant calculator

Tools to calculate determinants, including of minor-matrix determinants.
The main goal is to work with NumPy array of any `dtype`, like:
  fractions, decimal, numbers, `numpy.polynomial.Polynomial`, etc.
The `determinant.det()` should work just like `numpy.linalg.det()`, with preserved `dtype`.
"""
import itertools
from typing import Callable, Optional
import numpy as np
import numpy.typing as npt


# Limit the size of a determinants to be calculated at once, in order to:
# - Avoid precision loss, due to huge intermediate values, leading to incorrect result,
#   like: "1e6**3 + 1 - 1e6**3 == 0"
# - Better performance
MAX_DET_SIZE = 3

ASSERT_LEVEL = 1

#
# Type annotation aliases
#
DataArray = npt.NDArray
# Use int8 as allocation of more than "factorial(128)" elements is impossible anyway
IndexType = np.uint8
IndexArray = npt.NDArray[IndexType]
MaskArray = npt.NDArray[np.bool_]

# Precalculated permutations to avoid `itertools` for small sizes
PRECALC_PERMS: list[IndexArray] = list(np.array(vals, IndexType) for vals in (
        [[]],               # size=0
        [[0]],              # size=1
        [[0, 1], [1, 0]],   # size=2
    ))

def permutation_parity(perm_idxs: IndexArray) -> MaskArray:
    """Obtain the parity of permutations from the number of inversions

    Parameters
    ----------
    perm_idxs : (..., N) array_like
        Array of permutations represented as indices, individual permutations are in the
        last dimension

    Returns
    -------
    odd_mask : (...) array_like
        Array of booleans for each permutation, value is 'True' where the parity is odd
    """
    # Indices of 'perm_idxs' elements, to calculate inversions/parity at once:
    # the combinations of their number over 2
    idxs = np.fromiter(itertools.combinations(range(perm_idxs.shape[-1]), 2), dtype=(int, 2))

    # Regroup permutation indices in couples to check for inversions
    idxs = perm_idxs[...,idxs]
    return np.logical_xor.reduce(idxs[...,0] > idxs[...,1], axis=-1)

def permutations_indices(size: int) -> IndexArray:
    """Obtain all permutations for specific number range"""
    if size < len(PRECALC_PERMS):
        return PRECALC_PERMS[size]
    return np.fromiter(itertools.permutations(range(size)),
            dtype=(IndexType, [size]))

def combinations_parity(comb_mask: MaskArray, rem_mask: MaskArray|None = None) -> MaskArray:
    """Obtain the parity of combinations from the number of inversions

    Parameters
    ----------
    comb_mask : (..., N) array_like
        Array of combinations represented as mask, individual combinations are in the last dimension
    rem_mask : (..., N) array_like, optional
        Array of combination remainders to compare with. The number of inversions is the swaps
        between two adjacent elements, in order to move all the `rem_mask` elements after
        `comb_mask`.
        When `None`, the inverted `comb_mask` is used instead, which counts the swaps to move all
        `comb_mask` elements in front.

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
    return np.logical_xor.reduce(odd_mask, axis=-1)

def combinations_masks(size: int, comb_size: int) -> MaskArray:
    """Obtain all combinations for specific number range

    Note: The returned mask can be inverted to get the remainder from combination"""
    if comb_size == 1:
        return np.identity(size, dtype=bool)
    if comb_size == size - 1:
        return ~np.identity(size, dtype=bool)[::-1]
    idxs = np.fromiter(itertools.combinations(range(size), comb_size),
            dtype=(IndexType, [comb_size]))
    masks = np.zeros(shape=(idxs.shape[0], size), dtype=bool)
    masks[np.arange(idxs.shape[0])[...,np.newaxis], idxs] = True
    return masks

def take_by_masks(data: DataArray, masks: MaskArray):
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

def take_data_matrix(data: DataArray, col_idxs: IndexArray, row_base: int) -> DataArray:
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

def take_data_toeplitz(data: DataArray, num_rows: int, num_dets: int|None,
        col_idx: IndexArray, row_base: int) -> DataArray:
    """Callback to convert data to Toeplitz matrix: https://en.wikipedia.org/wiki/Toeplitz_matrix

    Parameters
    ----------
    num_rows : int
        Number of rows in the topmost Toeplitz matrix
    num_dets : int or None
        Number of sequential matrices to be returned (for parallel processing)
    - Others: see take_data_matrix()
    """
    # Convert indices using the Toeplitz matrix data-order
    col_idx += num_rows - 1 - (np.arange(col_idx.shape[-1], dtype=IndexType) + row_base)
    if num_dets is not None:
        # Retrieve data for multiple determinants starting at each possible index
        col_idx = col_idx + np.arange(num_dets, dtype=IndexType
                                      ).reshape((-1,) + (1,) * col_idx.ndim)
    return data[..., col_idx]

def det_sum_simple(products: DataArray, odd_masks: MaskArray) -> DataArray:
    """Simple determinant sum implementation

    Sum the products by negating the odd-permutations
    """
    res = np.negative(products, out=products, where=odd_masks)
    return res.sum(-1)

def det_sum_split(products: DataArray, odd_masks: MaskArray) -> DataArray:
    """Split determinant sum implementation

    This should reduce the float precision loss in intermediate sum() results
    """
    even_res = products[..., ~odd_masks]
    odd_res = products[..., odd_masks]
    # First sum the common pairs of even and odd permutations
    comm_size = min(even_res.shape[-1], odd_res.shape[-1])
    res = (even_res[..., :comm_size] - odd_res[..., :comm_size]).sum(-1)
    # Then add the remainder from even or odd ones
    return res + even_res[..., comm_size:].sum(-1) - odd_res[..., comm_size:].sum(-1)

def det_minors_of_columns(take_data: Callable, col_idxs: npt.NDArray[np.integer], row_base: int, *,
        minor_size: int, left_only=False,
        max_det_size: int = MAX_DET_SIZE, det_sum: Callable = det_sum_simple
        ) -> tuple[DataArray, Optional[DataArray], MaskArray]:
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
    return *res, combinations_parity(masks)

def det_of_columns(take_data: Callable, col_idxs: npt.NDArray[np.integer], row_base: int, *,
        max_det_size: int = MAX_DET_SIZE, det_sum: Callable = det_sum_simple) -> DataArray:
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

def det(data: DataArray, *, max_det_size: int = MAX_DET_SIZE) -> DataArray:
    """Compute the determinant of an array - our version of `numpy.linalg.det()`

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
    return det_of_columns(take_data, np.arange(data.shape[-1], dtype=IndexType), 0,
            max_det_size=max_det_size)

def det_minors(data: DataArray, *, max_det_size: int = MAX_DET_SIZE) -> DataArray:
    """Compute the minor determinants of an array

    Parameters
    ----------
    data : (..., M, N) array_like
        Input array to compute minor determinants for (`M < N`).

    - Others: see `det()`
    """
    data = np.asarray(data)
    assert data.shape[-1] >= data.shape[-2], 'Unexpected shape of data matrix'

    def take_data(col_idxs, row_base):
        return take_data_matrix(data, col_idxs, row_base)
    res, _, odd_masks = det_minors_of_columns(take_data,
            np.arange(data.shape[-1], dtype=IndexType), 0,
            minor_size=data.shape[-2], left_only=True,
            max_det_size=max_det_size)
    return np.negative(res, out=res, where=odd_masks)
