"""Test parity of permutations"""
import math
import numpy as np
import pytest

# Module to be tested
from numpy_determinant import determinant
determinant.ASSERT_LEVEL = 5    # Keep all internal tests on


def ref_perm_parity(perm_idxs: determinant.IndexArray) -> determinant.MaskArray:
    """Obtain the parity of permutations from the number of inversions

    Uses nested for-loops, as a reference
    See:
        https://statlect.com/matrix-algebra/sign-of-a-permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation
    """
    odd_mask = np.zeros(1, dtype=bool)
    # Plain nested for-loops implementation
    for i in range(perm_idxs.shape[-1] - 1):
        for j in range(i + 1, perm_idxs.shape[-1]):
            odd_mask = odd_mask ^ (perm_idxs[..., i] > perm_idxs[..., j])
    return odd_mask

@pytest.mark.parametrize('size', range(10))
def test_permutation_parity(size):
    """Test the vectorized permutation-parity implementation vs the slow reference one"""
    perm_idxs = determinant.permutations_indices(size)
    assert perm_idxs.shape[1] == size, 'Incorect permutation size'
    assert perm_idxs.shape[0] == math.factorial(size), 'Incorect number of permutations'

    odd_mask = determinant.permutation_parity(perm_idxs)
    np.testing.assert_equal(odd_mask, ref_perm_parity(perm_idxs),
                            err_msg='Wrong optimized permutation parity')

def combinations_masks(size, comb_size):
    """Helper to recombine "batched" data returned by determinant.combinations_masks()"""
    comb_mask = np.empty((0, size), dtype=bool)
    for masks in determinant.combinations_masks(size, comb_size,
                max_batch=determinant.MAX_COMBINATION_BATCH):
        comb_mask = np.append(comb_mask, masks, axis=0)
    return comb_mask

@pytest.mark.parametrize('size', range(1, 8))
def test_combinations_parity(size):
    """Test the vectorized combination-parity implementation vs the slow reference one"""
    for comb_size in range(1, size + 1):
        comb_mask = combinations_masks(size, comb_size)
        # Try remainder masks of various sizes (including full coverage)
        for rem_size in range(size - comb_size + 1):
            if rem_size:
                # Select some remainder-combination masks
                mask = combinations_masks(size - comb_size, rem_size)
                rem_mask = ~comb_mask
                rem_mask[rem_mask] = np.take(mask, np.arange(rem_mask.shape[0]),
                                             0, mode='wrap').flat
            else:
                # Test the "standard" behavior
                rem_mask = None
            odd_mask = determinant.combinations_parity(comb_mask, rem_mask)

            # Combine permutation indices for the reference result
            if rem_mask is None:
                rem_mask = ~comb_mask
            combs = determinant.take_by_masks(np.arange(comb_mask.shape[-1]), comb_mask)
            rems = determinant.take_by_masks(np.arange(rem_mask.shape[-1]), rem_mask)
            perm = np.concatenate((combs, rems), axis=-1)
            # Use a reference result
            np.testing.assert_equal(odd_mask, ref_perm_parity(perm),
                                    err_msg='Wrong optimized combinations parity')
