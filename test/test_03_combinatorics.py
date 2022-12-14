"""Test combinatorics functionality"""
import sys
import numpy as np
import pytest

 # Module to be tested
import combinatorics


def test_permutations():
    """Permutations tests"""
    print('\n* Permutation generation')
    for size in range(10):
        perms = combinatorics.permutation(size)
        assert perms.count == np.math.factorial(size), 'Unexpected permutation size'
        unique, counts = np.unique(perms.to_indices(), return_counts=True)
        np.testing.assert_equal(unique, np.arange(size), err_msg='Permutations unique values do not match')
        np.testing.assert_equal(counts, perms.count, err_msg='Unevenly distributed permutation values')

def test_permutations_parity():
    """Permutations parity tests"""
    print('\n* Parity of permutations')
    for size in range(10):
        perms = combinatorics.permutation(size)
        parity = perms.permutation_parity()
        ref_parity = combinatorics._ref_perm_parity(perms.to_indices())
        np.testing.assert_equal(parity, ref_parity, 'Unexpected permutations parity')

def test_combinations():
    """Combinations tests"""
    print('\n* Combination generation')
    for size in range(20):
        for comb_size in range(size + 1):
            ref_count = np.math.comb(size, comb_size)

            # combination_bools class
            combs_bool = combinatorics.combination_bools(size, comb_size)
            np.testing.assert_equal(combs_bool.size, size, err_msg='Unexpected size of combination')
            np.testing.assert_equal(combs_bool.count, ref_count, err_msg='Unexpected combinations count')

            # combination_indices class
            combs_ind = combinatorics.combination_indices(size, comb_size)
            np.testing.assert_equal(combs_ind.size, size, err_msg='Unexpected size of combination')
            np.testing.assert_equal(combs_ind.count, ref_count, err_msg='Unexpected combinations count')
            np.testing.assert_equal(combs_ind.to_comb_index(), np.arange(combs_ind.count),
                    err_msg='Unexpected combinations as index')

            # combination_bits class
            combs_bits = combinatorics.combination_bits(size, comb_size)
            np.testing.assert_equal(combs_bits.size, size, err_msg='Unexpected size of combination')
            np.testing.assert_equal(combs_bits.count, ref_count, err_msg='Unexpected combinations count')

            # Test values via cross conversions
            # 'bools' via 'index' representations
            combs = combinatorics.combination_indices.from_combination(combs_bool)
            np.testing.assert_equal(combs.to_bool_mask(), combs_bool.to_bool_mask(),
                    err_msg='Mismatch after boolean-mask to index conversion')

            # 'index' via 'bits' representations
            combs = combinatorics.combination_bits.from_combination(combs_ind)
            np.testing.assert_equal(combs.to_comb_index(), combs_ind.to_comb_index(),
                    err_msg='Mismatch after index to bit-mask conversion')

            # 'bits' via 'bools'  representations
            combs = combinatorics.combination_bools.from_combination(combs_bits)
            np.testing.assert_equal(combs.to_bit_mask(), combs_bits.to_bit_mask(),
                    err_msg='Mismatch after bit-mask to boolean-mask conversion')

def test_combinations_parity():
    """Combinations parity tests"""
    print('\n* Parity of combinations')
    from determinant import take_by_masks
    for size in range(20):
        for comb_size in range(size + 1):
            combs = combinatorics.combination_bools(size, comb_size)
            parity = combs.combinations_parity()

            # Combine permutation indices for the reference result
            comb_mask = combs.to_bool_mask()
            rem_mask = ~comb_mask
            comb_idxs = take_by_masks(np.arange(size), comb_mask)
            rem_idxs = take_by_masks(np.arange(size), rem_mask)
            perms = np.concatenate((comb_idxs, rem_idxs), axis=-1)
            ref_parity = combinatorics._ref_perm_parity(perms)

            np.testing.assert_equal(parity, ref_parity, 'Unexpected combinations parity')


#
# For non-pytest debugging
#
if __name__ == '__main__':
    res = test_permutations()
    if res:
        sys.exit(res)
    res = test_permutations_parity()
    if res:
        sys.exit(res)
    res = test_combinations()
    if res:
        sys.exit(res)
    res = test_combinations_parity()
    if res:
        sys.exit(res)
