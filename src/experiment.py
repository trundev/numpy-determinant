"""Alternative determinant calculation approaches"""
import numpy as np
import determinant

from test_01_basic import PRIME_NUMBERS

#
# Arrays of bit-mask values to replace some of boolean arrays
#
def mask_to_bits(mask: np.array) -> np.array:
    """The last 'mask' (boolean array) dimension will be converted to bit-mask"""
    assert mask.shape[-1] < 64, f'The number of bits {mask.shape[-1]}, is too high'
    dtype = np.uint16 if mask.shape[-1] < 16 else \
            np.uint32 if mask.shape[-1] < 32 else \
            np.uint64
    res = np.where(mask, 1<<np.arange(mask.shape[-1], dtype=dtype), 0)
    return res.sum(-1)

def bits_to_mask(bits: np.array, width: int) -> np.array:
    """The bits are converted to boolean array with extra last-dimension, corresponding to each bit"""
    res = bits[..., np.newaxis] & 1<<np.arange(width, dtype=bits.dtype)
    return res != 0

def row_crawl(data, num_rows=None, row_step=1):
    #NOTE: 'row_step' higher than 1 not yet supported
    if num_rows is None:
        num_rows = data.shape[-2]
    if row_step > num_rows:
        row_step = num_rows
    print(f'* Row crawling method: {data.shape=}, {num_rows=}')
    def minors_of_masks(comb_masks, row_base):
        if (comb_masks.sum(-1) == 1).all():
            # Single element per row - this is the data itself, instead of minor-determinant
            res = determinant.take_by_masks(data[...,row_base,:], comb_masks)
            return res.reshape(res.shape[:-1])      # The last dimension is 1
        # Form single row, this must be (except the parity):
        # return determinant.det_minors(src_data[:1])
        idxs = determinant.take_by_masks(np.arange(data.shape[-1]), comb_masks)
        def take_data(col_idxs, row_base):
            return determinant.take_data_matrix(data, col_idxs, row_base)
        return determinant.det_of_columns(take_data, idxs, row_base)

    # Get the initial minors to build on top of - left-side minors
    comb_masks = np.zeros(data.shape[-1], dtype=bool)
    minors = np.empty(shape=0)

    for row in range(0, num_rows, row_step):
        # Keep 'row_step' in boundaries
        if row_step > num_rows - row:
            row_step = num_rows - row
        print(f'{row=}, {row_step=}:')
        print(f'  {minors.shape=}')

        np.testing.assert_equal(comb_masks.sum(-1), row,
                err_msg='Combined mask does not match the row-number')
        # New combinations, based on the remainders from current ones
        masks = determinant.combinations_masks(data.shape[-1] - row, row_step)
        if minors.size:
            masks = np.broadcast_to(masks[np.newaxis,...], minors.shape[-1:] + masks.shape)

        # Masks for the right-side elements:
        # the new combinations spread over the current remainders
        r_masks = ~comb_masks
        # Extra pre-last dimension for each right-side combination
        r_masks = np.stack((r_masks,) * masks.shape[-2], axis=-2)
        r_masks[r_masks] = masks.flat
        np.testing.assert_equal(r_masks.sum(-1), row_step,
                err_msg='Combined mask does not match next minors')

        # Calculate the new minors
        r_minors = minors_of_masks(r_masks, row)
        if minors.size:
            minors = (minors[...,np.newaxis] * r_minors).reshape(*minors.shape[:-1], -1)
        else:
            minors = r_minors

        # Calculate the new combination masks and parity
        comb_masks = comb_masks[...,np.newaxis,:]
        odd_masks = determinant.combinations_parity(comb_masks, r_masks).flatten()
        comb_masks = (comb_masks | r_masks).reshape(-1, comb_masks.shape[-1])
        np.testing.assert_equal(comb_masks.sum(-1), row + row_step,
                err_msg='Combined mask does not match combined minors')
        print(f'  After cross-multiplication: {minors.shape=} - [expected increase by x{np.math.comb(data.shape[-1] - row, row_step)}')

        # Get the new combinations and where the current ones are duplicated (identify overlaps)
        new_comb_masks = determinant.combinations_masks(data.shape[-1], row + row_step)
        masks = (comb_masks == new_comb_masks[...,np.newaxis,:]).all(-1)
        expected_dups = np.math.comb(row + row_step, row_step)
        np.testing.assert_equal(masks.sum(-1), expected_dups,
                err_msg='Unexpected number of repeated combinations')
        np.testing.assert_equal(masks.sum(-2), 1,
                err_msg='Duplicate matches, "new_comb_masks" elements may not be unique')
        comb_masks = new_comb_masks

        # Sum the minors along the columns where the current combinations overlap
        minors = determinant.take_by_masks(minors, masks)
        odd_masks = determinant.take_by_masks(odd_masks, masks)
        minors = determinant.det_sum_simple(minors, odd_masks)

        print(f'  After reduce-summation: {minors.shape=} - [expected decrease by /{expected_dups}]')
        print(f'     {minors}')

    print(f'Finally: {minors.shape=}')
    return minors

####
def experiment(src_data, minor_size):
    total_size = src_data.shape[-1]
    print(f'{total_size=}, {minor_size=}')

    ### Reference
    res = determinant.det_minors(src_data)

    min_r = determinant.det_minors(src_data[:2,...])
    min_l = determinant.det_minors(src_data[2:5,...])

    masks = determinant.combinations_masks(total_size, minor_size)

    ### New approach
    if True:
        print(f'\n# Use masks')
        m = np.logical_and(masks[:, np.newaxis], masks)
        m = ~np.logical_or.reduce(m, axis=-1)
        print(f'{m.shape=}, non-zeros {np.count_nonzero(m)}')
        print('  non-zeros:', ','.join(f'{np.count_nonzero(m, axis=i)}' for i in range(m.ndim)))
        np.testing.assert_equal(m, m.T)

        prods = min_r[:, np.newaxis] * min_l
        prods = prods[m]

        m2 = masks[:, np.newaxis] | masks

    if True:
        print(f'\n# Use bits')
        bits = mask_to_bits(masks)
        bits_and = np.bitwise_and(bits[:, np.newaxis], bits)
        m = bits_and == 0

        masks2 = determinant.combinations_masks(total_size, 2*minor_size)
        bits2 = mask_to_bits(masks2)

        bits_or = np.bitwise_or(bits[:, np.newaxis], bits)
        m = bits_or[...,np.newaxis] == bits2
        print(f'{m.shape=}, non-zeros {np.count_nonzero(m)}')
        print(f'  non-zeros in last dim {np.count_nonzero(m, axis=-1)}')
        assert (np.count_nonzero(m, axis=-1) <= 1).all(), 'Last dimention must have single or none matches'
        assert np.count_nonzero(m.any(-1)) == np.count_nonzero(m), 'Same as above'

        # Extract combination indices
        idxs = np.broadcast_to(np.arange(m.shape[-1]).reshape(1,1,-1), shape=m.shape)[m]
        print(f'{idxs.shape=}')
        assert np.count_nonzero(m.any(-1)) == idxs.size, 'Combination indices do not match actual combinations'
        idxs_shaped = np.full(m.shape[:-1], -1, dtype=int)
        idxs_shaped[m.any(-1)] = idxs
        print(f'idxs_shaped: {idxs_shaped}')
        print(f'  valid indices: {np.count_nonzero(idxs_shaped >= 0)}')
        hist = (idxs[:,np.newaxis] == np.arange(idxs.max() + 1)).sum(0)
        print(f'  hist: {hist}')
        assert (hist[0] == hist[...]).all(), 'Combinatios/permutations not evenly distributed'
        assert hist.size == bits2.size, 'CHECKME'

        #prods = min_r[:, np.newaxis] * min_l



#
# Main
#
if __name__ == '__main__':
    # Test bit-masks
    mask = determinant.combinations_masks(5, 2)
    bits = mask_to_bits(mask)
    print('\n'.join(f'  {m}:  {b:08X}' for m,b in zip(mask, bits)))
    np.testing.assert_array_equal(bits_to_mask(bits, mask.shape[-1]), mask), 'Convert-back failed'
    del mask, bits

    total_size = 5
    minor_size = 2

    #src_data = MATRIX_DET(total_size)[:-1,...]
    src_data = np.arange(total_size*2*minor_size).reshape(-1, total_size)
    src_data[minor_size:] += 100
    #### Simple trackable data
    if True:
        total_size = 3
#        src_data = np.array([
#                np.arange(total_size),
#                (10,) * total_size,
#                (100,) * total_size,
#            ])
        src_data = np.asarray(PRIME_NUMBERS[:total_size**2]).reshape(total_size, total_size)
        minor_size = 1
    ####

    print(f'{src_data.shape=} {minor_size=}')
    res = row_crawl(src_data, src_data.shape[-2])
    print(res)

    res = experiment(src_data, minor_size)
    print(res)
