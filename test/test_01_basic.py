"""Test basic determinant calculations"""
import sys
import numpy as np
import pytest

 # Module to be tested
import determinant
determinant.ASSERT_LEVEL = 5    # Keep all internal tests on


# First 25 prime numbers as non-linearly dependent data
PRIME_NUMBERS = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
)

def test_det():
    """Test determinant calculation vs. numpy.linalg.det()"""
    print('\n* Compare to numpy.linalg.det() results')
    # linalg.det() returns exact results with first 9 prime-numbers only
    degree = 3
    data = np.array(PRIME_NUMBERS[:degree*degree]).reshape(degree, degree)
    ref_res = np.linalg.det(data)
    assert (ref_res != 0).all(), 'Must select non-linearly dependent data'
    res = determinant.det(data)
    assert res.dtype == data.dtype, 'Determiant data-type was changed'
    np.testing.assert_equal(res, ref_res, err_msg='Determinant value does not match the reference')

    # Interleave all 25 prime-numbers with ones to get 3 5x5 matrices
    # Do not expect exact results as "numpy.linalg.det()"
    degree = 5
    data = np.ones((3, degree, degree), dtype=int)
    data.flat[::3] = PRIME_NUMBERS[:degree*degree]
    ref_res = np.linalg.det(data)
    assert (ref_res != 0).all(), 'Must select non-linearly dependent data'
    res = determinant.det(data)
    np.testing.assert_allclose(res, ref_res, err_msg='Determinant value(s) does not match the reference')

def test_asserts():
    """Provoked failures"""
    print('\n* Provoke some failures')
    #determinant.ASSERT_LEVEL = 10

    # take_by_masks() expects uniform mask
    masks = np.array([[False, False], [True, False]])
    with pytest.raises(AssertionError):
        determinant.take_by_masks(np.zeros_like(masks), masks)

    # Complete determinant calculations cause float64 overflows
    def sample_dets(max_det_size=determinant.MAX_DET_SIZE):
        degree = 7
        data = np.arange(degree*degree, dtype=float) * 1e3
        res = determinant.det(data.reshape(degree, degree),
                max_det_size=max_det_size)
        return res
    # Some regular data test
    res = sample_dets()
    np.testing.assert_allclose(res, 0, err_msg='Non-zero determinant')
    # Same data calculated w/o sub-determinants
    res = sample_dets(10)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(res, 0, err_msg='Expected float overflow')

def test_dtypes():
    """Test determinant calculation from generic dtype-s"""
    print('\n* Use custom data-types')
    degree = 3
    data = np.arange(4*degree*degree).reshape(-1, degree, degree)
    res = determinant.det(data)
    np.testing.assert_equal(res, 0, err_msg='Non-zero determinant from linearly dependent vectors')

    # Increase all elements, except the top-left ones
    data[...] += 1
    data[..., 0,0] -= 1
    res = determinant.det(data)
    np.testing.assert_equal(res, 3, err_msg='Unexpected determinant value')

    #
    # Arbitrary data-type tests
    #
    src_data = data
    # Use fractions
    import fractions
    # The "data / 3." will cause loss of precision
    data = src_data * fractions.Fraction(1, 3)
    print(f'Source array type: {type(data.flat[0])}')
    res = determinant.det(data)
    assert isinstance(res.flat[0], fractions.Fraction), 'Determiant data-type was changed'
    res *= 3**data.shape[-1]    # Drop the 'denominator' (1/3**n)
    np.testing.assert_equal(res, 3., err_msg='Unexpected loss of precision')

    # Use decimal
    import decimal
    data = src_data * decimal.Decimal('.2')     # 1/5 should NOT lose precision
    print(f'Source array type: {type(data.flat[0])}')
    res = determinant.det(data)
    assert isinstance(res.flat[0], decimal.Decimal), 'Determiant data-type was changed'
    res *= 5**data.shape[-1]    # Drop the 'denominator' (1/5**n)
    np.testing.assert_equal(res, 3., err_msg='Unexpected loss of precision')

    # Use numpy Polynomial
    # use 'src_data' as 0-th coefficient for each polynomial
    data = np.apply_along_axis(np.polynomial.Polynomial, -1, src_data[..., np.newaxis])
    print(f'Source array type: {type(data.flat[0])}')
    res = determinant.det(data)
    assert isinstance(res.flat[0], np.polynomial.Polynomial), 'Determiant data-type was changed'
    np.testing.assert_equal(res, np.polynomial.Polynomial([3.]), err_msg='Unexpected polynomial result')

def test_det_minors():
    """Test minor determinant (determinant of sub matrix) calculation"""
    print('\n* Minor matrix determinants')
    degree = 3
    data = np.arange(4*degree*degree).reshape(-1, degree, degree)
    res = determinant.det_minors(data[...,:-1,:])
    np.testing.assert_equal(res.shape[-1], 3, err_msg='Unexpected number of minor determinants')
    assert (res != 0).all(), 'Zero determinant result from non-linearly dependent vectors'
    # Total determinant from the left and right minors
    res = (res * data[...,-1,:]).sum(-1)
    np.testing.assert_equal(res, 0, err_msg='Non-zero determinant from linearly dependent vectors')

def test_range_size():
    """Test derivatives from range of matrix sizes"""
    print('\n* Derivatives by increasing the matrix size')
    max_degree = 16
    # Combine some non-linearly dependent data (upto 10x10)
    # Keep values as small as possible, but still need 64-bit integers
    src_data = np.ones(shape=max_degree*max_degree, dtype=np.int64)
    v = 1
    for pn in PRIME_NUMBERS[1:6]:   # Five prime numbers after 2 (3-13)
        src_data[::pn] += v
        v = -v

    # Range the matrix sizes, start at empty one (0x0)
    for degree in range(max_degree + 1):
        data = src_data[:degree*degree].reshape(degree, degree)
        print(f'Matrix shape {data.shape}')
        # The numpy.linalg.det() result is inexact, but since it must be integer,
        # we can just round it to get the exact value
        ref_res = np.round(np.linalg.det(data))
        assert (ref_res != 0).all(), 'Must select non-linearly dependent data'
        res = determinant.det(data)
        np.testing.assert_equal(res, ref_res, err_msg='Determinant value does not match the reference')

#
# For non-pytest debugging
#
if __name__ == '__main__':
    res = test_det()
    if res:
        sys.exit(res)
    res = test_asserts()
    if res:
        sys.exit(res)
    res = test_dtypes()
    if res:
        sys.exit(res)
    res = test_det_minors()
    if res:
        sys.exit(res)
    res = test_range_size()
    if res:
        sys.exit(res)
