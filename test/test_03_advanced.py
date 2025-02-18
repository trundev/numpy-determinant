"""Test advanced determinant calculation options"""
import numpy as np

 # Module to be tested
from numpy_determinant import determinant
determinant.ASSERT_LEVEL = 5    # Keep all internal tests on


# Matrix of known determinant value of 3
MATRIX_DET_3 = np.array([
    [0, 2, 3],
    [4, 5, 6],
    [7, 8, 9] ])

def test_custom_dtype():
    """Test user defined data type"""
    print('\n* Custom data-type')

    #
    # Data-type object with minimal set of operations
    #
    mul_cnt = 0
    neg_cnt = 0
    add_cnt = 0
    class MyDtype:
        """Custom `dtype` supporting minimal operations only"""
        def __init__(self, val):
            self.val = val
        def __mul__(self, obj):
            nonlocal mul_cnt
            mul_cnt += 1
            return MyDtype(self.val * obj.val)
        def __neg__(self):
            nonlocal neg_cnt
            neg_cnt += 1
            return MyDtype(-self.val)
        def __add__(self, obj):
            nonlocal add_cnt
            add_cnt += 1
            return MyDtype(self.val + obj.val)
    print(f'Data-type object operations: {set(dir(MyDtype)) - set(dir(object))}')

    # Create array of `MyDtype` objects and calculate its determinant
    data = np.vectorize(MyDtype)(MATRIX_DET_3)
    res = determinant.det(data)
    # Validate the result
    assert isinstance(res, MyDtype), 'Determiant data-type was changed'
    np.testing.assert_equal(res.val, 3, err_msg='Unexpected determinat result')

    print(f'Matrix shape: {data.shape}, number of operations:')
    print(f'  mul: {mul_cnt}, neg: {neg_cnt}, add: {add_cnt}')

    #
    # Data-type object with only `mul` operation, need custom det_sum()
    #
    mul_cnt = 0
    del neg_cnt
    del add_cnt
    class MyMulDtype:
        """Custom `dtype` supporting `mul` operations only"""
        def __init__(self, val):
            self.val = val
        def __mul__(self, obj):
            nonlocal mul_cnt
            mul_cnt += 1
            return MyMulDtype(self.val * obj.val)
    print(f'Data-type object operations: {set(dir(MyMulDtype)) - set(dir(object))}')
    def to_my_dtype(arr):
        return MyMulDtype(arr) if np.isscalar(arr) else np.vectorize(MyMulDtype)(arr)

    # Custom det_sum() function
    det_sum_cnt = 0
    det_sum_vals = 0
    def my_det_sum(products, odd_masks):
        """Extract values, run original det_sum(), then combine them back in `MyMulDtype`"""
        nonlocal det_sum_cnt, det_sum_vals
        det_sum_cnt += 1
        det_sum_vals += products.size
        vals = np.vectorize(lambda o: o.val)(products)
        res = determinant.det_sum_simple(vals, odd_masks)
        return to_my_dtype(res)

    # Create array of `MyMulDtype` objects and calculate its determinant
    data: determinant.DataArray = to_my_dtype(MATRIX_DET_3)
    # This is `determinant.det()`, but using a custom 'det_sum' callback
    def take_data(*args):
        return determinant.take_data_matrix(data, *args)
    res = determinant.det_of_columns(take_data, np.arange(data.shape[-1]), 0,
            det_sum=my_det_sum)
    # Validate the result
    assert isinstance(res, MyMulDtype), 'Determiant data-type was changed'
    np.testing.assert_equal(res.val, 3, err_msg='Unexpected determinat result')

    print(f'Matrix shape: {data.shape}, number of operations:')
    print(f'  mul: {mul_cnt} det_sum() calls: {det_sum_cnt} for {det_sum_vals} vals')

def test_det_of_columns():
    """Test feed data by array of column indices and custom take_data()"""
    print('\n* Feed data for multiple matrices')
    # Prepare:
    # Nine overlapping 4x4 matrices with non-zero determinants
    degree = 4
    src_data = np.arange(9*degree).reshape(degree, -1)
    src_data.flat[::5] = 1
    # Re-arrange data in nine matrices
    src_idxs = np.arange(degree)
    src_idxs = np.arange(src_data.shape[-1] - degree + 1)[:, np.newaxis] + src_idxs
    data = np.moveaxis(src_data[...,src_idxs], 0, 1)
    # Get reference result, ensure it is exact by rounding
    ref_res = np.round(np.linalg.det(data))

    # Option 1:
    # Call det_of_columns() with multiple matrices
    print(f'Request multiple matrices: shape {src_idxs.shape}')
    take_data_cnt = 0
    take_data_vals = 0
    def take_data(col_idxs, *args):
        """Take single element for each 'col_idxs'"""
        nonlocal take_data_cnt, take_data_vals
        take_data_cnt += 1
        take_data_vals += col_idxs.size
        return determinant.take_data_matrix(src_data, col_idxs, *args)
    res = determinant.det_of_columns(take_data, src_idxs, 0)
    np.testing.assert_equal(res, ref_res, err_msg='Determinant value does not the match reference')
    print(f'  number of take_data() calls: {take_data_cnt} for {take_data_vals} vals')

    # Option 2:
    # Call det_of_columns() with single matrix, but feed multiple via take_data()
    num_matrices = src_idxs.shape[0]
    src_idxs = src_idxs[0]
    print(f'Request single matrix: shape {src_idxs.shape}')
    take_data_cnt = 0
    take_data_vals = 0
    def take_data_2(col_idxs, *args):
        """Take multiple elements for each 'col_idxs'"""
        nonlocal take_data_cnt, take_data_vals
        take_data_cnt += 1
        # Expand with extra dimension for multiple matrices
        col_idxs = np.arange(num_matrices).reshape((-1,) + (1,)*col_idxs.ndim) + col_idxs
        take_data_vals += col_idxs.size
        return determinant.take_data_matrix(src_data, col_idxs, *args)
    res = determinant.det_of_columns(take_data_2, src_idxs, 0)
    np.testing.assert_equal(res, ref_res, err_msg='Determinant value does not the match reference')
    print(f'  number of take_data_2() calls: {take_data_cnt} for {take_data_vals} vals')

def test_polynomials():
    """Extensive test using `numpy.polynomial.Polynomial` as data-type"""
    print('\n* Array of numpy.polynomial.Polynomial-s')
    # Select coefficients for matrix of polynomials
    # Reference data as 1-st coefficient for each polynomial, 0-th is one: "1 + <n>*x"
    coefs = np.stack(np.broadcast_arrays(1, MATRIX_DET_3), axis=-1)
    data = np.apply_along_axis(np.polynomial.Polynomial, -1, coefs)
    print(f'Single 3x3 matrix (data-type {type(data.flat[0])}):')
    print(np.vectorize(str)(data))
    res = determinant.det(data)
    assert isinstance(res, np.polynomial.Polynomial), 'Determiant data-type was changed'
    np.testing.assert_equal(res, np.polynomial.Polynomial([0,0,0,3]),
                            err_msg='Unexpected polynomial result')

    # 21 4x4 matrices with non-zero determinants
    degree = 4
    data = np.arange(3*7*degree*degree).reshape(3, 7, degree, degree)
    data.flat[::5] = 1
    print(f'Mutiple {data.shape[:-2]}, 4x4 matrices (data-type {type(data.flat[0])})')
    # Get reference result, ensure it is exact by rounding
    ref_res = np.round(np.linalg.det(data))
    assert (ref_res != 0).all(), 'Must select non-linearly dependent data'
    # The expected polynomials will be of 4-th degree
    coefs = np.stack(np.broadcast_arrays( *(0,)*degree, ref_res), axis=-1)
    ref_res = np.apply_along_axis(np.polynomial.Polynomial, -1, coefs)

    # Select coefficients for matrix of 1-st degree polynomials
    coefs = np.stack(np.broadcast_arrays(0, data), axis=-1)
    data = np.apply_along_axis(np.polynomial.Polynomial, -1, coefs)
    res = determinant.det(data)
    assert isinstance(res.flat[0], np.polynomial.Polynomial), 'Determiant data-type was changed'
    np.testing.assert_equal(res, ref_res, err_msg='Unexpected polynomial result')
