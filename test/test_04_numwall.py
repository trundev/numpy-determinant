"""Test determinant calculation by using Toeplitz matrices

Validate determinant calculation, by decomposing exponential-polynomial functions,
using number-wall matrices.

References:
- YouTube video by Mathologer: https://www.youtube.com/watch?v=NO1_-qptr6c
- Exponential polynomials: https://en.wikipedia.org/wiki/Exponential_polynomial
"""
from typing import Iterable
import numpy as np
import numpy.typing as npt

 # Module to be tested
from numpy_determinant import determinant
determinant.ASSERT_LEVEL = 5    # Keep all internal tests on

ABS_TOLERANCE = 1e-08

def test_combinations():
    """Test if the order of det_minors_of_columns() 'minors' result, is as expected
    by numwall_decomposition()"""
    print('\n* Order of minor-matrix combinations, when size is by one less-than-total')
    for sz in range(3, 10):
        # The 'minors' are ordered by the combinations_masks() result
        masks = determinant.combinations_masks(sz, sz-1)
        # The first combination must include all but the last elements,
        # in each next, the skipped element must crawl-back by one
        for idx, m in enumerate(masks[::-1]):
            m[idx] ^= True
            assert m.all(), 'Unexpected minor-matrix combinations order'

def exp_poly_repr(poly_coefs: Iterable, poly_powers: Iterable, exp_bases: Iterable, *,
                  mul_sym='*', pwr_sym='^', var='t'):
    """Representation of exponential polynomial"""
    coef_fmt = '{}'
    poly_fmt = f'{var}{pwr_sym}{{}}'
    exp_fmt = f'({{}}){pwr_sym}{var}'
    components = []
    for a, p, b in zip(poly_coefs, poly_powers, exp_bases):
        if not a:
            continue
        sub_comp = (s for s, cond in ( (coef_fmt.format(a), a!=1),
                                       (poly_fmt.format(p), p!=0),
                                       (exp_fmt.format(b), b!=1) ) if cond)
        components.append(mul_sym.join(sub_comp))
    return ' + '.join(components)

def numwall_decomposition(data: npt.NDArray, degree: int, round_roots: int|None=None):
    """Decomposition of exponential polynomial function"""
    # Note: 'use_left_only' needs one less 'num_rows', to provide one more 'num_dets'
    num_dets = data.shape[-1] - 2*degree + 1
    def take_data(col_idx, row_base):
        return determinant.take_data_toeplitz(data, degree, num_dets, col_idx, row_base)
    minors, _, odd_masks = determinant.det_minors_of_columns(take_data, np.arange(degree + 1), 0,
            minor_size=degree, left_only=True)

    # Find valid solutions
    minors = minors[...,::-1]   # Reorder 'minors' order to match 'data'
    vals_arr = data[..., np.arange(minors.shape[-2])[:, np.newaxis] + np.arange(2*degree)]
    res = determinant.det_sum_simple(minors * vals_arr[...,:minors.shape[-1]], odd_masks)
    # Only where the determinant is zero (the solution is valid)
    zero_mask = np.isclose(res, 0)
    if not zero_mask.any():
        print('No zero determinants found')
        return None

    # Mask-out invalid solutions
    minors = minors[zero_mask]
    vals_arr = vals_arr[zero_mask]
    # Apply sub-matrix parity
    np.negative(minors, out=minors, where=odd_masks)
    # Obtain the values of 't'
    t_total = np.arange(res.shape[-1])[zero_mask]

    # Recheck minors vs. right-side boundary data (odd_masks are already applied)
    det_sum = determinant.det_sum_simple(minors * vals_arr[...,-minors.shape[-1]:],
                                         np.asarray(False))
    np.testing.assert_allclose(det_sum, 0, atol=ABS_TOLERANCE)
    if not minors.any():
        print('No non-zero minor-determinants found')
        return None

    # The left-side minors are characteristic polynomials
    minors = np.apply_along_axis(np.polynomial.Polynomial, -1, minors)
    roots_arr: npt.NDArray = np.frompyfunc(np.polynomial.Polynomial.roots, 1, 1)(minors)
    for roots, poly, vals, t in zip(roots_arr, minors, vals_arr, t_total):
        print(f'- Characteristic polynomial at {t}: {poly}')
        # Check if any roots were found
        if roots.size:
            #FIXME: The polynomial roots usually deviate
            # However, it is important to identify the duplicate ones
            if round_roots is not None:
                roots = np.real_if_close(np.round(roots, round_roots))
            print(f'    Exponent bases: {roots}')
            if np.iscomplex(roots).any():
                print(f'        period: {2*np.pi/np.angle(roots)} samples')
            # Select powers of the polynomial:
            # each duplicated root increases the power (polynomial degree)
            t_powers = np.zeros_like(roots, dtype=int)
            for idx, fl in enumerate(np.isclose(roots[1:], roots[:-1])):
                np.putmask(t_powers[idx + 1,...], fl, t_powers[idx,...] + 1)
            # Range of the main function argument (t)
            t_range = np.arange(roots.shape[-1], dtype=roots.dtype)[...,np.newaxis] + t
            # Spread polynomial: t power individual polynomial degrees
            system_matrix = t_range ** t_powers
            # Spread exponent: root on power t
            system_matrix *= roots ** t_range
            # Solve system of equations
            inv = np.linalg.inv(system_matrix)
            vals = vals[:roots.size]
            a_coefs = inv.dot(vals)
            print(f'    Polynomial coefficients: {a_coefs}')
            print(f'    Decomposed function: {exp_poly_repr(a_coefs, t_powers, roots)}')

            # Calculate-back the polynomial and compare with data
            t_range = np.arange(2*degree)[...,np.newaxis] + t
            total = (a_coefs * t_range ** t_powers * roots ** t_range).sum(-1)
            ref_data = data[t:t + len(total)]
            print(f'        Results: {total}')
            print(f'        Expected: {ref_data}')
            #FIXME: Hand-picked absolute tolerance
            np.testing.assert_allclose(total, ref_data, atol=ABS_TOLERANCE)
    return True

def test_degree2():
    """Decompose second degree exponential-polynomials"""
    degree = 2
    print(f'\n* Exponential-polynomials of {degree} degree ')
    t = np.arange(2*degree)     # Need 2 samples for each degree

    data = (1) * 2**t + (1) * 3**t
    print(f'Sum of two exponents: {data}')
    #FIXME: Hand-picked round_roots
    res = numwall_decomposition(data, degree, round_roots=14)
    print()

    data = (1) * 2**t + (3) * 1**t
    print(f'Exponent plus constant (0-deg polynomial): {data}')
    res = numwall_decomposition(data, degree)
    print()

    data = (3*t + 5) * 2**t
    print(f'Exponent by 1-deg polynomial: {data}')
    #FIXME: Hand-picked round_roots
    res = numwall_decomposition(data, degree, round_roots=14)
    print()

def test_degree3():
    """Decompose third degree exponential-polynomials"""
    degree = 3
    print(f'\n* Exponential-polynomials of {degree} degree ')
    t = np.arange(2 + 2*degree)     # Need 2 samples for each degree, +2 for total 3 results

    data = (1) * 2**t + (1) * 3**t + (1) * 5**t
    print(f'Sum of three exponents: {data}')
    #FIXME: Hand-picked round_roots
    res = numwall_decomposition(data, degree, round_roots=14)
    print()

    data = (1) * 2**t + (3*t + 5) * 1**t
    print(f'Exponent plus exponent by 1-deg polynomial: {data}')
    #FIXME: Hand-picked round_roots
    res = numwall_decomposition(data, degree, round_roots=6)
    print()

    data = (3*t**2 + 1) * 2**t
    print(f'Exponent by 2-deg polynomial: {data}')
    #FIXME: Hand-picked round_roots
    res = numwall_decomposition(data, degree, round_roots=4)
    print()

def test_sine():
    """Decompose 'sine' function (second degree complex exponential-polynomial)"""
    degree = 2
    print(f'\n* Sine: complex exponential-polynomials of {degree} degree ')
    samples = 48000 / 440   # 440Hz at 48kHz sample-rate
    data = np.sin(np.arange(1 + 2*degree) * 2 * np.pi / samples)
    data *= 2000    # Scale values
    print(f'Sine wave: {data}')
    res = numwall_decomposition(data, degree)
    print()
