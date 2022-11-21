# Matrix Determinant Calculator

Python module to calculate matrix and minor-matrix determinants from [NumPy](https://numpy.org/) arrays.

## Main goal

The module is intended to allow as-generic-as-posible determinant calculation, that does not rely on specific data-type.
NumPy arrays of any [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html) can be used as the source matrix.

The source `ndarray.dtype` objects should only support product (`*`) operation, [`sum()`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) and [`negative()`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html) are optional.
This includes generic Python objects, like.
- [Fractions](https://docs.python.org/3/library/fractions.html)
- [Decimals](https://docs.python.org/3/library/decimal.html)
- [Numbers](https://docs.python.org/3/library/numbers.html) - numeric abstract classes
- [NumPy polynomials](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial)


## Installation

The module can be installed from the GitHub repo:

> pip install "git+https://github.com/trundev/numpy-determinant.git@main"


## Usage

The main `numpy_determinant.det()` can be used just like the `numpy.linalg.det()`:

```python
>>> import numpy as np
>>> import numpy_determinant

>>> a = np.arange(9).reshape(3, 3)  # Some linearly dependent data
>>> a += a != 0         # Increase all elements, but the first one
>>> a
array([[0, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> numpy_determinant.det(a)
3
```

See [test](./test) folder for more examples
