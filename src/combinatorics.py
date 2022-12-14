"""Combinatorics routines for matrix determinant calculation


The core is wrappers to combinations and permutations itertools.

See:
https://en.wikipedia.org/wiki/Combinatorics
https://en.wikipedia.org/wiki/Combination
https://en.wikipedia.org/wiki/Permutation
"""
import itertools
import numpy as np

#
# Permutations
#

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

class permutation:
    """Storage for set of permutations"""
    # Precalculated permutations to avoid 'itertools' for small sizes
    PRECALC_PERMS = list(np.array(vals, np.int8) for vals in (
            [[]],               # size=0
            [[0]],              # size=1
            [[0, 1], [1, 0]],   # size=2
        ))

    def __init__(self, size: int) -> np.array:
        """Create all permulations for specific number range"""
        if size < len(self.PRECALC_PERMS):
            self._perm_idxs = self.PRECALC_PERMS[size]
        else:
            # Use int8 as allocation of more than "factorial(128)" elements is impossible anyway
            self._perm_idxs = np.fromiter(itertools.permutations(range(size)),
                    dtype=(np.int8, [size]))

    @property
    def size(self):
        """Number of permutated elements"""
        return self._perm_idxs.shape[-1]

    @property
    def count(self):
        """Total number of permutations (1 when `size` is 0)"""
        # There is 1 empty permulation, when 'size' is 0 (!?)
        return self._perm_idxs[...,0].size if self.size else 1

    def to_indices(self) -> np.array:
        """Return the permutated indices"""
        return self._perm_idxs

    def permutation_parity(self) -> np.array:
        """Obtain the parity of permulations from the number of inversions

        Parameters
        ----------
        None

        Returns
        -------
        odd_mask : (...) array_like
            Array of booleans for each permulation, value is 'True' where the parity is odd
        """
        # Indices of '_perm_idxs' elements, to calculate inversions/parity at once:
        # the combinations of their number over 2
        idxs = np.fromiter(itertools.combinations(range(self._perm_idxs.shape[-1]), 2), dtype=(int, 2))

        # Regroup permutation indices in couples to check for inversions
        idxs = self._perm_idxs[...,idxs]
        odd_mask = np.logical_xor.reduce(idxs[...,0] > idxs[...,1], axis=-1)
        return odd_mask

#
# Combinations
#
class combination:
    """Abstract storage for set of combinations

    Combinations can be kept as boolen-mask, bit-mask or combination-index
    """

    def combinations_parity(self, rem_comb: object or None = None) -> np.array:
        """Obtain the parity of combinations from the number of inversions

        Parameters
        ----------
        rem_comb : (..., N) array_like, optional
            Array of combination remainders to compare with. The number of inversions is the swaps between two
            adjacent elements, in order to move all the `rem_comb` elements after `self` elements.
            When `None`, the inverted `self` is used instead, which counts the swaps to move all `self`
            elements in front.

        Returns
        -------
        odd_mask : (...) array_like
            Array of booleans for each combination, value is 'True' where the parity is odd
        """
        comb_mask = self.to_bool_mask()
        if rem_comb is None:
            rem_comb = ~comb_mask
        else:
            comb_mask, rem_comb = np.broadcast_arrays(comb_mask, rem_comb.to_bool_mask())
        odd_mask = np.logical_xor.accumulate(rem_comb, axis=-1)
        odd_mask[~comb_mask] = False
        odd_mask = np.logical_xor.reduce(odd_mask, axis=-1)
        return odd_mask

class combination_bools(combination):
    """Combinations storage for set as boolen-mask"""
    def __init__(self, size: int, comb_size: int):
        """Create all combinations for specific number range

        Parameters
        ----------
        size : int
            Number of elements to select from
        comb_size : int
            Number of elements to be selected
        """
        if comb_size == 0:
            self._bool_mask = np.zeros((1, size), dtype=bool)
        elif comb_size == 1:
            self._bool_mask = np.identity(size, dtype=bool)
        elif comb_size == size - 1:
            self._bool_mask = ~np.identity(size, dtype=bool)[::-1]
        else:
            idxs = np.fromiter(itertools.combinations(range(size), comb_size),
                    dtype=(np.int8, [comb_size]))
            self._bool_mask = np.zeros(shape=(idxs.shape[0], size), dtype=bool)
            self._bool_mask[np.arange(idxs.shape[0])[...,np.newaxis], idxs] = True

    @classmethod
    def from_bool_mask(cls, bool_mask: np.array) -> combination:
        """Encapsulate boolean-mask in 'combination_bools'"""
        comb = super().__new__(cls)
        comb._bool_mask = bool_mask
        return comb

    @classmethod
    def from_combination(cls, src: combination) -> combination:
        """Convert any combination to 'combination_bools' representation"""
        return cls.from_bool_mask(src.to_bool_mask())

    @property
    def size(self):
        """Number of combinated elements"""
        return self._bool_mask.shape[-1]

    @property
    def count(self):
        """Total number of combinations"""
        # There is 1 empty combination, when 'size' is 0 (!)
        return self._bool_mask[...,0].size if self.size else 1

    def comb_size(self, collapse=True):
        """Number of selected elements by each combination"""
        res = self._bool_mask.sum(-1)
        # Collapse _comb_size if they are all the same (anti-broadcast)
        if collapse and (res.flat[0] == res.flat).all():
            res = res.flat[0]
        return res

    def is_singular(self):
        """Check if all combinations are choose-all or choose-none"""
        return self._bool_mask.all() or not self._bool_mask.any()

    def to_bool_mask(self) -> np.array:
        """Convert the combinations to boolean-mask (already done)"""
        return self._bool_mask

    def to_bit_mask(self) -> np.array:
        """Convert the combinations to bit-mask"""
        assert self.size < 64, f'The number of bits {self.size}, is too high'
        dtype = np.uint16 if self.size < 16 else \
                np.uint32 if self.size < 32 else \
                np.uint64
        res = np.where(self._bool_mask, 1<<np.arange(self.size, dtype=dtype), 0)
        return res.sum(-1)

    def to_comb_index(self) -> np.array:
        """Convert the combinations to unique dense index"""
        comb_sizes = self._bool_mask.sum(-1) - 1
        indices = np.zeros(shape=self._bool_mask.shape[:-1], dtype=int)
        comb_vectorized = np.vectorize(np.math.comb, otypes=indices.dtype.char)
        for idx in range(self.size):
            masks = self._bool_mask[...,idx]
            comb_sizes[masks] -= 1
            masks = ~masks & (comb_sizes >= 0)
            if masks.any():
                indices[masks] += comb_vectorized(self.size - idx - 1, comb_sizes[masks])
        return indices

class combination_indices(combination):
    """Combinations storage for set as unique dense index"""
    def __init__(self, size: int, comb_size: int):
        self._size = size
        self._comb_size = comb_size
        # Initially, just range generator object, instead of complete array
        self._comb_indices = range(np.math.comb(self._size, self._comb_size))

    @classmethod
    def from_combination(cls, src: combination) -> combination:
        """Convert any combination to 'combination_indices' representation"""
        comb = super().__new__(cls)
        comb._comb_indices = src.to_comb_index()
        comb._size = src.size
        comb._comb_size = src.comb_size()
        return comb

    @property
    def size(self):
        """Number of elements being combined"""
        return self._size

    @property
    def count(self):
        """Total number of combinations"""
        # Note: '_comb_indices' can be a generator object
        return np.asarray(self._comb_indices).size

    def comb_size(self):
        """Number of selected elements by each combination"""
        return self._comb_size

    def to_bool_mask(self) -> np.array:
        """Convert the combinations to boolean-mask"""
        indices, comb_size = np.broadcast_arrays(self._comb_indices, self._comb_size - 1)
        comb_masks = np.zeros(shape=indices.shape + (self.size,), dtype=bool)
        comb_vectorized = np.vectorize(np.math.comb, otypes=comb_size.dtype.char)
        val_masks = comb_size >= 0      # Elements, that are being processed
        # This also makes a copy, thus '_comb_indices' and '_comb_size' will be preserved
        indices = indices[val_masks]
        comb_size = comb_size[val_masks]
        for idx in range(self.size):
            # Drop elements where 'comb_size' is exhausted
            masks = comb_size >= 0
            if not masks.any():
                break   # All elements are processed
            if not masks.all():
                val_masks[val_masks] = masks
                comb_size = comb_size[masks]
                indices = indices[masks]

            split_idx = comb_vectorized(self.size - idx - 1, comb_size)
            masks = indices < split_idx
            comb_masks[...,idx][val_masks] = masks
            comb_size[masks] -= 1
            masks = ~masks
            indices[masks] -= split_idx[masks]
        return comb_masks

    def to_bit_mask(self) -> np.array:
        """Convert the combinations to bit-mask"""
        #TODO: Replace this combination_bools() wrapper
        comb = combination_bools.from_combination(self)
        return comb.to_bit_mask()

    def to_comb_index(self) -> np.array:
        """Convert the combinations to unique dense index (already done)"""
        return np.asarray(self._comb_indices)

class combination_bits(combination):
    """Combinations storage for set as bit-mask"""
    def __init__(self, size: int, comb_size: int):
        """Create all combinations for specific number range"""
        #TODO: Replace this combination_bools() wrapper
        comb = combination_bools(size, comb_size)
        self._bit_mask = comb.to_bit_mask()
        self._size = comb.size
        self._comb_size = comb.comb_size()

    @classmethod
    def from_combination(cls, src: combination) -> combination:
        """Convert any combination to 'combination_bits' representation"""
        comb = super().__new__(cls)
        comb._bit_mask = src.to_bit_mask()
        comb._size = src.size
        comb._comb_size = src.comb_size()
        return comb

    @property
    def size(self):
        """Number of elements being combined"""
        return self._size

    @property
    def count(self):
        """Total number of combinations"""
        return self._bit_mask.size

    def comb_size(self):
        """Number of selected elements by each combination"""
        #TODO: Count bits in each '_bit_mask' element
        return self._comb_size

    def to_bool_mask(self) -> np.array:
        """Convert the combinations to boolean-mask"""
        masks = self._bit_mask[..., np.newaxis] & 1<<np.arange(self.size, dtype=self._bit_mask.dtype)
        return masks != 0

    def to_bit_mask(self) -> np.array:
        """Convert the combinations to bit-mask (already done)"""
        return self._bit_mask

    def to_comb_index(self) -> np.array:
        """Convert the combinations to unique dense index"""
        #TODO: Replace this combination_bools.to_comb_index() wrapper
        comb = combination_bools.from_combination(self)
        return comb.to_comb_index()
