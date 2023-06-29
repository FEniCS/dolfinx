# Copyright (C) 2017-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra functionality"""


import numpy.typing as npt
import numpy as np
from dolfinx.cpp.common import IndexMap
from dolfinx.cpp.la import BlockMode, InsertMode, Norm
from dolfinx.cpp.la.petsc import create_vector as create_petsc_vector

from dolfinx import cpp as _cpp

__all__ = ["orthonormalize", "is_orthonormal", "create_petsc_vector", "matrix_csr", "vector",
           "MatrixCSR", "Norm", "InsertMode", "Vector", ]


class MatrixCSR:
    def __init__(self, A):
        """A distributed sparse matrix that uses compressed sparse row storage.

        Args:
            sp: The sparsity pattern that defines the nonzero structure
            of the matrix the parallel distribution of the matrix
            bm: The block mode (compact or expanded). Relevant only if
            block size is greater than one.

        Note:
            Objects of this type should be created using
            :func:`matrix_csr` and not created using the class
            initialiser.

        """
        self._cpp_object = A

    def index_map(self, i: int) -> IndexMap:
        """Index map for row/column.

        Arg:
            i: 0 for row map, 1 for column map.

        """
        return self._cpp_object.index_map(i)

    def add(self, x, rows, cols, bs=1) -> None:
        self._cpp_object.add(x, rows, cols, bs)

    def set(self, x, rows, cols, bs=1) -> None:
        self._cpp_object.set(x, rows, cols, bs)

    def set_value(self, x: np.floating) -> None:
        """Set all non-zero entries to a value.

        Args:
            x: The value to set all non-zero entries to.

        """
        self._cpp_object.set(x)

    def finalize(self) -> None:
        self._cpp_object.finalize()

    def squared_norm(self) -> np.floating:
        """Compute the squared Frobenius norm.

        Note:
            This operation is collective and requires communication.

        """
        return self._cpp_object.squared_norm()

    @property
    def data(self) -> npt.NDArray[np.floating]:
        """Underlying matrix entry data."""
        return self._cpp_object.data

    @property
    def indices(self) -> npt.NDArray[np.int32]:
        """Local column indices."""
        return self._cpp_object.indices

    @property
    def indptr(self) -> npt.NDArray[np.int64]:
        """Local row pointers."""
        return self._cpp_object.indptr

    def to_dense(self) -> npt.NDArray[np.floating]:
        """Copy to a dense 2D array.

        Note:
            Typically used for debugging.

        """
        return self._cpp_object.to_dense()


def matrix_csr(sp, block_mode=BlockMode.compact, dtype=np.float64) -> MatrixCSR:
    """Create a distributed sparse matrix.

    The matrix uses compressed sparse row storage.

    Args:
        sp: The sparsity pattern that defines the nonzero structure of
            the matrix the parallel distribution of the matrix.
        dtype: The scalar type.

    Returns:
        A sparse matrix.

    """
    if dtype == np.float32:
        ftype = _cpp.la.MatrixCSR_float32
    elif dtype == np.float64:
        ftype = _cpp.la.MatrixCSR_float64
    elif dtype == np.complex64:
        ftype = _cpp.la.MatrixCSR_complex64
    elif dtype == np.complex128:
        ftype = _cpp.la.MatrixCSR_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    return MatrixCSR(ftype(sp, block_mode))


class Vector:
    def __init__(self, x):
        """A distributed vector object.

        Args:
            map: Index map the describes the size and distribution of the vector
            bs: Block size

        Note:
            Objects of this type should be created using :func:`vector`
            and not created using the class initialiser.

        """
        self._cpp_object = x

    @property
    def index_map(self) -> IndexMap:
        return self._cpp_object.index_map

    @property
    def array(self) -> np.ndarray:
        return self._cpp_object.array

    def set(self, value: np.floating) -> None:
        self._cpp_object.set(value)

    def scatter_forward(self) -> None:
        self._cpp_object.scatter_forward()

    def scatter_reverse(self, mode: InsertMode):
        self._cpp_object.scatter_reverse(mode)

    def norm(self, type=_cpp.la.Norm.l2) -> np.floating:
        return self._cpp_object.norm(type)


def vector(map, bs=1, dtype=np.float64) -> Vector:
    """Create a distributed vector.

    Args:
        map: Index map the describes the size and distribution of the
            vector.
        bs: Block size.
        dtype: The scalar type.

    Returns:
        A distributed vector.

    """
    if dtype == np.float32:
        vtype = _cpp.la.Vector_float32
    elif dtype == np.float64:
        vtype = _cpp.la.Vector_float64
    elif dtype == np.complex64:
        vtype = _cpp.la.Vector_complex64
    elif dtype == np.complex128:
        vtype = _cpp.la.Vector_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    return Vector(vtype(map, bs))


def create_petsc_vector_wrap(x: Vector):
    """Wrap a distributed DOLFINx vector as a PETSc vector.

    Args:
        x: The vector to wrap as a PETSc vector.

    Returns:
        A PETSc vector that shares data with `x`.

    Note:
        The vector `x` must not be destroyed before the returned PETSc
        object.

    """
    return _cpp.la.petsc.create_vector_wrap(x._cpp_object)


def orthonormalize(basis):
    """Orthogoalise set of PETSc vectors in-place"""
    for i, x in enumerate(basis):
        for y in basis[:i]:
            alpha = x.dot(y)
            x.axpy(-alpha, y)
        x.normalize()


def is_orthonormal(basis, eps: float = 1.0e-12) -> bool:
    """Check that list of PETSc vectors are orthonormal"""
    for x in basis:
        if abs(x.norm() - 1.0) > eps:
            return False
    for i, x in enumerate(basis[:-1]):
        for y in basis[i + 1:]:
            if abs(x.dot(y)) > eps:
                return False
    return True
