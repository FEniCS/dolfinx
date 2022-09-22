# Copyright (C) 2017-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra functionality"""


import numpy as np

from dolfinx import cpp as _cpp
from dolfinx.cpp.la import Norm, ScatterMode
from dolfinx.cpp.la.petsc import create_vector as create_petsc_vector

__all__ = ["orthonormalize", "is_orthonormal", "create_petsc_vector", "matrix_csr", "vector",
           "MatrixCSRMetaClass", "Norm", "ScatterMode", "VectorMetaClass", ]


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


class MatrixCSRMetaClass:
    def __init__(self, sp):
        """A distributed sparse matrix that uses compressed sparse row storage.

        Args:
            sp: The sparsity pattern that defines the nonzero structure
            of the matrix the parallel distribution of the matrix

        Note:
            Objects of this type should be created using
            :func:`matrix_csr` and not created using the class
            initialiser.

        """
        super().__init__(sp)


def matrix_csr(sp, dtype=np.float64) -> MatrixCSRMetaClass:
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

    matrixcls = type("MatrixCSR", (MatrixCSRMetaClass, ftype), {})
    return matrixcls(sp)


class VectorMetaClass:
    def __init__(self, map, bs):
        """A distributed vector object.

        Args:
            map: Index map the describes the size and distribution of
                the vector
            bs: Block size

        Note:
            Objects of this type should be created using :func:`vector`
            and not created using the class initialiser.

        """
        super().__init__(map, bs)  # type: ignore

    @property
    def array(self) -> np.ndarray:
        return super().array  # type: ignore


def vector(map, bs=1, dtype=np.float64) -> VectorMetaClass:
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

    vectorcls = type("Vector", (VectorMetaClass, vtype), {})
    return vectorcls(map, bs)
