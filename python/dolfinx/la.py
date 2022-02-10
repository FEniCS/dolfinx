# Copyright (C) 2017-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra functionality"""


import numpy as np

from dolfinx import cpp as _cpp
from dolfinx.cpp.la.petsc import create_vector as create_petsc_vector

__all__ = ["orthonormalize", "is_orthonormal", "create_petsc_vector"]


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


class MatrixCSRCMetaClass:
    def __init__(self, sp):
        """A sparse matrix that uses compressed sparse row storage.

        Args:
            sp: The sparsity pattern that defines the nonzero structure
            of the matrix the parallel distribution of the matrix
        """
        super().__init__(sp)


def matrix_csr(sp, dtype=np.float64) -> MatrixCSRCMetaClass:
    """Create a sparse matrix.

    The matrix uses compressed sparse row storage.

    Args:
        sp: The sparsity pattern that defines the nonzero structure of
        the matrix the parallel distribution of the matrix.

    Returns:
        A sparse matrix.
    """
    if dtype == np.float32:
        ftype = _cpp.la.MatrixCSR_float32
    elif dtype == np.float64:
        ftype = _cpp.la.MatrixCSR_float64
    elif dtype == np.complex128:
        ftype = _cpp.la.MatrixCSR_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    matrixcls = type("MatrixCSR", (MatrixCSRCMetaClass, ftype), {})
    return matrixcls(sp)
