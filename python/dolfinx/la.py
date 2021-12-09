# Copyright (C) 2017-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra functionality"""

from dolfinx.cpp.la import create_petsc_vector

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
