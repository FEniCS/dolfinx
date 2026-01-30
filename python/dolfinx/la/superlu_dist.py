# Copyright (C) 2026 Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""SuperLU Dist support."""

import numpy as np

import dolfinx
import dolfinx.cpp as _cpp

assert dolfinx.has_superlu_dist


class SuperLUDistSolver:
    """SuperLU_dist Solver."""

    _cpp_object: (
        _cpp.la.SuperLUDistSolver_float32
        | _cpp.la.SuperLUDistSolver_float64
        | _cpp.la.SuperLUDistSolver_complex128
    )

    def __init__(self, solver):
        """Create a SuperLU_DIST solver.

        Args:
            solver: C++ SuperLUDistSolver object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`superlu_dist_solver` to create a
            SuperLUDistSolver object.
        """
        self._cpp_object = solver

    def solve(self, b: dolfinx.la.Vector, u: dolfinx.la.Vector) -> dolfinx.la.Vector:
        """Solver linear system Au = b.

        Note:
            The caller must `u.scatter_forward()` after the solve.

        Args:
            b: Right-hand side vector
            u: Solution vector

        Returns:
            Solution vector, same object as u.
        """
        return self._cpp_object.solve(b._cpp_object, u._cpp_object)


def superlu_dist_solver(A: dolfinx.la.MatrixCSR) -> SuperLUDistSolver:
    """Create a SuperLU_DIST linear solver.

    Args:
        A: MatrixCSR object.
    """
    dtype = A.data.dtype
    if np.issubdtype(dtype, np.float32):
        stype = _cpp.la.SuperLUDistSolver_float32
    elif np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUDistSolver_float64
    elif np.issubdtype(dtype, np.complex128):
        stype = _cpp.la.SuperLUDistSolver_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    return SuperLUDistSolver(stype(A._cpp_object))
