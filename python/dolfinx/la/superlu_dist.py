# Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""SuperLU_DIST linear solver support.

This module provides basic support for parallel solution of linear systems
assembled into :class:`dolfinx.la.MatrixCSR`.

Note:
  Users with more advanced linear solver needs should use PETSc/petsc4py.
"""

import numpy as np

import dolfinx
import dolfinx.cpp as _cpp

assert dolfinx.has_superlu_dist

__all__ = ["SuperLUDistSolver", "superlu_dist_solver"]


class SuperLUDistSolver:
    """SuperLU_DIST solver."""

    _cpp_object: (
        _cpp.la.SuperLUDistSolver_float64
    )

    def __init__(self, solver):
        """Create a SuperLU_DIST solver.

        Args:
            solver: C++ SuperLUDistSolver object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`superlu_dist_solver` to create a
            :class:`SuperLUDistSolver` object.
        """
        self._cpp_object = solver

    def set_option(self, name: str, value: str):
        """Set SuperLU_DIST option for solve.

        See SuperLU_DIST User's Guide for option names and values.

        Examples:
            solver.set_option("SymmetricMode", "YES")
            solver.set_option("Trans", "NOTRANS")

        Args:
            name: Option name.
            value: Option value.
        """
        self._cpp_object.set_option(name, value)

    def solve(self, b: dolfinx.la.Vector, u: dolfinx.la.Vector) -> int:
        """Solve linear system :math:`Au = b`.

        Note:
            The caller must check the return integer for success.
            ``(== 0)``.

        Note:
            The caller must ``u.scatter_forward()`` after the solve.

        Note:
            Vectors must have size and parallel layout compatible with
            ``A``.

        Args:
            b: Right-hand side vector :math:`b`.
            u: Solution vector :math:`u`, overwritten during solve.

        Returns:
           SuperLU_DIST return integer from ``p*gssvx`` routine.
        """
        return self._cpp_object.solve(b._cpp_object, u._cpp_object)


def superlu_dist_solver(A: dolfinx.la.MatrixCSR) -> SuperLUDistSolver:
    """Create a SuperLU_DIST linear solver.

    Solve linear system :math:`Au = b` via LU decomposition.

    The SuperLU_DIST solver has options set to upstream defaults, except
    PrintStat (verbose solver output) set to NO.

    Args:
        A: Assembled left-hand side matrix :math:`A`.
    """
    dtype = A.data.dtype
    if np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUDistSolver_float64
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    return SuperLUDistSolver(stype(A._cpp_object))
