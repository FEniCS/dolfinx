# Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""SuperLU_DIST linear solver support.

This module provides support for parallel solution of linear systems
assembled into :class:`dolfinx.la.MatrixCSR` using SuperLU_DIST.

Note:
  Users with advanced linear solver requirements should use PETSc/petsc4py.
"""

import numpy as np
import numpy.typing as npt

import dolfinx
import dolfinx.cpp as _cpp

assert dolfinx.has_superlu_dist

__all__ = ["SuperLUDistMatrix", "SuperLUDistSolver", "superlu_dist_matrix", "superlu_dist_solver"]


class SuperLUDistMatrix:
    """SuperLU_DIST matrix."""

    _cpp_object: (
        _cpp.la.SuperLUDistMatrix_float32
        | _cpp.la.SuperLUDistMatrix_float64
        | _cpp.la.SuperLUDistMatrix_complex128
    )
    _dtype: npt.DTypeLike

    def __init__(self, matrix):
        """Create a SuperLU_DIST matrix.

        Args:
            matrix: C++ SuperLUDistMatrix object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`superlu_dist_matrix` to create a
            :class:`SuperLUDistMatrix` object.
        """
        self._cpp_object = matrix

    @property
    def dtype(self) -> npt.DTypeLike:
        """Dtype of matrix values."""
        return self._cpp_object.dtype


def superlu_dist_matrix(A: dolfinx.la.MatrixCSR) -> SuperLUDistMatrix:
    """Create a SuperLU_DIST matrix.

    Deep copies all required data from ``A``.

    Args:
        A: Assembled matrix.

    Returns:
        A SuperLU_DIST matrix.
    """
    dtype = A.data.dtype
    if np.issubdtype(dtype, np.float32):
        stype = _cpp.la.SuperLUDistMatrix_float32
    elif np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUDistMatrix_float64
    elif np.issubdtype(dtype, np.complex128):
        stype = _cpp.la.SuperLUDistMatrix_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    return SuperLUDistMatrix(stype(A._cpp_object))


class SuperLUDistSolver:
    """SuperLU_DIST solver."""

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

    def set_A(self, A: SuperLUDistMatrix):
        """Set assembled left-hand side matrix.

        For advanced use with SuperLU_DIST option `Factor` allowing
        use of previously computed factors with new matrix A.

        Args:
            A: Assembled left-hand side matrix :math:`A`.
        """
        self._cpp_object.set_A(A._cpp_object)

    def solve(self, b: dolfinx.la.Vector, u: dolfinx.la.Vector) -> int:
        """Solve linear system :math:`Au = b`.

        Note:
            The caller must check the return integer for success
            ``(== 0)``.

        Note:
            The caller must ``u.scatter_forward()`` after the solve.

        Note:
            The values of ``A`` are modified in-place during the solve.

        Note:
            To solve with successive right-hand sides :math:`b`
            the user must set ``solver.set_option("Factor", "FACTORED")``
            after the first solve.

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


def superlu_dist_solver(A: SuperLUDistMatrix) -> SuperLUDistSolver:
    """Create a SuperLU_DIST linear solver.

    Solve linear system :math:`Au = b` via LU decomposition.

    The SuperLU_DIST solver has options set to upstream defaults, except
    PrintStat (verbose solver output) set to NO.

    Args:
        A: Assembled left-hand side matrix :math:`A`.

    Returns:
        A SuperLU_DIST solver.
    """
    dtype = A.dtype
    if np.issubdtype(dtype, np.float32):
        stype = _cpp.la.SuperLUDistSolver_float32
    elif np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUDistSolver_float64
    elif np.issubdtype(dtype, np.complex128):
        stype = _cpp.la.SuperLUDistSolver_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    return SuperLUDistSolver(stype(A._cpp_object))
