# Copyright (C) 2026 Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""SuperLU Dist support."""

import numpy as np
import numpy.typing as npt

import dolfinx.cpp as _cpp


class SuperLUSolver:
    """SuperLU Solver."""

    _cpp_object: (
        _cpp.la.SuperLUSolver_float32
        | _cpp.la.SuperLUSolver_float64
        | _cpp.la.SuperLUSolver_complex128
    )

    def __init__(self, solver):
        """Create a SuperLU-dist solver.

        Args:
            solver: C++ SuperLUSolver object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`superlu_solver` to create a
            SuperLUSolver object.
        """
        self._cpp_object = solver

    def set_operator(self, A):
        """Set Operator."""
        self._cpp_object.set_operator(A._cpp_object)

    def solve(self, b, u):
        """Solver A.u=b."""
        self._cpp_object.solve(b._cpp_object, u._cpp_object)


def superlu_solver(comm, dtype: npt.DTypeLike = np.float64):
    """Create a SuperLU-dist solver object."""
    if np.issubdtype(dtype, np.float32):
        stype = _cpp.la.SuperLUSolver_float32
    elif np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUSolver_float64
    elif np.issubdtype(dtype, np.complex128):
        stype = _cpp.la.SuperLUSolver_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")
    return SuperLUSolver(stype(comm))
