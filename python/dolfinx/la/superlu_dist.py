# Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
# Copyright (C) 2026 Garth N. Wells
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

from collections.abc import Sequence
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

import dolfinx
import dolfinx.cpp as _cpp

if not dolfinx.has_superlu_dist:
    raise RuntimeError("DOLFINx was not built with SuperLU_DIST support.")

__all__ = ["SuperLUDistMatrix", "SuperLUDistSolver", "superlu_dist_matrix", "superlu_dist_solver"]

# As of 2026, SuperLU_DIST only supports these types, so the general Scalar
# type including np.complex64 cannot be used.
_T = TypeVar("_T", np.float32, np.float64, np.complex128)


class SuperLUDistMatrix(Generic[_T]):
    """SuperLU_DIST matrix."""

    _cpp_object: (
        _cpp.la.SuperLUDistMatrix_float32
        | _cpp.la.SuperLUDistMatrix_float64
        | _cpp.la.SuperLUDistMatrix_complex128
    )

    def __init__(
        self,
        matrix: (
            _cpp.la.SuperLUDistMatrix_float32
            | _cpp.la.SuperLUDistMatrix_float64
            | _cpp.la.SuperLUDistMatrix_complex128
        ),
    ):
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


def superlu_dist_matrix(
    A: dolfinx.la.MatrixCSR[_T] | Sequence[Sequence[dolfinx.la.MatrixCSR[_T] | None]],
) -> SuperLUDistMatrix[_T]:
    """Create a SuperLU_DIST matrix.

    Note:
        The matrix is copied into a SuperLU_DIST matrix, rather than being a
        view of the original matrices.

    Args:
        A: Assembled matrix, or a rectangular array of assembled blocks
            ``A[i][j]``, with ``None`` for a zero block. The blocks of a
            block row must have the same owned rows, and the blocks of a
            block column the same owned columns. Every block row and
            block column must have a block, and all blocks must have
            the same dtype.

    Returns:
        A SuperLU_DIST matrix.
    """
    if isinstance(A, dolfinx.la.MatrixCSR):
        dtype = A.data.dtype
        cpp_object = A._cpp_object
    else:
        blocks = [B for row in A for B in row if B is not None]
        if not blocks:
            raise ValueError("Block matrix has no blocks.")
        dtype = blocks[0].data.dtype
        if any(B.data.dtype != dtype for B in blocks):
            dtypes = sorted({str(B.data.dtype) for B in blocks})
            raise TypeError(f"Blocks have different dtypes: {', '.join(dtypes)}.")
        cpp_object = [[None if B is None else B._cpp_object for B in row] for row in A]
    stype: (
        type[_cpp.la.SuperLUDistMatrix_float32]
        | type[_cpp.la.SuperLUDistMatrix_float64]
        | type[_cpp.la.SuperLUDistMatrix_complex128]
    )
    if np.issubdtype(dtype, np.float32):
        stype = _cpp.la.SuperLUDistMatrix_float32
    elif np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUDistMatrix_float64
    elif np.issubdtype(dtype, np.complex128):
        stype = _cpp.la.SuperLUDistMatrix_complex128
    else:
        raise TypeError(f"Type {dtype} not supported.")
    return SuperLUDistMatrix(stype(cpp_object))  # type: ignore[arg-type]


class SuperLUDistSolver(Generic[_T]):
    """SuperLU_DIST solver."""

    _cpp_object: (
        _cpp.la.SuperLUDistSolver_float32
        | _cpp.la.SuperLUDistSolver_float64
        | _cpp.la.SuperLUDistSolver_complex128
    )

    def __init__(
        self,
        solver: (
            _cpp.la.SuperLUDistSolver_float32
            | _cpp.la.SuperLUDistSolver_float64
            | _cpp.la.SuperLUDistSolver_complex128
        ),
    ):
        """Create a SuperLU_DIST solver.

        Args:
            solver: C++ SuperLUDistSolver object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`superlu_dist_solver` to create a
            :class:`SuperLUDistSolver` object.
        """
        self._cpp_object = solver

    def set_option(self, name: str, value: str) -> None:
        """Set SuperLU_DIST option for solve.

        See SuperLU_DIST User's Guide for option names and values.

        Example::

            solver.set_option("SymmetricMode", "YES")
            solver.set_option("Trans", "NOTRANS")

        Args:
            name: Option name.
            value: Option value.
        """
        self._cpp_object.set_option(name, value)

    def set_A(self, A: SuperLUDistMatrix[_T], fact: str) -> None:
        """Set assembled left-hand side matrix.

        Args:
            A: Assembled left-hand side matrix :math:`A`.
            fact: One of ``"DOFACT"``, ``"SamePattern"``,
                ``"SamePattern_SameRowPerm"``. See the SuperLU_DIST
                documentation for the meaning of these values.
        """
        self._cpp_object.set_A(A._cpp_object, fact)  # type: ignore[arg-type]

    def solve(
        self,
        b: dolfinx.la.Vector[_T] | Sequence[dolfinx.la.Vector[_T]],
        u: dolfinx.la.Vector[_T] | Sequence[dolfinx.la.Vector[_T]],
    ) -> int:
        """Solve linear system :math:`Au = b`.

        Note:
            Vectors must have size and parallel layout compatible with
            ``A``. For a matrix created from blocks, ``b`` and ``u`` can
            be sequences of vectors, one per block row and block column,
            e.g. on the spaces of a block form. Their owned entries are
            stacked in the layout of the matrix for the solve.

        Note:
            The caller must check the return integer for success
            ``(== 0)``.

        Note:
            The caller must ``u.scatter_forward()`` (each vector in
            ``u``) after the solve.

        Note:
            The values of ``A`` are modified in-place during the solve.

        Note:
            To solve with successive right-hand sides `b` the caller must
            ``solver.set_option("Fact", "FACTORED")`` after the first
            solve.

        Args:
            b: Right-hand side vector :math:`b`, or its blocks.
            u: Solution vector :math:`u`, or its blocks, overwritten
                during solve.

        Returns:
           SuperLU_DIST return integer from ``p*gssvx`` routine.
        """
        if isinstance(b, dolfinx.la.Vector) and isinstance(u, dolfinx.la.Vector):
            return self._cpp_object.solve(b._cpp_object, u._cpp_object)  # type: ignore[arg-type]
        if isinstance(b, dolfinx.la.Vector) or isinstance(u, dolfinx.la.Vector):
            raise TypeError("b and u must both be vectors or both be sequences of vectors.")
        if len(b) != len(u) or len(b) == 0:
            raise ValueError("b and u must have the same, non-zero number of blocks.")

        # Owned entries of each block, stacked
        sizes = [bi.index_map.size_local * bi.block_size for bi in b]
        if any(
            ui.index_map.size_local * ui.block_size != n for ui, n in zip(u, sizes, strict=True)
        ):
            raise ValueError("Blocks of b and u have different owned sizes.")
        offsets = np.cumsum([0, *sizes])
        imap = dolfinx.common.index_map(b[0].index_map.comm, int(offsets[-1]))
        b_stacked = dolfinx.la.vector(imap, dtype=b[0].array.dtype)
        u_stacked = dolfinx.la.vector(imap, dtype=b[0].array.dtype)
        for bi, n, o in zip(b, sizes, offsets[:-1], strict=True):
            b_stacked.array[o : o + n] = bi.array[:n]

        status = self._cpp_object.solve(b_stacked._cpp_object, u_stacked._cpp_object)  # type: ignore[arg-type]
        for ui, n, o in zip(u, sizes, offsets[:-1], strict=True):
            ui.array[:n] = u_stacked.array[o : o + n]
        return status


def superlu_dist_solver(A: SuperLUDistMatrix[_T]) -> SuperLUDistSolver[_T]:
    """Create a SuperLU_DIST linear solver.

    Solve linear system :math:`Au = b` via LU decomposition.

    The SuperLU_DIST solver has options set to upstream defaults, except
    PrintStat (verbose solver output) set to NO and, when SuperLU_DIST is
    built with ParMETIS support, ColPerm set to PARMETIS.

    Args:
        A: Assembled left-hand side matrix :math:`A`.

    Returns:
        A SuperLU_DIST solver.
    """
    dtype = A.dtype
    stype: (
        type[_cpp.la.SuperLUDistSolver_float32]
        | type[_cpp.la.SuperLUDistSolver_float64]
        | type[_cpp.la.SuperLUDistSolver_complex128]
    )
    if np.issubdtype(dtype, np.float32):
        stype = _cpp.la.SuperLUDistSolver_float32
    elif np.issubdtype(dtype, np.float64):
        stype = _cpp.la.SuperLUDistSolver_float64
    elif np.issubdtype(dtype, np.complex128):
        stype = _cpp.la.SuperLUDistSolver_complex128
    else:
        raise TypeError(f"Type {dtype} not supported.")
    return SuperLUDistSolver(stype(A._cpp_object))  # type: ignore[arg-type]
