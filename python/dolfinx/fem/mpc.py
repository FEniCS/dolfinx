# Copyright (C) 2025 Jørgen S. Dokken and Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Pure-Python wrappers for the Multi-Point Constraint (MPC) API.

These wrappers hide the internal ``dolfinx.cpp`` bindings so that
user code never needs to import that private module directly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import dolfinx.cpp as _cpp
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import FunctionSpace
from dolfinx.la import BlockMode, MatrixCSR, SparsityPattern

__all__ = [
    "MPC",
    "apply_mpc_solution",
    "apply_mpc_vector",
    "assemble_matrix_mpc",
    "build_sparsity_pattern_mpc",
    "matrix_csr_mpc",
]

# Maps geometry dtype → (cpp MPC type, scalar dtype).
# For real-valued meshes the scalar type equals the geometry type;
# for complex problems a complex scalar lives on a real-geometry mesh.
_mpc_types: dict = {
    np.dtype(np.float32): (_cpp.fem.MPC_float32, np.dtype(np.float32)),
    np.dtype(np.float64): (_cpp.fem.MPC_float64, np.dtype(np.float64)),
}


class MPC:
    """Multi-Point Constraint of the form u = Σ c_k u_ref + g.

    Reference dofs are supplied as global indices with matching
    coefficients.  A constant term *g* is encoded as a reference with
    a negative global dof index; its coefficient is the constant value.

    See :func:`apply_mpc_solution` and :func:`apply_mpc_vector`.
    """

    def __init__(
        self,
        V: FunctionSpace,
        constrained_dofs_local: npt.NDArray[np.int32],
        global_dofs: list[npt.NDArray[np.int64]],
        global_coeffs: list[npt.NDArray],
    ):
        """Construct an MPC.

        Args:
            V: The function space.
            constrained_dofs_local: Local dof indices to constrain.
            global_dofs: For each constrained dof, a 1-D array of global
                reference dof indices.  A negative index encodes a constant
                term; its magnitude is irrelevant — only the sign matters.
            global_coeffs: Matching coefficient arrays.  For a constant
                entry (negative global index) the coefficient is the
                constant value *g*.
        """
        gdtype = V.mesh.geometry.x.dtype
        entry = _mpc_types.get(np.dtype(gdtype))
        if entry is None:
            raise TypeError(f"No MPC type for geometry dtype={gdtype}")
        cpp_type, dtype = entry
        self._cpp_object = cpp_type(
            V._cpp_object,
            np.asarray(constrained_dofs_local, dtype=np.int32),
            [np.asarray(d, dtype=np.int64) for d in global_dofs],
            [np.asarray(c, dtype=dtype) for c in global_coeffs],
        )

    @property
    def V(self) -> _cpp.fem.FunctionSpace_float64:  # type: ignore[return]
        """The (extended) FunctionSpace.

        Includes extra ghost reference dofs beyond the original V.
        """
        return self._cpp_object.V()

    def cells(self) -> npt.NDArray[np.int32]:
        """Return cells that contain at least one constrained dof."""
        return self._cpp_object.cells()

    def constraints(self):
        """Return constraint data for every local dof.

        Returns:
            Tuple ``(offsets, ref_dof, ref_coeff)``.  ``offsets`` is a
            prefix-sum array of length *num_dofs + 1*;
            ``ref_dof`` holds local reference dof indices;
            ``ref_coeff`` holds matching coefficients.
        """
        return self._cpp_object.constraints()


def build_sparsity_pattern_mpc(
    sp: SparsityPattern,
    a: Form,
    mpc_row: MPC,
    mpc_col: MPC,
) -> None:
    """Add MPC links to a sparsity pattern.

    Args:
        sp: Sparsity pattern (not yet finalised).
        a: Bilinear form.
        mpc_row: MPC for the row space.
        mpc_col: MPC for the column space.
    """
    _cpp.fem.build_sparsity_pattern_mpc(
        sp._cpp_object, a._cpp_object, mpc_row._cpp_object, mpc_col._cpp_object
    )


def matrix_csr_mpc(
    sp: SparsityPattern,
    block_mode: BlockMode = BlockMode.compact,
    dtype: npt.DTypeLike = np.float64,
) -> MatrixCSR:
    """Create a :class:`~dolfinx.la.MatrixCSR` from an MPC pattern.

    Same as :func:`dolfinx.la.matrix_csr` but accepts a finalised
    sparsity pattern so the caller does not unwrap ``._cpp_object``.

    Args:
        sp: A finalised :class:`~dolfinx.la.SparsityPattern`.
        block_mode: Block storage mode.
        dtype: Scalar type of the matrix.

    Returns:
        A new sparse matrix.
    """
    from dolfinx import la as _la  # avoid circular import at module level

    return _la.matrix_csr(sp, block_mode=block_mode, dtype=dtype)


def assemble_matrix_mpc(
    mpc: MPC,
    A: MatrixCSR,
    a: Form,
    bcs: list[DirichletBC],
) -> None:
    """Assemble a bilinear form into *A* with MPC row replacement.

    Args:
        mpc: The multipoint constraint.
        A: Matrix to assemble into (must have MPC sparsity pre-built).
        a: Bilinear form.
        bcs: Dirichlet boundary conditions.
    """
    _cpp.fem.assemble_matrix_mpc(
        mpc._cpp_object,
        A._cpp_object,
        a._cpp_object,
        [bc._cpp_object for bc in bcs],
    )


def apply_mpc_vector(b: npt.NDArray, mpc: MPC) -> None:
    """Apply the MPC P^T transformation to an assembled RHS vector.

    Distributes each constrained dof's entry to its reference dofs and
    zeros (or sets to the constant g) the constrained row.

    Args:
        b: Local solution-vector array (1-D, writable).
        mpc: The multipoint constraint.
    """
    _cpp.fem.apply_mpc_vector(b, mpc._cpp_object)


def apply_mpc_solution(u: npt.NDArray, mpc: MPC) -> None:
    """Recover constrained dof values after a linear solve.

    Sets ``u[constrained] = Σ c_k · u[ref_k] + g``.

    Args:
        u: Local solution-vector array (1-D, writable).  Reference ghost
            values must be current (call ``scatter_forward`` first).
        mpc: The multipoint constraint.
    """
    _cpp.fem.apply_mpc_solution(u, mpc._cpp_object)
