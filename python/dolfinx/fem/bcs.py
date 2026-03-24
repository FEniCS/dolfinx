# Copyright (C) 2017-2021 Chris N. Richardson, Garth N. Wells and
# Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Dirichlet boundary conditions.

Representations of Dirichlet boundary conditions that are enforced via
modification of linear systems.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx.fem.function import Constant, Function, FunctionSpace


def locate_dofs_geometrical(
    V: dolfinx.fem.FunctionSpace | Iterable[dolfinx.fem.FunctionSpace],
    marker: Callable,
) -> np.ndarray:
    """Locate degrees-of-freedom geometrically using a marker function.

    Args:
        V: Function space(s) in which to search for degree-of-freedom
            indices.
        marker: A function that takes an array of points ``x`` with
            shape ``(gdim, num_points)`` and returns an array of
            booleans of length ``num_points``, evaluating to ``True``
            for entities whose degree-of-freedom should be returned.

    Returns:
        An array of degree-of-freedom indices (local to the process) for
        degrees-of-freedom whose coordinate evaluates to True for the
        marker function.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.
    """
    if not isinstance(V, Iterable):
        return _cpp.fem.locate_dofs_geometrical(V._cpp_object, marker)  # type: ignore

    _V = [space._cpp_object for space in V]  # type: ignore
    return _cpp.fem.locate_dofs_geometrical(_V, marker)


def locate_dofs_topological(
    V: dolfinx.fem.FunctionSpace | Iterable[dolfinx.fem.FunctionSpace],
    entity_dim: int,
    entities: npt.NDArray[np.int32],
    remote: bool = True,
) -> np.ndarray:
    """Locate degrees-of-freedom belonging to mesh entities topologically.

    Args:
        V: Function space(s) in which to search for degree-of-freedom
            indices.
        entity_dim: Topological dimension of entities where
            degrees-of-freedom are located.
        entities: Indices of mesh entities of dimension ``entity_dim``
            where degrees-of-freedom are located.
        remote: True to return also "remotely located" degree-of-freedom
            indices.

    Returns:
        An array of degree-of-freedom indices (local to the process) for
        degrees-of-freedom topologically belonging to mesh entities.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.
    """
    _entities = np.asarray(entities, dtype=np.int32)
    if not isinstance(V, Iterable):
        return _cpp.fem.locate_dofs_topological(V._cpp_object, entity_dim, _entities, remote)  # type: ignore

    _V = [space._cpp_object for space in V]  # type: ignore
    return _cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)


class DirichletBC:
    """Representation of Dirichlet boundary conditions.

    The conditions are imposed on a linear system.
    """

    _cpp_object: (
        _cpp.fem.DirichletBC_complex64
        | _cpp.fem.DirichletBC_complex128
        | _cpp.fem.DirichletBC_float32
        | _cpp.fem.DirichletBC_float64
    )

    def __init__(self, bc):
        """Initialise a Dirichlet boundary condition.

        Note:
            Dirichlet boundary conditions  should normally be
            constructed using :func:`fem.dirichletbc` and not using this
            class initialiser. This class is combined with different
            base classes that depend on the scalar type of the boundary
            condition.

        Args:
            bc: C++ wrapped Dirichlet condition.
        """
        self._cpp_object = bc

    @property
    def g(self) -> Function | Constant | np.ndarray:
        """The boundary condition value(s)."""
        return self._cpp_object.value

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Function space on which the boundary condition is defined."""
        return self._cpp_object.function_space

    def set(
        self, x: npt.NDArray, x0: npt.NDArray[np.int32] | None = None, alpha: float = 1
    ) -> None:
        """Set array entries that are constrained by a Dirichlet condition.

        Entries in ``x`` that are constrained by a Dirichlet boundary
        conditions are set to ``alpha * (x_bc - x0)``, where ``x_bc`` is
        the (interpolated) boundary condition value.

        For elements with point-wise evaluated degrees-of-freedom, e.g.
        Lagrange elements, ``x_bc`` is the value of the boundary condition
        at the degree-of-freedom. For elements with moment
        degrees-of-freedom, ``x_bc`` is the value of the boundary condition
        interpolated into the finite element space.

        If `x` includes ghosted entries (entries available on the calling
        rank but owned by another rank), ghosted entries constrained by a
        Dirichlet condition will also be set.

        Args:
            x: Array to modify for Dirichlet boundary conditions.
            x0: Optional array used in computing the value to set. If
                not provided it is treated as zero.
            alpha: Scaling factor.
        """
        self._cpp_object.set(x, x0, alpha)

    def dof_indices(self) -> tuple[npt.NDArray[np.int32], int]:
        """Dof indices to  which a Dirichlet condition is applied.

        Note:
            Returned array is read-only.

        Returns:
            (i) Sorted array of dof indices (unrolled) and (ii) index to
            the first entry in the dof index array that is not owned.
            Entries `dofs[:pos]` are owned and entries `dofs[pos:]` are
            ghosts.
        """
        return self._cpp_object.dof_indices()


def dirichletbc(
    value: Function | Constant | np.ndarray,
    dofs: npt.NDArray[np.int32],
    V: dolfinx.fem.FunctionSpace | None = None,
) -> DirichletBC:
    """Representation of Dirichlet boundary condition.

    Args:
        value: Lifted boundary values function. It must have a ``dtype``
            property.
        dofs: Local indices of degrees of freedom in function space to
            which boundary condition applies. Expects array of size
            (number of dofs, 2) if function space of the problem, ``V``,
            is passed. Otherwise assumes function space of the problem
            is the same of function space of boundary values function.
        V: Function space of a problem to which boundary conditions are
            applied.

    Returns:
        A representation of the boundary condition for modifying linear
        systems.
    """
    if isinstance(value, numbers.Number):
        value = np.asarray(value)

    try:
        dtype = value.dtype
        if np.issubdtype(dtype, np.float32):
            bctype = _cpp.fem.DirichletBC_float32
        elif np.issubdtype(dtype, np.float64):
            bctype = _cpp.fem.DirichletBC_float64
        elif np.issubdtype(dtype, np.complex64):
            bctype = _cpp.fem.DirichletBC_complex64
        elif np.issubdtype(dtype, np.complex128):
            bctype = _cpp.fem.DirichletBC_complex128
        else:
            raise NotImplementedError(f"Type {value.dtype} not supported.")
    except AttributeError:
        raise AttributeError("Boundary condition value must have a dtype attribute.")

    # Unwrap value object, if required
    if isinstance(value, np.ndarray):
        _value = value
    else:
        try:
            _value = value._cpp_object
        except AttributeError:
            _value = value  # type: ignore[assignment]

    if V is not None:
        try:
            bc = bctype(_value, dofs, V)
        except TypeError:
            bc = bctype(_value, dofs, V._cpp_object)
    else:
        bc = bctype(_value, dofs)

    return DirichletBC(bc)


def bcs_by_block(
    spaces: Iterable[FunctionSpace | None], bcs: Iterable[DirichletBC]
) -> list[list[DirichletBC]]:
    """Arrange boundary conditions by the space that they constrain.

    Given a sequence of function spaces ``spaces`` and a sequence of
    DirichletBC objects ``bcs``, return a list where the ith entry is
    the list of DirichletBC objects whose space is contained in
    ``space[i]``.
    """

    def _bc_space(V, bcs):
        """Return list of bcs that have the same space as V."""
        return [bc for bc in bcs if V.contains(bc.function_space)]

    return [_bc_space(V, bcs) if V is not None else [] for V in spaces]
