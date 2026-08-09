# Copyright (C) 2017-2026 Chris N. Richardson, Garth N. Wells and
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

from collections.abc import Callable, Iterable, Sequence
from typing import ClassVar, Generic, overload

import numpy as np
import numpy.typing as npt

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx.fem.function import Constant, Function, FunctionSpace
from dolfinx.typing import Scalar


@overload
def locate_dofs_geometrical(V: dolfinx.fem.FunctionSpace, marker: Callable) -> np.ndarray: ...
@overload
def locate_dofs_geometrical(
    V: Iterable[dolfinx.fem.FunctionSpace], marker: Callable
) -> list[np.ndarray]: ...
def locate_dofs_geometrical(
    V: dolfinx.fem.FunctionSpace | Iterable[dolfinx.fem.FunctionSpace],
    marker: Callable,
) -> np.ndarray | list[np.ndarray]:
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

        If ``V`` is an iterable of function spaces, a list with one
        such array per space is returned instead, in the same order as
        ``V``.
    """
    if not isinstance(V, Iterable):
        return _cpp.fem.locate_dofs_geometrical(V._cpp_object, marker)

    _V = [space._cpp_object for space in V]
    return _cpp.fem.locate_dofs_geometrical(_V, marker)  # type: ignore[arg-type]


@overload
def locate_dofs_topological(
    V: dolfinx.fem.FunctionSpace,
    entity_dim: int,
    entities: npt.NDArray[np.int32],
    remote: bool = True,
) -> np.ndarray: ...
@overload
def locate_dofs_topological(
    V: Iterable[dolfinx.fem.FunctionSpace],
    entity_dim: int,
    entities: npt.NDArray[np.int32],
    remote: bool = True,
) -> list[np.ndarray]: ...
def locate_dofs_topological(
    V: dolfinx.fem.FunctionSpace | Iterable[dolfinx.fem.FunctionSpace],
    entity_dim: int,
    entities: npt.NDArray[np.int32],
    remote: bool = True,
) -> np.ndarray | list[np.ndarray]:
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

        If ``V`` is an iterable of function spaces, a list with one
        such array per space is returned instead, in the same order as
        ``V``.
    """
    _entities = np.asarray(entities, dtype=np.int32)
    if not isinstance(V, Iterable):
        return _cpp.fem.locate_dofs_topological(V._cpp_object, entity_dim, _entities, remote)

    _V = [space._cpp_object for space in V]
    return _cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)  # type: ignore[arg-type]


class DirichletBC(Generic[Scalar]):
    """Representation of Dirichlet boundary conditions.

    The conditions are imposed on a linear system.
    """

    # Matched-precision built-ins (geometry == real scalar part). Public:
    # extend with additional (scalar, geometry) dtype pairs as needed.
    cpp_types: ClassVar[dict] = {
        (np.dtype(np.float32), np.dtype(np.float32)): _cpp.fem.DirichletBC_float32,
        (np.dtype(np.float64), np.dtype(np.float64)): _cpp.fem.DirichletBC_float64,
        (np.dtype(np.complex64), np.dtype(np.float32)): _cpp.fem.DirichletBC_complex64,
        (np.dtype(np.complex128), np.dtype(np.float64)): _cpp.fem.DirichletBC_complex128,
    }
    _cpp_object: (
        _cpp.fem.DirichletBC_complex64
        | _cpp.fem.DirichletBC_complex128
        | _cpp.fem.DirichletBC_float32
        | _cpp.fem.DirichletBC_float64
    )

    def __init__(self, bc, V: FunctionSpace, g: Function | Constant):
        """Initialise a Dirichlet boundary condition.

        Note:
            Dirichlet boundary conditions  should normally be
            constructed using :func:`fem.dirichletbc` and not using this
            class initialiser. This class is combined with different
            base classes that depend on the scalar type of the boundary
            condition.

        Args:
            bc: C++ wrapped Dirichlet condition.
            V: Function space on which the boundary condition is
                defined.
            g: The boundary condition value(s).
        """
        self._cpp_object = bc
        self._V = V
        self._g = g

    @property
    def g(self) -> Function | Constant:
        """The boundary condition value(s)."""
        return self._g

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Function space on which the boundary condition is defined."""
        return self._V

    def set(
        self, x: npt.NDArray[Scalar], x0: npt.NDArray[Scalar] | None = None, alpha: float = 1
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

        ``x`` may be sized to hold only owned entries or to also
        include ghost entries (entries available on the calling rank
        but owned by another rank); a constrained degree-of-freedom
        beyond the end of ``x`` is silently skipped. Passing an
        owned-only array therefore sets only owned entries, while an
        array that also includes ghosts (e.g. the full array of a
        :class:`Vector<dolfinx.la.Vector>`) additionally sets ghost
        entries constrained by the condition.

        Args:
            x: Array to modify for Dirichlet boundary conditions. May
                include ghost entries.
            x0: Optional array used in computing the value to set. If
                not provided it is treated as zero. Must be at least
                as long as ``x`` (checked only in Developer builds).
            alpha: Scaling factor.
        """
        self._cpp_object.set(x, x0, alpha)  # type: ignore[arg-type]

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
    value: Function
    | Constant
    | npt.NDArray[Scalar]
    | np.floating
    | np.complexfloating
    | float
    | complex,
    dofs: npt.NDArray[np.int32] | Sequence[npt.NDArray[np.int32]],
    V: dolfinx.fem.FunctionSpace | None = None,
) -> DirichletBC[Scalar]:
    """Representation of Dirichlet boundary condition.

    Args:
        value: Lifted boundary values function. It must have a ``dtype``
            property.
        dofs: Local indices of degrees of freedom in function space to
            which boundary condition applies. When ``V`` is a sub-space
            and ``value``'s function space is a different (e.g.
            collapsed) space, this is a pair of arrays -- dof indices
            in ``V`` and the matching dof indices in ``value``'s
            function space -- as returned by
            :func:`locate_dofs_topological` or
            :func:`locate_dofs_geometrical` when passed a pair of
            spaces. Otherwise assumes function space of the problem is
            the same of function space of boundary values function.
        V: Function space of a problem to which boundary conditions are
            applied.

    Returns:
        A representation of the boundary condition for modifying linear
        systems.
    """
    if isinstance(value, float | complex):
        value = np.asarray(value)

    bctype: (
        type[_cpp.fem.DirichletBC_float32]
        | type[_cpp.fem.DirichletBC_float64]
        | type[_cpp.fem.DirichletBC_complex64]
        | type[_cpp.fem.DirichletBC_complex128]
    )
    try:
        dtype = value.dtype
    except AttributeError as err:
        raise AttributeError("Boundary condition value must have a dtype attribute.") from err

    # Geometry type is the mesh geometry type of the function space (or the
    # value's space), defaulting to matched precision when neither has one.
    if V is not None:
        geometry_dtype = np.dtype(V.mesh.geometry.x.dtype)
    elif isinstance(value, Function):
        geometry_dtype = np.dtype(value.function_space.mesh.geometry.x.dtype)
    else:
        geometry_dtype = np.dtype(dtype).type(0).real.dtype
    bctype = DirichletBC.cpp_types[dtype, geometry_dtype]

    if V is not None:
        V_used = V
    else:
        # The cpp constructor's V-less overload only accepts a Function,
        # so value must be one here.
        assert isinstance(value, Function)
        V_used = value.function_space

    # Promote a raw array/scalar to a Constant *before* constructing the
    # cpp object, so that mutating value.value in place afterwards
    # changes the same underlying storage that the boundary condition
    # reads.
    if not isinstance(value, Function | Constant):
        value = Constant(V_used.mesh, value)

    _value = value._cpp_object

    if V is not None:
        try:
            bc = bctype(_value, dofs, V)  # type: ignore
        except TypeError:
            bc = bctype(_value, dofs, V._cpp_object)  # type: ignore[arg-type]
    else:
        bc = bctype(_value, dofs)  # type: ignore

    return DirichletBC(bc, V_used, value)


def bcs_by_block(
    spaces: Iterable[FunctionSpace | None], bcs: Iterable[DirichletBC[Scalar]]
) -> list[list[DirichletBC[Scalar]]]:
    """Arrange boundary conditions by the space that they constrain.

    Given a sequence of function spaces ``spaces`` and a sequence of
    DirichletBC objects ``bcs``, return a list where the ith entry is
    the list of DirichletBC objects whose space is contained in
    ``space[i]``.
    """

    def _bc_space(V, bcs):
        """Return list of bcs that have the same space as V."""
        # V may be a wrapped FunctionSpace or a raw cpp FunctionSpace
        # (Form.function_spaces returns the latter), so normalise both
        # sides to cpp objects before calling the cpp-level contains().
        V_cpp = V._cpp_object if isinstance(V, FunctionSpace) else V
        return [bc for bc in bcs if V_cpp.contains(bc.function_space._cpp_object)]

    return [_bc_space(V, bcs) if V is not None else [] for V in spaces]
