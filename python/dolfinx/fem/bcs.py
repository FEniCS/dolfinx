# Copyright (C) 2017-2021 Chris N. Richardson, Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for representing Dirichlet boundary conditions that are enforced
via modification of linear systems."""

from __future__ import annotations

import numbers
import typing

import numpy.typing

if typing.TYPE_CHECKING:
    from dolfinx.fem.function import Constant, Function

import numpy as np

import dolfinx
from dolfinx import cpp as _cpp


def locate_dofs_geometrical(
    V: typing.Union[dolfinx.fem.FunctionSpace, typing.Iterable[dolfinx.fem.FunctionSpace]],
    marker: typing.Callable,
) -> np.ndarray:
    """Locate degrees-of-freedom geometrically using a marker function.

    Args:
        V: Function space(s) in which to search for degree-of-freedom indices.
        marker: A function that takes an array of points ``x`` with shape
            ``(gdim, num_points)`` and returns an array of booleans of
            length ``num_points``, evaluating to ``True`` for entities whose
            degree-of-freedom should be returned.

    Returns:
        An array of degree-of-freedom indices (local to the process)
        for degrees-of-freedom whose coordinate evaluates to True for the
        marker function.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.

    """

    try:
        return _cpp.fem.locate_dofs_geometrical(V._cpp_object, marker)  # type: ignore
    except AttributeError:
        _V = [space._cpp_object for space in V]
        return _cpp.fem.locate_dofs_geometrical(_V, marker)


def locate_dofs_topological(
    V: typing.Union[dolfinx.fem.FunctionSpace, typing.Iterable[dolfinx.fem.FunctionSpace]],
    entity_dim: int,
    entities: numpy.typing.NDArray[np.int32],
    remote: bool = True,
) -> np.ndarray:
    """Locate degrees-of-freedom belonging to mesh entities topologically.

    Args:
        V: Function space(s) in which to search for degree-of-freedom indices.
        entity_dim: Topological dimension of entities where degrees-of-freedom are located.
        entities: Indices of mesh entities of dimension ``entity_dim`` where
            degrees-of-freedom are located.
        remote: True to return also "remotely located" degree-of-freedom indices.

    Returns:
        An array of degree-of-freedom indices (local to the process) for
        degrees-of-freedom topologically belonging to mesh entities.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.
    """

    _entities = np.asarray(entities, dtype=np.int32)
    try:
        return _cpp.fem.locate_dofs_topological(V._cpp_object, entity_dim, _entities, remote)  # type: ignore
    except AttributeError:
        _V = [space._cpp_object for space in V]
        return _cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)


class DirichletBC:
    _cpp_object: typing.Union[
        _cpp.fem.DirichletBC_complex64,
        _cpp.fem.DirichletBC_complex128,
        _cpp.fem.DirichletBC_float32,
        _cpp.fem.DirichletBC_float64,
    ]

    def __init__(self, bc):
        """Representation of Dirichlet boundary condition which is imposed on
        a linear system.

        Note:
            Dirichlet boundary conditions  should normally be
            constructed using :func:`fem.dirichletbc` and not using this
            class initialiser. This class is combined with different
            base classes that depend on the scalar type of the boundary
            condition.

        Args:
            value: Lifted boundary values function.
            dofs: Local indices of degrees of freedom in function space to which
                boundary condition applies. Expects array of size (number of
                dofs, 2) if function space of the problem, ``V`` is passed.
                Otherwise assumes function space of the problem is the same
                of function space of boundary values function.
            V: Function space of a problem to which boundary conditions are applied.

        """
        self._cpp_object = bc

    @property
    def g(self) -> typing.Union[Function, Constant, np.ndarray]:
        """The boundary condition value(s)"""
        return self._cpp_object.value

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """The function space on which the boundary condition is defined"""
        return self._cpp_object.function_space


def dirichletbc(
    value: typing.Union[Function, Constant, np.ndarray],
    dofs: numpy.typing.NDArray[np.int32],
    V: typing.Optional[dolfinx.fem.FunctionSpace] = None,
) -> DirichletBC:
    """Create a representation of Dirichlet boundary condition which
    is imposed on a linear system.

    Args:
        value: Lifted boundary values function. It must have a ``dtype``
        property.
        dofs: Local indices of degrees of freedom in function space to
            which boundary condition applies. Expects array of size
            (number of dofs, 2) if function space of the problem, ``V``,
            is passed. Otherwise assumes function space of the problem
            is the same of function space of boundary values function.
        V: Function space of a problem to which boundary conditions are applied.

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
            _value = value._cpp_object  # type: ignore
        except AttributeError:
            _value = value

    if V is not None:
        try:
            bc = bctype(_value, dofs, V)
        except TypeError:
            bc = bctype(_value, dofs, V._cpp_object)
    else:
        bc = bctype(_value, dofs)

    return DirichletBC(bc)


def bcs_by_block(
    spaces: typing.Iterable[typing.Union[dolfinx.fem.FunctionSpace, None]],
    bcs: typing.Iterable[DirichletBC],
) -> list[list[DirichletBC]]:
    """Arrange Dirichlet boundary conditions by the function space that
    they constrain.

    Given a sequence of function spaces ``spaces`` and a sequence of
    DirichletBC objects ``bcs``, return a list where the ith entry is
    the list of DirichletBC objects whose space is contained in
    ``space[i]``.

    """

    def _bc_space(V, bcs):
        "Return list of bcs that have the same space as V"
        return [bc for bc in bcs if V.contains(bc.function_space)]

    return [_bc_space(V, bcs) if V is not None else [] for V in spaces]
