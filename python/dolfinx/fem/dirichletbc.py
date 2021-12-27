# -*- coding: utf-8 -*-
# Copyright (C) 2017-2021 Chris N. Richardson, Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for representing Dirichlet boundary conditions that are enforced
via modification of linear systems.

"""

import collections.abc
import types
import typing

import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem.function import Constant, Function, FunctionSpace


def locate_dofs_geometrical(V: typing.Iterable[typing.Union[_cpp.fem.FunctionSpace, FunctionSpace]],
                            marker: types.FunctionType):
    """Locate degrees-of-freedom geometrically using a marker function.

    Parameters
    ----------
    V
        Function space(s) in which to search for degree-of-freedom indices.

    marker
        A function that takes an array of points ``x`` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to ``True`` for entities whose
        degree-of-freedom should be returned.

    Returns
    -------
    numpy.ndarray
        An array of degree-of-freedom indices (local to the process)
        for degrees-of-freedom whose coordinate evaluates to True for the
        marker function.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.

    """

    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
        return _cpp.fem.locate_dofs_geometrical(_V, marker)
    else:
        try:
            return _cpp.fem.locate_dofs_geometrical(V, marker)
        except TypeError:
            return _cpp.fem.locate_dofs_geometrical(V._cpp_object, marker)


def locate_dofs_topological(V: typing.Iterable[typing.Union[_cpp.fem.FunctionSpace, FunctionSpace]],
                            entity_dim: int,
                            entities: typing.List[int],
                            remote: bool = True):
    """Locate degrees-of-freedom belonging to mesh entities topologically.

    Parameters
    ----------
    V
        Function space(s) in which to search for degree-of-freedom indices.
    entity_dim
        Topological dimension of entities where degrees-of-freedom are located.
    entities
        Indices of mesh entities of dimension ``entity_dim`` where
        degrees-of-freedom are located.
    remote : True
        True to return also "remotely located" degree-of-freedom indices.

    Returns
    -------
    numpy.ndarray
        An array of degree-of-freedom indices (local to the process) for
        degrees-of-freedom topologically belonging to mesh entities.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.
    """

    _entities = np.asarray(entities, dtype=np.int32)
    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
        return _cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)
    else:
        try:
            return _cpp.fem.locate_dofs_topological(V, entity_dim, _entities, remote)
        except TypeError:
            return _cpp.fem.locate_dofs_topological(V._cpp_object, entity_dim, _entities, remote)


class DirichletBC:
    def __init__(self, value: typing.Union[ufl.Coefficient, Function, Constant],
                 dofs: typing.List[int], V: FunctionSpace = None):
        """Representation of Dirichlet boundary condition which is imposed on
        a linear system.

        Parameters
        ----------
        value
            Lifted boundary values function.
        dofs
            Local indices of degrees of freedom in function space to which
            boundary condition applies.
            Expects array of size (number of dofs, 2) if function space of the
            problem, ``V``, is passed. Otherwise assumes function space of the
            problem is the same of function space of boundary values function.
        V : optional
            Function space of a problem to which boundary conditions are applied.
        """

        # Unwrap value object, if required
        try:
            _value = value._cpp_object
        except AttributeError:
            _value = value

        if V is not None:
            try:
                super().__init__(_value, dofs, V)
            except TypeError:
                super().__init__(_value, dofs, V._cpp_object)
        else:
            super().__init__(_value, dofs)

    @property
    def g(self):
        """The boundary condition value(s)"""
        return self.value


def dirichletbc(value: typing.Union[ufl.Coefficient, Function, Constant],
                dofs: typing.List[int], V: FunctionSpace = None):

    # Determine the dtype
    try:
        dtype = value.x.array.dtype
    except AttributeError:
        dtype = value.dtype

    if dtype == np.float32:
        bctype = _cpp.fem.DirichletBC_float32
    elif dtype == np.float64:
        bctype = _cpp.fem.DirichletBC_float64
    elif dtype == np.complex64:
        bctype = _cpp.fem.DirichletBC_complex64
    elif dtype == np.complex128:
        bctype = _cpp.fem.DirichletBC_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    formcls = type("DirichletBC", (DirichletBC, bctype), {})
    return formcls(value, dofs, V)


def bcs_by_block(spaces: typing.Iterable[FunctionSpace],
                 bcs: typing.Iterable[DirichletBC]) -> typing.Iterable[typing.Iterable[DirichletBC]]:
    """Arrange Dirichlet boundary conditions by the function space that
    they constrain.

    Given a sequence of function spaces `spaces` and a sequence of
    DirichletBC objects `bcs`, return a list where the ith entry is the
    list of DirichletBC objects whose space is contained in
    `space[i]`."""
    def _bc_space(V, bcs):
        "Return list of bcs that have the same space as V"
        return [bc for bc in bcs if V.contains(bc.function_space)]

    return [_bc_space(V, bcs) if V is not None else [] for V in spaces]
