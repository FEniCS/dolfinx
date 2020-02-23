# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for representing Dirichlet boundary conditions that are enforced
via modification of linear systems.

"""

import types
import typing
import collections.abc

import ufl
from dolfinx import cpp, function


def locate_dofs_geometrical(V: typing.Iterable[typing.Union[cpp.function.FunctionSpace, function.FunctionSpace]],
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
    else:
        try:
            _V = [V._cpp_object]
        except AttributeError:
            _V = [V]

    return cpp.fem.locate_dofs_geometrical(_V, marker)


def locate_dofs_topological(V: typing.Iterable[typing.Union[cpp.function.FunctionSpace, function.FunctionSpace]],
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

    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
    else:
        try:
            _V = [V._cpp_object]
        except AttributeError:
            _V = [V]

    return cpp.fem.locate_dofs_topological(_V, entity_dim, entities, remote)


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(
            self,
            value: typing.Union[ufl.Coefficient, function.Function, cpp.function.Function],
            dofs: typing.List[int],
            V: typing.Union[function.FunctionSpace] = None):
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

        # Construct bc value
        if isinstance(value, ufl.Coefficient):
            _value = value._cpp_object
        elif isinstance(value, cpp.function.Function):
            _value = value
        elif isinstance(value, function.Function):
            _value = value._cpp_object
        else:
            raise NotImplementedError

        if V is not None:
            # Extract cpp function space
            try:
                _V = V._cpp_object
            except AttributeError:
                _V = V
            super().__init__(_value, dofs, _V)
        else:
            super().__init__(_value, dofs)
