# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for representing Dirichlet boundary conditions that are enforced
via modification of linear systems.

"""

import types
import typing

import ufl
from dolfin import cpp, function


def locate_dofs_geometrical(V: typing.Union[function.FunctionSpace],
                            marker: types.FunctionType):
    """Locate degrees-of-freedom geometrically using a marker function.

    Parameters
    ----------
    V
        Function space in which to search for degree-of-freedom indices.

    maker
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

    """

    try:
        return cpp.fem.locate_dofs_geometrical(V._cpp_object, marker)
    except AttributeError:
        return cpp.fem.locate_dofs_geometrical(V, marker)


def locate_dofs_topological(V: typing.Union[function.FunctionSpace],
                            entity_dim: int,
                            entities: typing.List[int]):
    """Return array of degree-of-freedom indices (local to the process)
    for degrees-of-freedom belonging to the closure the mesh entities of
    dimension `entity_dim` and index in `entities`.
    """

    try:
        return cpp.fem.locate_dofs_topological(V._cpp_object, entity_dim, entities)
    except AttributeError:
        return cpp.fem.locate_dofs_topological(V, entity_dim, entities)


def locate_pair_dofs_topological(V0: typing.Union[function.FunctionSpace],
                                 V1: typing.Union[function.FunctionSpace],
                                 entity_dim: int,
                                 entities: typing.List[int]):
    """Return 2D array of degree-of-freedom indices (local to the process)
    for degrees-of-freedom belonging to the closure the mesh entities of
    dimension `entity_dim` and index in `entities`.
    """

    try:
        return cpp.fem.locate_pair_dofs_topological(V0._cpp_object, V1._cpp_object, entity_dim, entities)
    except AttributeError:
        return cpp.fem.locate_pair_dofs_topological(V0, V1, entity_dim, entities)


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(
            self,
            value: typing.Union[ufl.Coefficient, function.Function, cpp.function.Function],
            dofs: typing.List[int],
            V: typing.Union[function.FunctionSpace] = None
            ):
        """Representation of Dirichlet boundary condition which is imposed on
        a linear system.

        Parameters
        ----------
        V
            Function space of a problem to which boundary conditions are applied.
        value
            Lifted boundary values function.
        V_dofs
            Local indices of degrees of freedom in V function space to which
            boundary condition applies.
        g_dofs : optional
            Local indices of degrees of freedom in the space of value function
            to which boundary condition applies.
            If not specified, ``V_dofs`` is used.
        """

        # FIXME: Handle (mesh function, index) marker type? If yes, use
        # tuple domain=(mf, index) to not have variable arguments

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
            super().__init__(_V, _value, dofs)
        else:
            super().__init__(_value, dofs)
