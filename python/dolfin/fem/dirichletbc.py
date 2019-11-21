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


def locate_dofs_geometrical(V: typing.Union[function.FunctionSpace], marker: types.FunctionType):
    # Extract cpp function space
    try:
        _V = V._cpp_object
    except AttributeError:
        _V = V

    return cpp.fem.locate_dofs_geometrical(_V, marker)


def locate_dofs_topological(V: typing.Union[function.FunctionSpace],
                            entity_dim: int,
                            entities: typing.List[int]):
    # Extract cpp function space
    try:
        _V = V._cpp_object
    except AttributeError:
        _V = V

    return cpp.fem.locate_dofs_topological(_V, entity_dim, entities)


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace],
            value: typing.Union[ufl.Coefficient, function.Function, cpp.function.Function],
            dof_indices: typing.Union[typing.List[int]]):
        """Representation of Dirichlet boundary condition which is imposed on
        a linear system.

        """

        # FIXME: Handle (mesh function, index) marker type? If yes, use
        # tuple domain=(mf, index) to not have variable arguments

        # Extract cpp function space
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V

        # Construct bc value
        if isinstance(value, ufl.Coefficient):
            _value = value._cpp_object
        elif isinstance(value, cpp.function.Function):
            _value = value
        elif isinstance(value, function.Function):
            _value = value._cpp_object
        else:
            raise NotImplementedError

        super().__init__(_V, _value, dof_indices)
