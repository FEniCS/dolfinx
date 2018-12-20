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
from dolfin import cpp, function, mesh


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace, cpp.function.
                            FunctionSpace],
            value: typing.Union[ufl.Coefficient, cpp.function.Function],
            domain: typing.Union[cpp.mesh.SubDomain, types.FunctionType, typing.List[int]],
            method: cpp.fem.DirichletBC.Method = cpp.fem.DirichletBC.Method.topological,
            check_midpoint: typing.Optional[bool] = None):
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
        else:
            raise NotImplementedError

        # Construct domain
        if isinstance(domain, types.FunctionType):
            # Keep reference to subdomain to avoid out-of-scope problem
            self._sub_domain = mesh.create_subdomain(domain)
            _domain = self._sub_domain
        else:
            _domain = domain

        if isinstance(_domain, cpp.mesh.SubDomain):
            if check_midpoint is None:
                check_midpoint = True
            super().__init__(_V, _value, _domain, method, check_midpoint)
        else:
            assert check_midpoint is None
            super().__init__(_V, _value, _domain, method)
