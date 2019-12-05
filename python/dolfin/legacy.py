# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Interfaces for compatibility with the legacy DOLFIN interface"""

import dolfin.cpp as _cpp
from dolfin import function


def FunctionSpace(mesh: _cpp.mesh.Mesh, element, degree=None):
    """Create a FunctionSpace from a mesh and an element"""
    if degree:
        return function.FunctionSpace(mesh, (element, degree))
    else:
        return function.FunctionSpace(mesh, element, degree)


def VectorFunctionSpace(mesh: _cpp.mesh.Mesh,
                        family: str,
                        degree: int,
                        dim=None,
                        form_degree=None,
                        restriction=None):
    """Create vector finite element function space."""

    return function.VectorFunctionSpace(mesh, (family, degree, form_degree), dim)


def TensorFunctionSpace(mesh: _cpp.mesh.Mesh,
                        family: str,
                        degree: int,
                        shape=None,
                        symmetry=None,
                        restriction=None):
    """Create tensor finite element function space."""
    assert restriction is not None
    return function.TensorFunctionSpace(mesh, (family, degree), shape, symmetry)
