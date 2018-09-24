# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import functools
import typing

import ufl
from dolfin import cpp
from dolfin.fem.assembling import _create_cpp_form
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.form import Form
from dolfin.jit.jit import ffc_jit

__all__ = ["Form"]


@functools.singledispatch
def assemble(M: typing.Union[Form, cpp.fem.Form]
             ) -> typing.Union[float, cpp.la.PETScMatrix, cpp.la.PETScVector]:
    """Assemble a form over mesh"""
    M_cpp = _create_cpp_form(M)
    return cpp.fem.assemble(M_cpp)


@assemble.register(cpp.la.PETScVector)
def assemble_vector(b: cpp.la.PETScVector,
                    L,
                    a=[],
                    bcs: typing.List[DirichletBC] = [],
                    scale: float = 1.0) -> cpp.la.PETScVector:
    """Re-assemble linear form into a vector, with modification for Dirichlet
    boundary conditions

    """
    try:
        L_cpp = [_create_cpp_form(form) for form in L]
        a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    except TypeError:
        L_cpp = [_create_cpp_form(L)]
        a_cpp = [[_create_cpp_form(form) for form in a]]

    cpp.fem.reassemble_blocked_vector(b, L_cpp, a_cpp, bcs, scale)
    return b


@assemble.register(cpp.la.PETScMatrix)
def assemble_matrix(A: cpp.la.PETScMatrix, a, bcs=[]) -> cpp.la.PETScMatrix:
    """Re-assemble bilinear form into a vector, with rows and columns with Dirichlet
    boundary conditions zeroed.

    """
    try:
        a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    except TypeError:
        a_cpp = [[_create_cpp_form(a)]]
    cpp.fem.reassemble_blocked_matrix(A, a_cpp, bcs)
    return A


def assemble_vector(L,
                    a,
                    bcs: typing.List[DirichletBC],
                    block_type,
                    scale: float = 1.0) -> cpp.la.PETScVector:
    """Assemble linear form into vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    return cpp.fem.assemble_blocked_vector(L_cpp, a_cpp, bcs, block_type,
                                           scale)


def assemble_matrix(a, bcs: typing.List[DirichletBC],
                    block_type) -> cpp.la.PETScMatrix:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    return cpp.fem.assemble_blocked_matrix(a_cpp, bcs, block_type)


def set_bc(b: cpp.la.PETScVector, L, bcs: typing.List[DirichletBC]) -> None:
    """Insert boundary condition values into vector"""
    cpp.fem.set_bc(b, L, bcs)


def create_coordinate_map(o):
    """Return a compiled UFC coordinate_mapping object"""

    try:
        # Create a compiled coordinate map from an object with the
        # ufl_mesh attribute
        cmap_ptr = ffc_jit(o.ufl_domain())
    except AttributeError:
        # FIXME: It would be good to avoid the type check, but ffc_jit
        # supports other objects so we could get, e.g., a compiled
        # finite element
        if isinstance(o, ufl.domain.Mesh):
            cmap_ptr = ffc_jit(o)
        else:
            raise TypeError(
                "Cannot create coordinate map from an object of type: {}".
                format(type(o)))
    except Exception:
        print("Failed to create compiled coordinate map")
        raise

    # Wrap compiled coordinate map and return
    ufc_cmap = cpp.fem.make_ufc_coordinate_mapping(cmap_ptr)
    return cpp.fem.CoordinateMapping(ufc_cmap)
