# -*- coding: utf-8 -*-
# Copyright (C) 2007-2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Functions for the assembly of variational forms."""

import functools
import typing

from dolfin import cpp
from dolfin.fem.assembling import _create_cpp_form
from dolfin.fem.dirichletbc import DirichletBC
from dolfin.fem.form import Form


@functools.singledispatch
def assemble(M: typing.Union[Form, cpp.fem.Form]
             ) -> typing.Union[float, cpp.la.PETScMatrix, cpp.la.PETScVector]:
    """Assemble a form over mesh"""
    M_cpp = _create_cpp_form(M)
    return cpp.fem.assemble(M_cpp)


@assemble.register(cpp.la.PETScVector)
def _assemble_vector(b: cpp.la.PETScVector,
                     L,
                     a=[],
                     bcs: typing.List[DirichletBC] = [],
                     x0: typing.Optional[cpp.la.PETScVector] = None,
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

    cpp.fem.reassemble_blocked_vector(b, L_cpp, a_cpp, bcs, x0, scale)
    return b


@assemble.register(cpp.la.PETScMatrix)
def _assemble_matrix(A: cpp.la.PETScMatrix, a, bcs=[],
                     diagonal: float = 1.0) -> cpp.la.PETScMatrix:
    """Re-assemble bilinear form into a vector, with rows and columns with Dirichlet
    boundary conditions zeroed.

    """
    try:
        a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    except TypeError:
        a_cpp = [[_create_cpp_form(a)]]
    cpp.fem.reassemble_blocked_matrix(A, a_cpp, bcs, diagonal)
    return A


def assemble_vector(L,
                    a,
                    bcs: typing.List[DirichletBC],
                    block_type: cpp.fem.BlockType,
                    x0: typing.Optional[cpp.la.PETScVector] = None,
                    scale: float = 1.0) -> cpp.la.PETScVector:
    """Assemble linear form into vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    return cpp.fem.assemble_blocked_vector(L_cpp, a_cpp, bcs, x0, block_type,
                                           scale)


def assemble_matrix(a,
                    bcs: typing.List[DirichletBC],
                    block_type,
                    diagonal: float = 1.0) -> cpp.la.PETScMatrix:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    return cpp.fem.assemble_blocked_matrix(a_cpp, bcs, block_type, diagonal)


def set_bc(b: cpp.la.PETScVector,
           bcs: typing.List[DirichletBC],
           x0: typing.Optional[cpp.la.PETScVector] = None,
           scale: float = 1.0) -> None:
    """Insert boundary condition values into vector"""
    cpp.fem.set_bc(b, bcs, x0, scale)
