# -*- coding: utf-8 -*-
# Copyright (C) 2007-2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Functions for the assembly of variational forms."""

import functools
import typing

from petsc4py import PETSc

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
                     L: typing.Union[Form, cpp.fem.Form]) -> cpp.la.PETScVector:
    """Re-assemble linear form into a vector."""
    L_cpp = _create_cpp_form(L)
    cpp.fem.assemble_vector(b.vec(), L_cpp)
    b.vec().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


@assemble.register(cpp.la.PETScMatrix)
def _assemble_matrix(A: cpp.la.PETScMatrix, a: typing.Union[Form, cpp.fem.Form], bcs=[],
                     diagonal: float = 1.0) -> cpp.la.PETScMatrix:
    """Assemble bilinear form into a vector, with rows and columns with Dirichlet
    boundary conditions zeroed.

    """
    A.mat().zeroEntries()
    a_cpp = _create_cpp_form(a)
    cpp.fem.assemble_matrix(A.mat(), a_cpp, bcs, diagonal)
    A.mat().assemble()
    return A


@functools.singledispatch
def assemble_vector(
        L: typing.Union[Form, cpp.fem.Form]) -> cpp.la.PETScVector:
    """Assemble linear form into a vector."""
    L_cpp = _create_cpp_form(L)
    _b = cpp.la.PETScVector(L_cpp.function_space(0).dofmap().index_map())
    cpp.fem.assemble_vector(_b.vec(), L_cpp)
    return _b


@assemble_vector.register(cpp.la.PETScVector)
def _reassemble_vector(
        b: cpp.la.PETScVector,
        L: typing.Union[Form, cpp.fem.Form]) -> cpp.la.PETScVector:
    """Re-assemble linear form into a vector."""
    L_cpp = _create_cpp_form(L)
    cpp.fem.assemble_vector(b.vec(), L_cpp)
    return b


def assemble_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                         a,
                         bcs: typing.List[DirichletBC],
                         x0: typing.Optional[cpp.la.PETScVector] = None,
                         scale: float = 1.0) -> cpp.la.PETScVector:
    """Assemble linear forms into a nested vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    _b = cpp.fem.create_vector_nest(L_cpp)
    _x0 = x0.vec() if x0 is not None else None
    cpp.fem.assemble_vector(_b.vec(), L_cpp, a_cpp, bcs, _x0, scale)
    return _b


def assemble_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                          a,
                          bcs: typing.List[DirichletBC],
                          x0: typing.Optional[cpp.la.PETScVector] = None,
                          scale: float = 1.0) -> cpp.la.PETScVector:
    """Assemble linear forms into a monolithic vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    b = cpp.fem.create_vector(L_cpp)
    _x0 = x0.vec() if x0 is not None else None
    cpp.fem.assemble_vector(b.vec(), L_cpp, a_cpp, bcs, _x0, scale)
    return b


def reassemble_vector(b: cpp.la.PETScVector,
                      L,
                      a=[],
                      bcs: typing.List[DirichletBC] = [],
                      x0: typing.Optional[cpp.la.PETScVector] = None,
                      scale: float = 1.0) -> cpp.la.PETScVector:
    """Re-assemble linear forms into a block vector, with modification for Dirichlet
    boundary conditions

    """
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    _x0 = x0.vec() if x0 is not None else None
    cpp.fem.reassemble_vector(b, L_cpp, a_cpp, bcs, _x0, scale)
    return b


def assemble_matrix_nest(a,
                         bcs: typing.List[DirichletBC],
                         diagonal: float = 1.0) -> cpp.la.PETScMatrix:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    A = cpp.fem.create_matrix_nest(a_cpp)
    A.mat().zeroEntries()
    cpp.fem.assemble_blocked_matrix(A.mat(), a_cpp, bcs, diagonal)
    return A


def assemble_matrix_block(a,
                          bcs: typing.List[DirichletBC],
                          diagonal: float = 1.0) -> cpp.la.PETScMatrix:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    A = cpp.fem.create_matrix_block(a_cpp)
    A.mat().zeroEntries()
    cpp.fem.assemble_blocked_matrix(A.mat(), a_cpp, bcs, diagonal)
    return A


def assemble_matrix(a,
                    bcs: typing.List[DirichletBC],
                    diagonal: float = 1.0) -> cpp.la.PETScMatrix:
    """Assemble bilinear form into matrix."""
    a_cpp = _create_cpp_form(a)
    A = cpp.fem.create_matrix(a_cpp)
    assemble(A, a_cpp, bcs, diagonal)
    return A


def apply_lifting(b: cpp.la.PETScVector,
                  a: typing.List[typing.Union[Form, cpp.fem.Form]],
                  bcs: typing.List[DirichletBC],
                  x0: typing.Optional[typing.List[cpp.la.PETScVector]] = [],
                  scale: float = 1.0) -> None:
    """Modify vector for lifting of boundary conditions."""
    a_cpp = [_create_cpp_form(form) for form in a]
    _x0 = [v.vec() if v is not None else None for v in x0]
    cpp.fem.apply_lifting(b.vec(), a_cpp, bcs, _x0, scale)


def set_bc(b: cpp.la.PETScVector,
           bcs: typing.List[DirichletBC],
           x0: typing.Optional[cpp.la.PETScVector] = None,
           scale: float = 1.0) -> None:
    """Insert boundary condition values into vector"""
    _x0 = x0.vec() if x0 is not None else None
    cpp.fem.set_bc(b.vec(), bcs, _x0, scale)
