# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import singledispatch
import typing

import dolfin.fem.dirichletbc
import ufl
from dolfin import cpp
from dolfin.fem.assembling import _create_cpp_form
from dolfin.jit.jit import ffc_jit


@singledispatch
def assemble(a) -> typing.Union[float, cpp.la.PETScMatrix, cpp.la.PETScVector]:
    """Assemble a form over mesh"""
    a_cpp = _create_cpp_form(a)
    return cpp.fem.assemble(a_cpp)


@assemble.register(cpp.la.PETScVector)
def _(b: cpp.la.PETScVector, L, a=[], bcs=[], scale=1.0: float) -> None:
    """Assemble linear form into vector"""
    L_cpp = _create_cpp_form(L)
    a_cpp = [_create_cpp_form(form) for form in a]
    cpp.fem.assemble(L_cpp, b, a_cpp, bcs, scale)


def set_bc(b: cpp.la.PETScVector, L, bcs: typing.List[dolfin.fem.dirichletbc.DirichletBC]) -> None:
    """Insert boundary condition values into vector"""
    cpp.fem.set_bc(b, L, bcs)


class Assembler:
    """Assemble variational forms"""
    def __init__(self, a, L, bcs=None, form_compiler_parameters=None):
        self.a = a
        self.L = L
        if bcs is None:
            self.bcs = []
        else:
            self.bcs = bcs
        self.assembler = None
        self.form_compiler_parameters = form_compiler_parameters

    def assemble(self, x: typing.Union[cpp.la.PETScMatrix, cpp.la.PETScVector]
                 ) -> typing.Union[cpp.la.PETScMatrix, cpp.la.PETScVector]:
        """Assemble a form into linear alebra object. The linear algebra
        object must already be initialised.

        """
        self._compile_forms()
        self.assembler.assemble(x)
        return x

    def assemble_matrix(self, mat_type=cpp.fem.Assembler.BlockType.monolithic
                        ) -> cpp.la.PETScMatrix:
        """Return assembled matrix from bilinear form"""
        self._compile_forms()
        return self.assembler.assemble_matrix(mat_type)

    def assemble_vector(self, mat_type=cpp.fem.Assembler.BlockType.monolithic
                        ) -> cpp.la.PETScVector:
        """Create assembled vector from linear form"""
        self._compile_forms()
        return self.assembler.assemble_vector(mat_type)

    # FIXME: simplify this function
    def _compile_forms(self):
        if self.assembler is None:
            try:
                a_forms = [[
                    _create_cpp_form(a, self.form_compiler_parameters)
                    for a in row
                ] for row in self.a]
            except TypeError:
                a_forms = [[
                    _create_cpp_form(self.a, self.form_compiler_parameters)
                ]]
            try:
                L_forms = [
                    _create_cpp_form(L, self.form_compiler_parameters)
                    for L in self.L
                ]
            except TypeError:
                L_forms = [
                    _create_cpp_form(self.L, self.form_compiler_parameters)
                ]

            self.assembler = cpp.fem.Assembler(a_forms, L_forms, self.bcs)


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
