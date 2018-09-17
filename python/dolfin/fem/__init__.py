# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

from dolfin import cpp
import dolfin.cpp
import ufl
from dolfin.fem.assembling import _create_cpp_form
from dolfin.jit.jit import ffc_jit


def assemble(a):
    """Assemble form mesh and return scalar value"""
    a_cpp = _create_cpp_form(a)
    if a_cpp.rank() == 0:
        return cpp.fem.assemble_scalar(a_cpp)
    elif a_cpp.rank() == 0:
        return cpp.fem.assemble_vector(a_cpp)
    else:
        raise RuntimeError("Assembly of Form of rank {} not supported".format(
            a_cpp.rank()))


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
