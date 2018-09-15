# -*- coding: utf-8 -*-
# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assemble variational forms into linear algebra objects.

"""

import dolfin.cpp as cpp
import ufl
from dolfin.fem.assembling import _create_dolfin_form
from dolfin.fem.form import Form


class Assembler:
    def __init__(self, a, L, bcs=None, form_compiler_parameters=None):

        self.a = a
        self.L = L
        if bcs is None:
            self.bcs = []
        else:
            self.bcs = bcs
        self.assembler = None
        self.form_compiler_parameters = form_compiler_parameters

    def assemble(self, x):
        self._compile_forms()
        self.assembler.assemble(x)
        return x

    def assemble_matrix(self, mat_type=cpp.fem.Assembler.BlockType.monolithic):
        self._compile_forms()
        return self.assembler.assemble_matrix(mat_type)

    def assemble_vector(self, mat_type=cpp.fem.Assembler.BlockType.monolithic):
        self._compile_forms()
        return self.assembler.assemble_vector(mat_type)

    # FIXME: simplify this function
    def _compile_forms(self):
        if self.assembler is None:
            try:
                a_forms = [[
                    _create_dolfin_form(a, self.form_compiler_parameters)
                    for a in row
                ] for row in self.a]
            except TypeError:
                a_forms = [[
                    _create_dolfin_form(self.a, self.form_compiler_parameters)
                ]]
            try:
                L_forms = [
                    _create_dolfin_form(L, self.form_compiler_parameters)
                    for L in self.L
                ]
            except TypeError:
                L_forms = [
                    _create_dolfin_form(self.L, self.form_compiler_parameters)
                ]

            # Create assembler
            self.assembler = cpp.fem.Assembler(a_forms, L_forms, self.bcs)
