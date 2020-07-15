# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import cffi

import ufl
from dolfinx import cpp, jit


class Form:
    def __init__(self, form: ufl.Form, form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
        """Create dolfinx Form

        Parameters
        ----------
        form
            Pure UFL form
        form_compiler_parameters
            Parameters used in FFCX compilation of this form. Run `ffcx --help` in the commandline
            to see all available options.
        jit_parameters
            Parameters controlling JIT compilation of C code.

        Note
        ----
        This wrapper for UFL form is responsible for the actual FFCX compilation
        and attaching coefficients and domains specific data to the underlying
        C++ Form.
        """

        # Extract subdomain data from UFL form
        sd = form.subdomain_data()
        self._subdomains, = list(sd.values())  # Assuming single domain
        domain, = list(sd.keys())  # Assuming single domain
        mesh = domain.ufl_cargo()

        # Compile UFL form with JIT
        ufc_form = jit.ffcx_jit(
            form,
            form_compiler_parameters=form_compiler_parameters,
            jit_parameters=jit_parameters,
            mpi_comm=mesh.mpi_comm())

        # For every argument in form extract its function space
        function_spaces = [
            func.ufl_function_space()._cpp_object for func in form.arguments()
        ]

        # Prepare dolfinx.cpp.fem.Form and hold it as a member
        ffi = cffi.FFI()
        self._cpp_object = cpp.fem.create_form(ffi.cast("uintptr_t", ufc_form), function_spaces)

        # Need to fill the form with coefficients data
        # For every coefficient in form take its C++ object
        original_coefficients = form.coefficients()
        for i in range(self._cpp_object.num_coefficients()):
            j = self._cpp_object.original_coefficient_position(i)
            self._cpp_object.set_coefficient(i, original_coefficients[j]._cpp_object)

        # Constants are set based on their position in original form
        original_constants = [c._cpp_object for c in form.constants()]

        self._cpp_object.set_constants(original_constants)

        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")

        # Attach mesh (because function spaces and coefficients may be
        # empty lists)
        if not function_spaces:
            self._cpp_object.set_mesh(mesh)

        # Attach subdomains to C++ Form if we have them
        subdomains = self._subdomains.get("cell")
        if subdomains:
            self._cpp_object.integrals.set_domains(cpp.fem.IntegralType.cell, subdomains)

        subdomains = self._subdomains.get("exterior_facet")
        if subdomains:
            self._cpp_object.integrals.set_domains(cpp.fem.IntegralType.exterior_facet, subdomains)

        subdomains = self._subdomains.get("interior_facet")
        if subdomains:
            self._cpp_object.integrals.set_domains(cpp.fem.IntegralType.interior_facet, subdomains)

        subdomains = self._subdomains.get("vertex")
        if subdomains:
            self._cpp_object.integrals.set_domains(cpp.fem.IntegralType.vertex, subdomains)
