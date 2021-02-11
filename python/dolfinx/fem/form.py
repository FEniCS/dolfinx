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
            See :py:func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_parameters
            See :py:func:`ffcx_jit <dolfinx.jit.ffcx_jit>`

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
        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")

        # Compile UFL form with JIT
        ufc_form = jit.ffcx_jit(
            mesh.mpi_comm(),
            form,
            form_compiler_parameters=form_compiler_parameters,
            jit_parameters=jit_parameters)

        # For every argument in form extract its function space
        function_spaces = [
            func.ufl_function_space()._cpp_object for func in form.arguments()
        ]

        # Prepare coefficients data. For every coefficient in form take
        # its C++ object.
        original_coefficients = form.coefficients()
        coeffs = [original_coefficients[ufc_form.original_coefficient_position(
            i)]._cpp_object for i in range(ufc_form.num_coefficients)]

        # Create dictionary of of subdomain markers (possible None for
        # some dimensions
        subdomains = {cpp.fem.IntegralType.cell: self._subdomains.get("cell"),
                      cpp.fem.IntegralType.exterior_facet: self._subdomains.get("exterior_facet"),
                      cpp.fem.IntegralType.interior_facet: self._subdomains.get("interior_facet"),
                      cpp.fem.IntegralType.vertex: self._subdomains.get("vertex")}

        # Prepare dolfinx.cpp.fem.Form and hold it as a member
        ffi = cffi.FFI()
        self._cpp_object = cpp.fem.create_form(ffi.cast("uintptr_t", ufc_form),
                                               function_spaces, coeffs,
                                               [c._cpp_object for c in form.constants()], subdomains, mesh)
