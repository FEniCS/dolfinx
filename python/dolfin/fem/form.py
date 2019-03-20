# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl

from dolfin import cpp, fem, jit
import cffi


class Form(ufl.Form):
    def __init__(self, form: ufl.Form, form_compiler_parameters: dict = None):
        """Create dolfin Form

        Parameters
        ----------
        form
            Pure UFL form
        form_compiler_parameters
            Parameters used in JIT FFC compilation of this form

        Note
        ----
        This wrapper for UFL form is responsible for the actual FFC compilation
        and attaching coefficients and domains specific data to the underlying
        C++ Form.
        """
        self.form_compiler_parameters = form_compiler_parameters

        # Extract subdomain data from UFL form
        sd = form.subdomain_data()
        self._subdomains, = list(sd.values())  # Assuming single domain
        domain, = list(sd.keys())  # Assuming single domain
        mesh = domain.ufl_cargo()

        # Compile UFL form with JIT
        ufc_form = jit.ffc_jit(
            form,
            form_compiler_parameters=self.form_compiler_parameters,
            mpi_comm=mesh.mpi_comm())

        # Cast compiled library to pointer to ufc_form
        ffi = cffi.FFI()
        ufc_form = fem.dofmap.make_ufc_form(ffi.cast("uintptr_t", ufc_form))

        # For every argument in form extract its function space
        function_spaces = [
            func.function_space()._cpp_object for func in form.arguments()
        ]

        # Prepare dolfin.Form and hold it as a member
        self._cpp_object = cpp.fem.Form(ufc_form, function_spaces)

        # Need to fill the form with coefficients data
        # For every coefficient in form take its CPP object
        original_coefficients = form.coefficients()
        for i in range(self._cpp_object.num_coefficients()):
            j = self._cpp_object.original_coefficient_position(i)
            self._cpp_object.set_coefficient(
                j, original_coefficients[i]._cpp_object)

        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")

        # Attach mesh (because function spaces and coefficients may be
        # empty lists)
        if not function_spaces:
            self._cpp_object.set_mesh(mesh)

        # Attach subdomains to C++ Form if we have them
        subdomains = self._subdomains.get("cell")
        self._cpp_object.set_cell_domains(subdomains)

        subdomains = self._subdomains.get("exterior_facet")
        self._cpp_object.set_exterior_facet_domains(subdomains)

        subdomains = self._subdomains.get("interior_facet")
        self._cpp_object.set_interior_facet_domains(subdomains)

        subdomains = self._subdomains.get("vertex")
        self._cpp_object.set_vertex_domains(subdomains)
