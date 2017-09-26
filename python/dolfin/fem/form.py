# -*- coding: utf-8 -*-
"""FIXME: Add description"""

# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# Distributed under the terms of the GNU Lesser Public License (LGPL),
# either version 3 of the License, or (at your option) any later
# version.

import ufl
import dolfin.cpp as cpp
from dolfin.jit.jit import ffc_jit

class Form(cpp.fem.Form):
    def __init__(self, form, **kwargs):

        # Check form argument
        if not isinstance(form, ufl.Form):
            raise RuntimeError("Expected a ufl.Form.")

        sd = form.subdomain_data()
        self.subdomains, = list(sd.values())  # Assuming single domain
        domain, = list(sd.keys())  # Assuming single domain
        mesh = domain.ufl_cargo()

        # Having a mesh in the form is a requirement
        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")

        form_compiler_parameters = kwargs.pop("form_compiler_parameters", None)

        # Add DOLFIN include paths (just the Boost path for special
        # math functions is really required)
        # FIXME: move getting include paths to elsewhere
        import pkgconfig
        d = pkgconfig.parse('dolfin')
        if form_compiler_parameters is None:
            form_compiler_parameters = {"external_include_dirs": d["include_dirs"]}
        else:
            # FIXME: add paths if dict entry already exists
            form_compiler_parameters["external_include_dirs"] = d["include_dirs"]

        ufc_form = ffc_jit(form, form_compiler_parameters=form_compiler_parameters,
                                 mpi_comm=mesh.mpi_comm())
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])

        function_spaces = [func.function_space()._cpp_object for func in form.arguments()]

        cpp.fem.Form.__init__(self, ufc_form, function_spaces)

        original_coefficients = form.coefficients()
        self.coefficients = []
        for i in range(self.num_coefficients()):
            j = self.original_coefficient_position(i)
            self.coefficients.append(original_coefficients[j].cpp_object())

        # Type checking coefficients
        if not all(isinstance(c, (cpp.function.GenericFunction))
                   for c in self.coefficients):
            coefficient_error = "Error while extracting coefficients. "
            raise TypeError(coefficient_error +
                            "Either provide a dict of cpp.function.GenericFunctions, " +
                            "or use Function to define your form.")

        for i in range(self.num_coefficients()):
            if isinstance(self.coefficients[i], cpp.function.GenericFunction):
                self.set_coefficient(i, self.coefficients[i])

        # Attach mesh (because function spaces and coefficients may be
        # empty lists)
        if not function_spaces:
            self.set_mesh(mesh)

        # Attach subdomains to C++ Form if we have them
        subdomains = self.subdomains.get("cell")
        if subdomains is not None:
            self.set_cell_domains(subdomains)
        subdomains = self.subdomains.get("exterior_facet")
        if subdomains is not None:
            self.set_exterior_facet_domains(subdomains)
        subdomains = self.subdomains.get("interior_facet")
        if subdomains is not None:
            self.set_interior_facet_domains(subdomains)
        subdomains = self.subdomains.get("vertex")
        if subdomains is not None:
            self.set_vertex_domains(subdomains)
