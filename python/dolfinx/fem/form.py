# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

import cffi
import numpy as np
import ufl
from dolfinx import cpp, jit
from dolfinx.fem import function


class Form:
    def __init__(self, form: ufl.Form, form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
        """Create DOLFINx Form

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
        This wrapper for UFL form is responsible for the actual FFCx compilation
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
        self._ufc_form, module, self._code = jit.ffcx_jit(
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
        coeffs = [original_coefficients[self._ufc_form.original_coefficient_position[
            i]]._cpp_object for i in range(self._ufc_form.num_coefficients)]

        # Create dictionary of of subdomain markers (possible None for
        # some dimensions
        subdomains = {cpp.fem.IntegralType.cell: self._subdomains.get("cell"),
                      cpp.fem.IntegralType.exterior_facet: self._subdomains.get("exterior_facet"),
                      cpp.fem.IntegralType.interior_facet: self._subdomains.get("interior_facet"),
                      cpp.fem.IntegralType.vertex: self._subdomains.get("vertex")}

        # Prepare dolfinx.cpp.fem.Form and hold it as a member
        ffi = cffi.FFI()
        self._cpp_object = cpp.fem.create_form(ffi.cast("uintptr_t", ffi.addressof(self._ufc_form)),
                                               function_spaces, coeffs,
                                               [c._cpp_object for c in form.constants()], subdomains, mesh)

    @property
    def rank(self):
        """Return the compiled ufc_form object"""
        return self._cpp_object.rank

    @property
    def function_spaces(self):
        """Return the compiled ufc_form object"""
        return self._cpp_object.function_spaces

    @property
    def ufc_form(self):
        """Return the compiled ufc_form object"""
        return self._ufc_form

    @property
    def code(self):
        """Return C code strings"""
        return self._code


_args = typing.Union[typing.Iterable[Form], typing.Iterable[typing.Iterable[Form]]]
_ret = typing.Union[typing.Iterable[function.FunctionSpace], typing.Iterable[typing.Iterable[function.FunctionSpace]]]


def extract_function_spaces(forms: _args) -> _ret:
    """Extract common function spaces from an array of forms. If `forms`
    is a list of linears form, this function returns of list of the the
    corresponding test functions. If `forms` is a 2D array of bilinear
    forms, this function returns a pair of arrays where the first array
    holds the common test function space for each row and the second
    array holds the common trial function space for each column."""
    _forms = np.array(forms)
    if _forms.ndim == 0:
        raise RuntimeError("Expected an array for forms, not a single form")
    elif _forms.ndim == 1:
        for form in _forms:
            if form is not None:
                assert form.rank == 1, "Expected linear form"
        return [form.function_spaces[0] if form is not None else None for form in forms]
    elif _forms.ndim == 2:
        extract_spaces = np.vectorize(lambda form, index: form.function_spaces[index] if form is not None else None)
        V0, V1 = extract_spaces(_forms, 0), extract_spaces(_forms, 1)

        def unique_spaces(V):
            V0 = V[:, 0]
            for col in range(1, V.shape[1]):
                for row in range(V.shape[1]):
                    if V0[row] is None and V[row, col] is not None:
                        V0[row] = V[row, col]
                    elif V0[row] is not None and V[row, col] is not None:
                        assert V0[row] == V[row, col], "Cannot extract unique function spaces"
            return V0
        return list(unique_spaces(V0)), list(unique_spaces(V1.transpose()))
    else:
        raise RuntimeError("Unsupported array of forms")
