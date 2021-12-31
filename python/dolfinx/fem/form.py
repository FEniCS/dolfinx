# -*- coding: utf-8 -*-
# Copyright (C) 2017-2021 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

import cffi
import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx import jit
from dolfinx.fem import function

from petsc4py import PETSc


class Form:
    def __init__(self, form: ufl.Form, dtype: np.dtype = PETSc.ScalarType,
                 form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
        """Create DOLFINx Form

        Parameters
        ----------
        form
            Pure UFL form
        dtype
            The scalar type to use for the compiled form
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
        if dtype == np.float32:
            form_compiler_parameters["scalar_type"] = "float"
        elif dtype == np.float64:
            form_compiler_parameters["scalar_type"] = "double"
        elif dtype == np.complex128:
            form_compiler_parameters["scalar_type"] = "double _Complex"
        else:
            raise RuntimeError(f"Unsupported scalar type {dtype} for Form.")

        self._ufc_form, module, self._code = jit.ffcx_jit(
            mesh.comm, form,
            form_compiler_parameters=form_compiler_parameters,
            jit_parameters=jit_parameters)

        # For every argument in form extract its function space
        function_spaces = [func.ufl_function_space()._cpp_object for func in form.arguments()]

        # Prepare coefficients data. For every coefficient in form take
        # its C++ object.
        original_coefficients = form.coefficients()
        coeffs = [original_coefficients[self._ufc_form.original_coefficient_position[
            i]]._cpp_object for i in range(self._ufc_form.num_coefficients)]

        # Create dictionary of of subdomain markers (possible None for
        # some dimensions
        subdomains = {_cpp.fem.IntegralType.cell: self._subdomains.get("cell"),
                      _cpp.fem.IntegralType.exterior_facet: self._subdomains.get("exterior_facet"),
                      _cpp.fem.IntegralType.interior_facet: self._subdomains.get("interior_facet"),
                      _cpp.fem.IntegralType.vertex: self._subdomains.get("vertex")}

        # Prepare dolfinx.cpp.fem.Form and hold it as a member
        def create_form(dtype):
            if dtype == np.float32:
                return _cpp.fem.create_form_float32
            elif dtype == np.float64:
                return _cpp.fem.create_form_float64
            elif dtype == np.complex128:
                return _cpp.fem.create_form_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        ffi = cffi.FFI()
        self._cpp_object = create_form(dtype)(ffi.cast("uintptr_t", ffi.addressof(self._ufc_form)),
                                              function_spaces, coeffs,
                                              [c._cpp_object for c in form.constants()], subdomains,
                                              mesh)

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


def extract_function_spaces(forms: _args, index: int = 0) -> typing.Iterable[function.FunctionSpace]:
    """Extract common function spaces from an array of forms. If `forms`
    is a list of linear form, this function returns of list of the the
    corresponding test functions. If `forms` is a 2D array of bilinear
    forms, for index=0 the list common test function space for each row
    is returned, and if index=1 the common trial function spaces for
    each column are returned."""
    _forms = np.array(forms)
    if _forms.ndim == 0:
        raise RuntimeError("Expected an array for forms, not a single form")
    elif _forms.ndim == 1:
        assert index == 0
        for form in _forms:
            if form is not None:
                assert form.rank == 1, "Expected linear form"
        return [form.function_spaces[0] if form is not None else None for form in forms]
    elif _forms.ndim == 2:
        assert index == 0 or index == 1
        extract_spaces = np.vectorize(lambda form: form.function_spaces[index] if form is not None else None)
        V = extract_spaces(_forms)

        def unique_spaces(V):
            # Pick spaces from first column
            V0 = V[:, 0]

            # Iterate over each column
            for col in range(1, V.shape[1]):
                # Iterate over entry in column, updating if current
                # space is None, or where both spaces are not None check
                # that they are the same
                for row in range(V.shape[0]):
                    if V0[row] is None and V[row, col] is not None:
                        V0[row] = V[row, col]
                    elif V0[row] is not None and V[row, col] is not None:
                        assert V0[row] is V[row, col], "Cannot extract unique function spaces"
            return V0

        if index == 0:
            return list(unique_spaces(V))
        elif index == 1:
            return list(unique_spaces(V.transpose()))
    else:
        raise RuntimeError("Unsupported array of forms")
