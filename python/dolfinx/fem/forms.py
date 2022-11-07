# Copyright (C) 2017-2021 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

import collections
import collections.abc
import typing

from dolfinx.fem.function import FunctionSpace

if typing.TYPE_CHECKING:
    from dolfinx.fem import function
    from dolfinx.mesh import Mesh

import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx import jit

from petsc4py import PETSc


class FormMetaClass:
    def __init__(self, form, V: list[_cpp.fem.FunctionSpace], coeffs, constants,
                 subdomains: dict[_cpp.mesh.MeshTags_int32, typing.Union[None, typing.Any]], mesh: _cpp.mesh.Mesh,
                 ffi, code):
        """A finite element form

        Notes:
            Forms should normally be constructed using
            :func:`forms.form` and not using this class initialiser.
            This class is combined with different base classes that
            depend on the scalar type used in the Form.

        Args:
            form: Compiled UFC form
            V: The argument function spaces
            coeffs: Finite element coefficients that appear in the form
            constants: Constants appearing in the form
            subdomains: Subdomains for integrals
            mesh: The mesh that the form is defined on

        """
        self._code = code
        self._ufcx_form = form
        super().__init__(ffi.cast("uintptr_t", ffi.addressof(self._ufcx_form)),
                         V, coeffs, constants, subdomains, mesh)  # type: ignore

    @property
    def ufcx_form(self):
        """The compiled ufcx_form object"""
        return self._ufcx_form

    @property
    def code(self) -> str:
        """C code strings"""
        return self._code

    @property
    def function_spaces(self) -> typing.List[FunctionSpace]:
        """Function spaces on which this form is defined"""
        return super().function_spaces  # type: ignore

    @property
    def dtype(self) -> np.dtype:
        """dtype of this form"""
        return super().dtype  # type: ignore

    @property
    def mesh(self) -> Mesh:
        """Mesh on which this form is defined"""
        return super().mesh  # type: ignore

    @property
    def integral_types(self):
        """Integral types in the form"""
        return super().integral_types  # type: ignore


form_types = typing.Union[FormMetaClass, _cpp.fem.Form_float32, _cpp.fem.Form_float64,
                          _cpp.fem.Form_complex64, _cpp.fem.Form_complex128]


def form(form: typing.Union[ufl.Form, typing.Iterable[ufl.Form]], dtype: np.dtype = PETSc.ScalarType,
         form_compiler_options: dict = {}, jit_options: dict = {}):
    """Create a DOLFINx Form or an array of Forms

    Args:
        form: A UFL form or list(s) of UFL forms
        dtype: Scalar type to use for the compiled form
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options:See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`

    Returns:
        Compiled finite element Form

    Notes:
        This function is responsible for the compilation of a UFL form
        (using FFCx) and attaching coefficients and domains specific
        data to the underlying C++ form. It dynamically create a
        :class:`Form` instance with an appropriate base class for the
        scalar type, e.g. `_cpp.fem.Form_float64`.


    """
    if dtype == np.float32:
        ftype = _cpp.fem.Form_float32
        form_compiler_options["scalar_type"] = "float"
    elif dtype == np.float64:
        ftype = _cpp.fem.Form_float64
        form_compiler_options["scalar_type"] = "double"
    elif dtype == np.complex64:
        ftype = _cpp.fem.Form_complex64
        form_compiler_options["scalar_type"] = "float _Complex"
    elif dtype == np.complex128:
        ftype = _cpp.fem.Form_complex128
        form_compiler_options["scalar_type"] = "double _Complex"
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    formcls = type("Form", (FormMetaClass, ftype), {})

    def _form(form):
        """"Compile a single UFL form"""
        # Extract subdomain data from UFL form
        sd = form.subdomain_data()
        domain, = list(sd.keys())  # Assuming single domain
        # Get subdomain data for each integral type
        subdomains = {}
        for integral_type, data in sd.get(domain).items():
            # Check that the subdomain data for each integral of this type is
            # the same
            assert all([id(d) == id(data[0]) for d in data])
            subdomains[integral_type] = data[0]

        mesh = domain.ufl_cargo()
        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")

        ufcx_form, module, code = jit.ffcx_jit(mesh.comm, form,
                                               form_compiler_options=form_compiler_options,
                                               jit_options=jit_options)

        # For each argument in form extract its function space
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]

        # Prepare coefficients data. For every coefficient in form take its
        # C++ object.
        original_coefficients = form.coefficients()
        coeffs = [original_coefficients[ufcx_form.original_coefficient_position[i]
                                        ]._cpp_object for i in range(ufcx_form.num_coefficients)]
        constants = [c._cpp_object for c in form.constants()]

        # Subdomain markers (possibly None for some dimensions)
        subdomains = {_cpp.fem.IntegralType.cell: subdomains.get("cell"),
                      _cpp.fem.IntegralType.exterior_facet: subdomains.get("exterior_facet"),
                      _cpp.fem.IntegralType.interior_facet: subdomains.get("interior_facet"),
                      _cpp.fem.IntegralType.vertex: subdomains.get("vertex")}

        return formcls(ufcx_form, V, coeffs, constants, subdomains, mesh, module.ffi, code)

    def _create_form(form):
        """Recursively convert ufl.Forms to dolfinx.fem.Form, otherwise
        return form argument"""
        if isinstance(form, ufl.Form):
            return _form(form)
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _create_form(sub_form), form))
        return form

    return _create_form(form)


def extract_function_spaces(forms: typing.Union[typing.Iterable[FormMetaClass],  # type: ignore [return]
                                                typing.Iterable[typing.Iterable[FormMetaClass]]],
                            index: int = 0) -> typing.Iterable[typing.Union[None, function.FunctionSpace]]:
    """Extract common function spaces from an array of forms. If `forms`
    is a list of linear form, this function returns of list of the
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
        return [form.function_spaces[0] if form is not None else None for form in forms]  # type: ignore[union-attr]
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
