# Copyright (C) 2017-2024 Chris N. Richardson, Garth N. Wells, Michal Habera and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

import collections
import typing
from dataclasses import dataclass
from itertools import chain

import numpy as np
import numpy.typing as npt

import ffcx
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_scalar_type, jit
from dolfinx.fem import IntegralType
from dolfinx.fem.function import FunctionSpace

if typing.TYPE_CHECKING:
    from mpi4py import MPI

    from dolfinx.fem import function
    from dolfinx.mesh import Mesh, MeshTags


class Form:
    _cpp_object: typing.Union[
        _cpp.fem.Form_complex64,
        _cpp.fem.Form_complex128,
        _cpp.fem.Form_float32,
        _cpp.fem.Form_float64,
    ]
    _code: typing.Optional[str]

    def __init__(
        self,
        form: typing.Union[
            _cpp.fem.Form_complex64,
            _cpp.fem.Form_complex128,
            _cpp.fem.Form_float32,
            _cpp.fem.Form_float64,
        ],
        ufcx_form=None,
        code: typing.Optional[str] = None,
    ):
        """A finite element form

        Note:
            Forms should normally be constructed using :func:`form` and
            not using this class initialiser. This class is combined
            with different base classes that depend on the scalar type
            used in the Form.

        Args:
            form: Compiled form object.
            ufcx_form: UFCx form
            code: Form C++ code
        """
        self._code = code
        self._ufcx_form = ufcx_form
        self._cpp_object = form

    @property
    def ufcx_form(self):
        """The compiled ufcx_form object"""
        return self._ufcx_form

    @property
    def code(self) -> typing.Union[str, None]:
        """C code strings"""
        return self._code

    @property
    def rank(self) -> int:
        return self._cpp_object.rank  # type: ignore

    @property
    def function_spaces(self) -> list[FunctionSpace]:
        """Function spaces on which this form is defined"""
        return self._cpp_object.function_spaces  # type: ignore

    @property
    def dtype(self) -> np.dtype:
        """Scalar type of this form"""
        return np.dtype(self._cpp_object.dtype)

    @property
    def mesh(self) -> typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64]:
        """Mesh on which this form is defined."""
        return self._cpp_object.mesh

    @property
    def integral_types(self):
        """Integral types in the form"""
        return self._cpp_object.integral_types


def get_integration_domains(
    integral_type: IntegralType,
    subdomain: typing.Optional[typing.Union[MeshTags, list[tuple[int, npt.NDArray[np.int32]]]]],
    subdomain_ids: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Get integration domains from subdomain data.

    The subdomain data is a meshtags object consisting of markers, or a None object.
    If it is a None object we do not pack any integration entities.
    Integration domains are defined as a list of tuples, where each input `subdomain_ids`
    is mapped to an array of integration entities, where an integration entity for a cell
    integral is the list of cells.
    For an exterior facet integral each integration entity is a
    tuple (cell_index, local_facet_index).
    For an interior facet integral each integration entity is a
    uple (cell_index0, local_facet_index0,
    cell_index1, local_facet_index1). Where the first cell-facet pair is
    the '+' restriction, the second the '-' restriction.

    Args:
        integral_type: The type of integral to pack integration entitites for
        subdomain: A meshtag with markers or manually specified integration domains.
        subdomain_ids: List of ids to integrate over
    """
    if subdomain is None:
        return []
    else:
        domains = []
        try:
            if integral_type in (IntegralType.exterior_facet, IntegralType.interior_facet):
                tdim = subdomain.topology.dim  # type: ignore
                subdomain._cpp_object.topology.create_connectivity(tdim - 1, tdim)  # type: ignore
                subdomain._cpp_object.topology.create_connectivity(tdim, tdim - 1)  # type: ignore
            # Compute integration domains only for each subdomain id in the integrals
            # If a process has no integral entities, insert an empty array
            for id in subdomain_ids:
                integration_entities = _cpp.fem.compute_integration_domains(
                    integral_type,
                    subdomain._cpp_object.topology,  # type: ignore
                    subdomain.find(id),  # type: ignore
                    subdomain.dim,  # type: ignore
                )
                domains.append((id, integration_entities))
            return [(s[0], np.array(s[1])) for s in domains]
        except AttributeError:
            return [(s[0], np.array(s[1])) for s in subdomain]  # type: ignore


def form_cpp_class(
    dtype: npt.DTypeLike,
) -> typing.Union[
    _cpp.fem.Form_float32, _cpp.fem.Form_float64, _cpp.fem.Form_complex64, _cpp.fem.Form_complex128
]:
    """Return the wrapped C++ class of a variational form of a specific scalar type.

    Args:
        dtype: Scalar type of the required form class.

    Returns:
        Wrapped C++ form class of the requested type.

    Note:
        This function is for advanced usage, typically when writing
        custom kernels using Numba or C.
    """
    if np.issubdtype(dtype, np.float32):
        return _cpp.fem.Form_float32
    elif np.issubdtype(dtype, np.float64):
        return _cpp.fem.Form_float64
    elif np.issubdtype(dtype, np.complex64):
        return _cpp.fem.Form_complex64
    elif np.issubdtype(dtype, np.complex128):
        return _cpp.fem.Form_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")


_ufl_to_dolfinx_domain = {
    "cell": IntegralType.cell,
    "exterior_facet": IntegralType.exterior_facet,
    "interior_facet": IntegralType.interior_facet,
    "vertex": IntegralType.vertex,
}


def form(
    form: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    dtype: npt.DTypeLike = default_scalar_type,
    form_compiler_options: typing.Optional[dict] = None,
    jit_options: typing.Optional[dict] = None,
    entity_maps: typing.Optional[dict[Mesh, np.typing.NDArray[np.int32]]] = None,
):
    """Create a Form or an array of Forms.

    Args:
        form: A UFL form or list(s) of UFL forms.
        dtype: Scalar type to use for the compiled form.
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
        entity_maps: If any trial functions, test functions, or
            coefficients in the form are not defined over the same mesh
            as the integration domain, `entity_maps` must be supplied.
            For each key (a mesh, different to the integration domain
            mesh) a map should be provided relating the entities in the
            integration domain mesh to the entities in the key mesh e.g.
            for a key-value pair (msh, emap) in `entity_maps`, `emap[i]`
            is the entity in `msh` corresponding to entity `i` in the
            integration domain mesh.

    Returns:
        Compiled finite element Form.

    Note:
        This function is responsible for the compilation of a UFL form
        (using FFCx) and attaching coefficients and domains specific
        data to the underlying C++ form. It dynamically create a
        :class:`Form` instance with an appropriate base class for the
        scalar type, e.g. :func:`_cpp.fem.Form_float64`.
    """
    if form_compiler_options is None:
        form_compiler_options = dict()

    form_compiler_options["scalar_type"] = dtype
    ftype = form_cpp_class(dtype)

    def _form(form):
        """Compile a single UFL form"""
        # Extract subdomain data from UFL form
        sd = form.subdomain_data()
        (domain,) = list(sd.keys())  # Assuming single domain
        # Check that subdomain data for each integral type is the same
        for data in sd.get(domain).values():
            assert all([d is data[0] for d in data])

        mesh = domain.ufl_cargo()
        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")
        ufcx_form, module, code = jit.ffcx_jit(
            mesh.comm, form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        # For each argument in form extract its function space
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]

        # Prepare coefficients data. For every coefficient in form take
        # its C++ object.
        original_coeffs = form.coefficients()
        coeffs = [
            original_coeffs[ufcx_form.original_coefficient_positions[i]]._cpp_object
            for i in range(ufcx_form.num_coefficients)
        ]
        constants = [c._cpp_object for c in form.constants()]

        # Make map from integral_type to subdomain id
        subdomain_ids = {type: [] for type in sd.get(domain).keys()}
        for integral in form.integrals():
            if integral.subdomain_data() is not None:
                # Subdomain ids can be strings, its or tuples with strings and ints
                if integral.subdomain_id() != "everywhere":
                    try:
                        ids = [sid for sid in integral.subdomain_id() if sid != "everywhere"]
                    except TypeError:
                        # If not tuple, but single integer id
                        ids = [integral.subdomain_id()]
                else:
                    ids = []
                subdomain_ids[integral.integral_type()].append(ids)

        # Chain and sort subdomain ids
        for itg_type, marker_ids in subdomain_ids.items():
            flattened_ids = list(chain.from_iterable(marker_ids))
            flattened_ids.sort()
            subdomain_ids[itg_type] = flattened_ids

        # Subdomain markers (possibly empty list for some integral types)
        subdomains = {
            _ufl_to_dolfinx_domain[key]: get_integration_domains(
                _ufl_to_dolfinx_domain[key], subdomain_data[0], subdomain_ids[key]
            )
            for (key, subdomain_data) in sd.get(domain).items()
        }

        if entity_maps is None:
            _entity_maps = dict()
        else:
            _entity_maps = {msh._cpp_object: emap for (msh, emap) in entity_maps.items()}

        f = ftype(
            module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
            V,
            coeffs,
            constants,
            subdomains,
            _entity_maps,
            mesh,
        )
        return Form(f, ufcx_form, code)

    def _create_form(form):
        """Recursively convert ufl.Forms to dolfinx.fem.Form, otherwise
        return form argument"""
        if isinstance(form, ufl.Form):
            return _form(form)
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _create_form(sub_form), form))
        return form

    return _create_form(form)


def extract_function_spaces(
    forms: typing.Union[
        typing.Iterable[Form],  # type: ignore [return]
        typing.Iterable[typing.Iterable[Form]],
    ],
    index: int = 0,
) -> typing.Iterable[typing.Union[None, function.FunctionSpace]]:
    """Extract common function spaces from an array of forms.

    If `forms` is a list of linear form, this function returns of list
    of the corresponding test functions. If `forms` is a 2D array of
    bilinear forms, for index=0 the list common test function space for
    each row is returned, and if index=1 the common trial function
    spaces for each column are returned.
    """
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
        extract_spaces = np.vectorize(
            lambda form: form.function_spaces[index] if form is not None else None
        )
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

    raise RuntimeError("Unsupported array of forms")


@dataclass
class CompiledForm:
    """
    Compiled UFL form without associated DOLFINx data

    """

    ufl_form: ufl.Form  # The original ufl form
    ufcx_form: typing.Any  # The compiled form
    module: typing.Any  #  The module
    code: str  # The source code
    dtype: npt.DTypeLike  # data type used for the `ufcx_form`


def compile_form(
    comm: MPI.Intracomm,
    form: ufl.Form,
    form_compiler_options: typing.Optional[dict] = {"scalar_type": default_scalar_type},
    jit_options: typing.Optional[dict] = None,
) -> CompiledForm:
    """Compile UFL form without associated DOLFINx data

    Args:
        comm: The MPI communicator used when compiling the form
        form: The UFL form to compile
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
    """
    p_ffcx = ffcx.get_options(form_compiler_options)
    p_jit = jit.get_options(jit_options)
    ufcx_form, module, code = jit.ffcx_jit(comm, form, p_ffcx, p_jit)
    return CompiledForm(form, ufcx_form, module, code, p_ffcx["scalar_type"])


def form_cpp_creator(
    dtype: npt.DTypeLike,
) -> typing.Union[
    _cpp.fem.Form_float32,
    _cpp.fem.Form_float64,
    _cpp.fem.Form_complex64,
    _cpp.fem.Form_complex128,
]:
    """Return the wrapped C++ constructor for creating a variational form of a specific scalar type.

    Args:
        dtype: Scalar type of the required form class.

    Returns:
        Wrapped C++ form class of the requested type.

    Note:
        This function is for advanced usage, typically when writing
        custom kernels using Numba or C.
    """
    if np.issubdtype(dtype, np.float32):
        return _cpp.fem.create_form_float32
    elif np.issubdtype(dtype, np.float64):
        return _cpp.fem.create_form_float64
    elif np.issubdtype(dtype, np.complex64):
        return _cpp.fem.create_form_complex64
    elif np.issubdtype(dtype, np.complex128):
        return _cpp.fem.create_form_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")


def create_form(
    form: CompiledForm,
    function_spaces: list[function.FunctionSpace],
    mesh: Mesh,
    subdomains: dict[IntegralType, list[tuple[int, np.ndarray]]],
    coefficient_map: dict[ufl.Function, function.Function],
    constant_map: dict[ufl.Constant, function.Constant],
    entity_maps: dict[Mesh, np.typing.NDArray[np.int32]] | None = None,
) -> Form:
    """
    Create a Form object from a data-independent compiled form

    Args:
        form: Compiled ufl form
        function_spaces: List of function spaces associated with the form.
            Should match the number of arguments in the form.
        mesh: Mesh to associate form with
        subdomains: A map from integral type to a list of pairs, where each pair corresponds to a
            subdomain id and the set of of integration entities to integrate over.
            Can be computed with {py:func}`dolfinx.fem.compute_integration_domains`.
        coefficient_map: Map from UFL coefficient to function with data
        constant_map: Map from UFL constant to constant with data
        entity_map: A map where each key corresponds to a mesh different to the integration
            domain `mesh`.
        The value of the map is an array of integers, where the i-th entry is the entity in
            the key mesh.
    """
    if entity_maps is None:
        _entity_maps = {}
    else:
        _entity_maps = {m._cpp_object: emap for (m, emap) in entity_maps.items()}

    _subdomain_data = subdomains.copy()
    for _, idomain in _subdomain_data.items():
        idomain.sort(key=lambda x: x[0])

    # Extract name of ufl objects and map them to their corresponding C++ object
    ufl_coefficients = ufl.algorithms.extract_coefficients(form.ufl_form)
    coefficients = {
        f"w{ufl_coefficients.index(u)}": uh._cpp_object for (u, uh) in coefficient_map.items()
    }
    ufl_constants = ufl.algorithms.analysis.extract_constants(form.ufl_form)
    constants = {f"c{ufl_constants.index(u)}": uh._cpp_object for (u, uh) in constant_map.items()}

    ftype = form_cpp_creator(form.dtype)
    f = ftype(
        form.module.ffi.cast("uintptr_t", form.module.ffi.addressof(form.ufcx_form)),
        [fs._cpp_object for fs in function_spaces],
        coefficients,
        constants,
        _subdomain_data,
        _entity_maps,
        mesh._cpp_object,
    )
    return Form(f, form.ufcx_form, form.code)
