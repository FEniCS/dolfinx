# Copyright (C) 2017-2024 Chris N. Richardson, Garth N. Wells,
# Michal Habera and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite element forms."""

from __future__ import annotations

import collections
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

import ffcx
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_scalar_type, jit
from dolfinx.fem import IntegralType
from dolfinx.fem.function import Constant, Function, FunctionSpace

if typing.TYPE_CHECKING:
    # import dolfinx.mesh just when doing type checking to avoid
    # circular import
    from dolfinx.mesh import EntityMap as _EntityMap
    from dolfinx.mesh import Mesh, MeshTags


class Form:
    """A finite element form."""

    _cpp_object: (
        _cpp.fem.Form_complex64
        | _cpp.fem.Form_complex128
        | _cpp.fem.Form_float32
        | _cpp.fem.Form_float64
    )
    _code: str | list[str] | None

    def __init__(
        self,
        form: _cpp.fem.Form_complex64
        | _cpp.fem.Form_complex128
        | _cpp.fem.Form_float32
        | _cpp.fem.Form_float64,
        ufcx_form=None,
        code: str | list[str] | None = None,
        module: types.ModuleType | list[types.ModuleType] | None = None,
    ):
        """Initialize a finite element form.

        Note:
            Forms should normally be constructed using :func:`form` and
            not using this class initialiser. This class is combined
            with different base classes that depend on the scalar type
            used in the Form.

        Args:
            form: Compiled form object.
            ufcx_form: UFCx form.
            code: Form C++ code.
            module: CFFI module.
        """
        self._code = code
        self._ufcx_form = ufcx_form
        self._cpp_object = form
        self._module = module

    @property
    def ufcx_form(self):
        """The compiled ufcx_form object."""
        return self._ufcx_form

    @property
    def code(self) -> str | list[str] | None:
        """C code strings."""
        return self._code

    @property
    def module(self) -> types.ModuleType | list[types.ModuleType] | None:
        """The CFFI module."""
        return self._module

    @property
    def rank(self) -> int:
        """Rank of this form."""
        return self._cpp_object.rank

    @property
    def function_spaces(self) -> list[FunctionSpace]:
        """Function spaces on which this form is defined."""
        return self._cpp_object.function_spaces

    @property
    def dtype(self) -> np.dtype:
        """Scalar type of this form."""
        return np.dtype(self._cpp_object.dtype)

    @property
    def mesh(self) -> _cpp.mesh.Mesh_float32 | _cpp.mesh.Mesh_float64:
        """Mesh on which this form is defined."""
        return self._cpp_object.mesh

    @property
    def integral_types(self):
        """Integral types in the form."""
        return self._cpp_object.integral_types

    def num_integrals(self, integral_type: IntegralType, kernel_index: int) -> int:
        """Number of integrals of a given type for a specific cell type.

        Args:
            integral_type: The type of integral to count.
            kernel_index: In the case of mixed topology, we have a kernel
                per cell type. For single-cell type meshes, this is zero.
        """
        return self._cpp_object.num_integrals(integral_type, kernel_index)


def get_integration_domains(
    integral_type: IntegralType,
    subdomain: MeshTags | list[tuple[int, np.ndarray]] | None,
    subdomain_ids: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Get integration domains from subdomain data.

    The subdomain data is a MeshTags object consisting of markers, or
    ``None``. If it is ``None``, we do not pack any integration
    entities. Integration domains are defined as a list of tuples, where
    each input ``subdomain_ids`` is mapped to an array of integration
    entities, where an integration entity for a cell integral is the
    list of cells. For an exterior facet integral each integration
    entity is a tuple ``(cell_index, local_facet_index)``. For an
    interior facet integral each integration entity is a tuple
    ``(cell_index0, local_facet_index0, cell_index1,
    local_facet_index1)``. Where the first cell-facet pair is the
    ``'+'`` restriction, the second the ``'-'`` restriction.

    Args:
        integral_type: The type of integral to pack integration
            entities for.
        subdomain: A MeshTag with markers or manually specified
            integration domains.
        subdomain_ids: List of ids to integrate over.

    Returns:
        A list of entities to integrate over. For cell integrals, this is a
        list of cells. For exterior facet integrals, this is a list of
        (cell, local_facet) pairs. For interior facet integrals, this is a
        list of (cell0, local_facet0, cell1, local_facet1) tuples.
    """
    if subdomain is None:
        return []
    else:
        domains = []
        if not isinstance(subdomain, list):
            if integral_type in (IntegralType.exterior_facet, IntegralType.interior_facet):
                tdim = subdomain.topology.dim
                subdomain._cpp_object.topology.create_connectivity(tdim - 1, tdim)
                subdomain._cpp_object.topology.create_connectivity(tdim, tdim - 1)

            if integral_type is IntegralType.vertex:
                tdim = subdomain.topology.dim
                subdomain._cpp_object.topology.create_connectivity(0, tdim)
                subdomain._cpp_object.topology.create_connectivity(tdim, 0)

            if integral_type is IntegralType.ridge:
                tdim = subdomain.topology.dim
                subdomain._cpp_object.topology.create_connectivity(tdim - 2, tdim)
                subdomain._cpp_object.topology.create_connectivity(tdim, tdim - 2)

            # Special handling for exterior facets, compared to other
            # one-sided entity integrals
            if integral_type is IntegralType.exterior_facet:
                exterior_facets = _cpp.mesh.exterior_facet_indices(subdomain.topology)

            # Compute integration domains only for each subdomain id in
            # the integrals. If a process has no integral entities,
            # insert an empty array.
            for id in subdomain_ids:
                entities = subdomain.find(id)
                if integral_type is IntegralType.exterior_facet:
                    # Compute intersection of tag an exterior facets
                    entities = np.intersect1d(entities, exterior_facets)

                integration_entities = _cpp.fem.compute_integration_domains(
                    integral_type,
                    subdomain._cpp_object.topology,
                    entities,
                )
                domains.append((id, integration_entities))
            return [(s[0], np.array(s[1])) for s in domains]
        else:
            return [(s[0], np.array(s[1])) for s in sorted(subdomain)]


def form_cpp_class(
    dtype: npt.DTypeLike,
) -> (
    _cpp.fem.Form_float32
    | _cpp.fem.Form_float64
    | _cpp.fem.Form_complex64
    | _cpp.fem.Form_complex128
):
    """Wrapped C++ class of a variational form of a specific scalar type.

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
    "ridge": IntegralType.ridge,
}


def mixed_topology_form(
    forms: Sequence[ufl.Form],
    dtype: npt.DTypeLike = default_scalar_type,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    jit_comm: MPI.IntraComm | None = None,
    entity_maps: Sequence[_EntityMap] | None = None,
):
    """Create a mixed-topology from from an array of Forms.

    # FIXME: This function is a temporary hack for mixed-topology
    meshes. # It is needed because UFL does not know about
    mixed-topology meshes, # so we need to pass a list of forms for each
    cell type.

    Args:
        forms: A list of UFL forms. Each form should be the same, just
            defined on different cell types.
        dtype: Scalar type to use for the compiled form.
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
        jit_comm: MPI communicator used when compiling the form. If
          ``None``, then ``form.mesh.comm``.
        entity_maps: If any trial functions, test functions, or
            coefficients in the form are not defined over the same mesh
            as the integration domain (the domain associated with the
            measure), `entity_maps` must be supplied. For each mesh in
            the form, there should be an entity map relating entities in
            that mesh to the integration domain mesh.

    Returns:
        Compiled finite element Form.
    """
    if form_compiler_options is None:
        form_compiler_options = dict()

    form_compiler_options["scalar_type"] = dtype
    ftype = form_cpp_class(dtype)

    # Extract subdomain data from UFL form
    sd = next(iter(forms)).subdomain_data()
    (domain,) = list(sd.keys())  # Assuming single domain

    # Check that subdomain data for each integral type is the same
    for data in sd.get(domain).values():
        assert all([d is data[0] for d in data if d is not None])

    mesh = domain.ufl_cargo()
    if mesh is None:
        raise RuntimeError("Expecting to find a Mesh in the form.")
    comm = mesh.comm if jit_comm is None else jit_comm

    ufcx_forms = []
    modules = []
    codes = []
    for form in forms:
        ufcx_form, module, code = jit.ffcx_jit(
            comm,
            form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        ufcx_forms.append(ufcx_form)
        modules.append(module)
        codes.append(code)

    # In a mixed-topology mesh, each form has the same C++ function
    # space, so we can extract it from any of them
    V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]

    # TODO coeffs, constants, subdomains, entity_maps
    f = ftype(
        [module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)) for ufcx_form in ufcx_forms],
        V,
        [],
        [],
        {},
        [],
        mesh,
    )
    return Form(f, ufcx_forms, codes, modules)


def form(
    form: ufl.Form | Sequence[ufl.Form] | Sequence[Sequence[ufl.Form]],
    dtype: npt.DTypeLike = default_scalar_type,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    jit_comm: MPI.IntraComm | None = None,
    entity_maps: Sequence[_EntityMap] | None = None,
):
    """Create a Form or list of Forms.

    Args:
        form: A UFL form or iterable of UFL forms.
        dtype: Scalar type to use for the compiled form.
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
        jit_comm: MPI communicator used when compiling the form. If
          `None`, then `form.mesh.comm`.
        entity_maps: If any trial functions, test functions, or
            coefficients in the form are not defined over the same mesh
            as the integration domain (the domain associated with the
            measure), `entity_maps` must be supplied. For each mesh in
            the form, there should be an entity map relating entities in
            that mesh to the integration domain mesh.

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
        """Compile a single UFL form."""
        # Extract subdomain data from UFL form
        sd = form.subdomain_data()
        (domain,) = list(sd.keys())  # Assuming single domain

        # Check that subdomain data for each integral type is the same
        for data in sd.get(domain).values():
            assert all([d is data[0] for d in data if d is not None])

        msh = domain.ufl_cargo()
        if msh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")
        if jit_comm is None:
            comm = msh.comm
        else:
            comm = jit_comm

        ufcx_form, module, code = jit.ffcx_jit(
            comm, form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        # For each argument in form extract its function space
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]
        part = form_compiler_options.get("part", "full")
        if part == "diagonal":
            V = [V[0]]

        # Prepare coefficients data. For every coefficient in form take
        # its C++ object.
        original_coeffs = form.coefficients()
        coeffs = [
            original_coeffs[ufcx_form.original_coefficient_positions[i]]._cpp_object
            for i in range(ufcx_form.num_coefficients)
        ]
        constants = [c._cpp_object for c in form.constants()]

        # Extract subdomain ids from ufcx_form
        subdomain_ids = {type: [] for type in sd.get(domain).keys()}
        integral_offsets = [ufcx_form.form_integral_offsets[i] for i in range(6)]
        for i in range(len(integral_offsets) - 1):
            integral_type = IntegralType(i)
            for j in range(integral_offsets[i], integral_offsets[i + 1]):
                subdomain_ids[integral_type.name].append(ufcx_form.form_integral_ids[j])

        # Subdomain markers (possibly empty list for some integral types)
        subdomains = {
            _ufl_to_dolfinx_domain[key]: get_integration_domains(
                _ufl_to_dolfinx_domain[key], subdomain_data[0], subdomain_ids[key]
            )
            for (key, subdomain_data) in sd.get(domain).items()
        }

        if entity_maps is None:
            _entity_maps = []
        else:
            _entity_maps = [entity_map._cpp_object for entity_map in entity_maps]

        f = ftype(
            [module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form))],
            V,
            coeffs,
            constants,
            subdomains,
            _entity_maps,
            msh,
        )
        return Form(f, ufcx_form, code, module)

    def _zero_form(form):
        """Compile a single 'zero' UFL form.

        I.e. a form with no integrals.
        """
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]
        assert len(V) > 0
        msh = V[0].mesh

        f = ftype(
            spaces=V,
            integrals={},
            coefficients=[],
            constants=[],
            need_permutation_data=False,
            entity_maps=[],
            mesh=msh,
        )
        return Form(f)

    def _create_form(form):
        """Recursively convert ufl.Forms to dolfinx.fem.Form.

        Args:
            form: UFL form or list of UFL forms to extract DOLFINx forms
                from.

        Returns:
            A ``dolfinx.fem.Form`` or a list of ``dolfinx.fem.Form``.
        """
        if isinstance(form, ufl.Form):
            return _form(form)
        elif isinstance(form, ufl.ZeroBaseForm):
            return _zero_form(form)
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _create_form(sub_form), form))
        else:
            return form

    return _create_form(form)


def extract_function_spaces(
    forms: Form | Sequence[Form] | Sequence[Sequence[Form]],
    index: int = 0,
) -> FunctionSpace | list[None | FunctionSpace]:
    """Extract common function spaces from an array of forms.

    If ``forms`` is a list of linear forms, this function returns of list
    of the corresponding test function spaces. If ``forms`` is a 2D
    array of bilinear forms, for ``index=0`` the list of common test
    function spaces for each row is returned, and if ``index=1`` the
    common trial function spaces for each column are returned.

    Args:
        forms: A list of forms or a 2D array of forms.
        index: Index of the function space to extract. If ``index=0``,
            the test function spaces are extracted, if ``index=1`` the
            trial function spaces are extracted.

    Returns:
        List of function spaces.
    """
    _forms = np.array(forms)
    if _forms.ndim == 0:
        form: Form = _forms.tolist()
        return form.function_spaces[0] if form is not None else None
    elif _forms.ndim == 1:
        assert index == 0, "Expected index=0 for 1D array of forms"
        for form in _forms:
            if form is not None:
                assert form.rank == 1, "Expected linear form"
        return [form.function_spaces[0] if form is not None else None for form in forms]  # type: ignore[union-attr]
    elif _forms.ndim == 2:
        assert index == 0 or index == 1, "Expected index=0 or index=1 for 2D array of forms"
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
    """Compiled UFL form without associated DOLFINx data."""

    ufl_form: ufl.Form  # The original ufl form
    ufcx_form: typing.Any  # The compiled form
    module: typing.Any  #  The module
    code: str  # The source code
    dtype: npt.DTypeLike  # data type used for the `ufcx_form`


def compile_form(
    comm: MPI.Intracomm,
    form: ufl.Form,
    form_compiler_options: dict | None = {"scalar_type": default_scalar_type},
    jit_options: dict | None = None,
) -> CompiledForm:
    """Compile UFL form without associated DOLFINx data.

    Args:
        comm: The MPI communicator used when compiling the form
        form: The UFL form to compile
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`.
    """
    p_ffcx = ffcx.get_options(form_compiler_options)
    p_jit = jit.get_options(jit_options)
    ufcx_form, module, code = jit.ffcx_jit(comm, form, p_ffcx, p_jit)
    scalar_type: npt.DTypeLike = p_ffcx["scalar_type"]  # type: ignore [assignment]
    return CompiledForm(form, ufcx_form, module, code, scalar_type)


def form_cpp_creator(
    dtype: npt.DTypeLike,
) -> (
    _cpp.fem.Form_float32
    | _cpp.fem.Form_float64
    | _cpp.fem.Form_complex64
    | _cpp.fem.Form_complex128
):
    """A wrapped C++ constructor for a form with a specified scalar type.

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
    V: list[FunctionSpace],
    msh: Mesh,
    subdomains: dict[IntegralType, list[tuple[int, np.ndarray]]],
    coefficient_map: dict[ufl.Coefficient, Function],
    constant_map: dict[ufl.Constant, Constant],
    entity_maps: Sequence[_EntityMap] | None = None,
) -> Form:
    """Create a Form object from a data-independent compiled form.

    Args:
        form: Compiled ufl form,
        V: List of function spaces associated with the form. Should
            match the number of arguments in the form.
        msh: Mesh to associate form with.
        subdomains: A map from integral type to a list of pairs, where
            each pair corresponds to a subdomain id and the set of of
            integration entities to integrate over. Can be computed with
            {py:func}`dolfinx.fem.compute_integration_domains`.
        coefficient_map: Map from UFL coefficient to function with data.
        constant_map: Map from UFL constant to constant with data.
            to the integration domain ``msh``. The value of the map is
            an array of integers, where the i-th entry is the entity in
            the key mesh.
        entity_maps: Entity maps to support cases where forms involve
            sub-meshes.

    Return:
        A Form object.
    """
    if entity_maps is None:
        _entity_maps = []
    else:
        _entity_maps = [entity_map._cpp_object for entity_map in entity_maps]

    _subdomain_data = subdomains.copy()
    for _, idomain in _subdomain_data.items():
        idomain.sort(key=lambda x: x[0])

    # Extract all coefficients of the compiled form in correct order
    coefficients = {}
    original_coefficients = ufl.algorithms.extract_coefficients(form.ufl_form)
    num_coefficients = form.ufcx_form.num_coefficients
    for c in range(num_coefficients):
        original_index = form.ufcx_form.original_coefficient_positions[c]
        original_coeff = original_coefficients[original_index]
        try:
            coefficients[f"w{c}"] = coefficient_map[original_coeff]._cpp_object
        except KeyError:
            raise RuntimeError(f"Missing coefficient {original_coeff}")

    # Extract all constants of the compiled form in correct order
    # NOTE: Constants are not eliminated
    original_constants = ufl.algorithms.analysis.extract_constants(form.ufl_form)
    num_constants = form.ufcx_form.num_constants
    if num_constants != len(original_constants):
        raise RuntimeError(
            f"Number of constants in compiled form ({num_constants})",
            f"does not match the original form {len(original_constants)}",
        )
    constants = {}
    for counter, constant in enumerate(original_constants):
        try:
            mapped_constant = constant_map[constant]
            constants[f"c{counter}"] = mapped_constant._cpp_object
        except KeyError:
            raise RuntimeError(f"Missing constant {constant}")

    ftype = form_cpp_creator(form.dtype)
    f = ftype(
        form.module.ffi.cast("uintptr_t", form.module.ffi.addressof(form.ufcx_form)),
        [fs._cpp_object for fs in V],
        coefficients,
        constants,
        _subdomain_data,
        _entity_maps,
        msh._cpp_object,
    )
    return Form(f, form.ufcx_form, form.code)


def derivative_block(
    F: ufl.Form | Sequence[ufl.Form],
    u: Function | Sequence[Function],
    du: ufl.Argument | Sequence[ufl.Argument] | None = None,
) -> ufl.Form | Sequence[Sequence[ufl.Form]]:
    """Return the UFL derivative of a (list of) UFL rank one form(s).

    This is commonly used to derive a block Jacobian from a block
    residual.

    If ``F_i`` is a list of forms, the Jacobian is a list of lists with
    :math:`J_{ij} = \\frac{\\partial F_i}{u_j}[\\delta u_j]` using
    ``ufl.derivative`` called component-wise.

    If ``F`` is a form, the Jacobian is computed as :math:`J =
    \\frac{\\partial F}{\\partial u}[\\delta u]`. This is identical to
    calling ``ufl.derivative`` directly.
    """  # noqa: D301
    if isinstance(F, ufl.Form):
        if not isinstance(u, Function):
            raise ValueError("Must provide a single function when F is a UFL form")
        if du is None:
            du = ufl.TrialFunction(u.function_space)
        return ufl.derivative(F, u, du)
    else:
        assert all([isinstance(Fi, ufl.Form) for Fi in F]), "F must be a sequence of UFL forms"
        assert len(F) == len(u), "Number of forms and functions must be equal"
        if du is not None:
            assert len(F) == len(du), "Number of forms and du must be equal"
        else:
            du = [ufl.TrialFunction(u_i.function_space) for u_i in u]
        return [[ufl.derivative(Fi, u_j, du_j) for u_j, du_j in zip(u, du)] for Fi in F]
