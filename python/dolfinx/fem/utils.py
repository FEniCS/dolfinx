# Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and
# Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Utility functions for finite element computations."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.fem import build_sparsity_pattern as _build_sparsity_pattern
from dolfinx.cpp.fem import compute_integration_domains as _compute_integration_domains
from dolfinx.cpp.fem import create_interpolation_data as _create_interpolation_data
from dolfinx.cpp.fem import create_sparsity_pattern as _create_sparsity_pattern
from dolfinx.cpp.fem import discrete_curl as _discrete_curl
from dolfinx.cpp.fem import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem import interpolation_matrix as _interpolation_matrix
from dolfinx.cpp.la import SparsityPattern
from dolfinx.fem.element import CoordinateElement
from dolfinx.fem.function import FunctionSpace
from dolfinx.geometry import PointOwnershipData as _PointOwnershipData
from dolfinx.la import MatrixCSR as _MatrixCSR

if typing.TYPE_CHECKING:
    import dolfinx.mesh
    from dolfinx.cpp.fem import _IntegralType as IntegralType


def create_sparsity_pattern(a: dolfinx.fem.forms.Form) -> SparsityPattern:
    """Create a sparsity pattern from a bilinear form.

    Args:
        a: Bilinear form to build a sparsity pattern for.

    Returns:
        Sparsity pattern for the form ``a``.

    Note:
        The pattern is not finalised, i.e. the caller is responsible for
        calling ``assemble`` on the sparsity pattern.
    """
    return _create_sparsity_pattern(a._cpp_object)


def build_sparsity_pattern(pattern: SparsityPattern, a: dolfinx.fem.forms.Form):
    """Build a sparsity pattern from a bilinear form.

    Args:
        pattern: The sparsity pattern to add to
        a: Bilinear form to build a sparsity pattern for.

    Returns:
        Sparsity pattern for the form ``a``.

    Note:
        The pattern is not finalised, i.e. the caller is responsible for
        calling ``assemble`` on the sparsity pattern.
    """
    return _build_sparsity_pattern(pattern, a._cpp_object)


def create_interpolation_data(
    V_to: FunctionSpace,
    V_from: FunctionSpace,
    cells: npt.NDArray[np.int32],
    padding: float = 1e-14,
) -> _PointOwnershipData:
    """Generate data for interpolating functions on different meshes.

    Args:
        V_to: Function space to interpolate into.
        V_from: Function space to interpolate from.
        cells: Indices of the cells associated with `V_to` on which to
            interpolate into.
        padding: Absolute padding of bounding boxes of all entities on
            mesh_to.

    Returns:
        Data needed to interpolation functions defined on function
        spaces on the meshes.
    """
    return _PointOwnershipData(
        _create_interpolation_data(
            V_to.mesh._cpp_object.geometry,
            V_to.element._cpp_object,
            V_from.mesh._cpp_object,
            cells,
            padding,
        )
    )


def discrete_curl(V0: FunctionSpace, V1: FunctionSpace) -> _MatrixCSR:
    """Assemble a discrete curl operator.

    The discrete curl operator interpolates the curl of H(curl) finite
    element function into a H(div) space.

    Args:
        V0: H1(curl) space to interpolate the curl from.
        V1: H(div) space to interpolate into.

    Returns:
        Discrete curl operator.
    """
    return _MatrixCSR(_discrete_curl(V0._cpp_object, V1._cpp_object))


def discrete_gradient(space0: FunctionSpace, space1: FunctionSpace) -> _MatrixCSR:
    """Assemble a discrete gradient operator.

    The discrete gradient operator interpolates the gradient of a H1
    finite element function into a H(curl) space. It is assumed that the
    H1 space uses an identity map and the H(curl) space uses a covariant
    Piola map.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete gradient operator.
    """
    return _MatrixCSR(_discrete_gradient(space0._cpp_object, space1._cpp_object))


def interpolation_matrix(space0: FunctionSpace, space1: FunctionSpace) -> _MatrixCSR:
    """Create interpolation matrix between spaces on the same mesh.

    Args:
        space0: space to interpolate from.
        space1: space to interpolate into.

    Returns:
        Interpolation matrix.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.
    """
    return _MatrixCSR(_interpolation_matrix(space0._cpp_object, space1._cpp_object))


def compute_integration_domains(
    integral_type: IntegralType,
    topology: dolfinx.mesh.Topology,
    entities: np.ndarray,
):
    """Determine compute integration entities.

    This function returns a list ``[(id, entities)]``. For cell
    integrals ``entities`` are the cell indices. For exterior facet
    integrals, ``entities`` is a list of ``(cell_index,
    local_facet_index)`` pairs. For interior facet integrals,
    ``entities`` is a list of ``(cell_index0, local_facet_index0,
    cell_index1, local_facet_index1)``. ``id`` refers to the subdomain
    id used in the definition of the integration measures of the
    variational form.

    Note:
        Owned mesh entities only are returned. Ghost entities are not
        included.

    Note:
        For facet integrals, the topology facet-to-cell and
        cell-to-facet connectivity must be computed before calling this
        function.

    Args:
        integral_type: Integral type.
        topology: Mesh topology.
        entities: List of mesh entities. For
            ``integral_type==IntegralType.cell``, ``entities`` should be
            cell indices. For other ``IntegralType``s, ``entities``
            should be facet indices.

    Returns:
        List of integration entities.
    """
    return _compute_integration_domains(integral_type, topology._cpp_object, entities)


def interpolate_geometry(msh: dolfinx.mesh.Mesh, cmap: CoordinateElement) -> dolfinx.mesh.Mesh:
    """From a mesh create a mesh with geometry interpolated into cmap.

    Useful for creating a higher-order mesh from a lower-order one for
    computation, or vice-versa, for IO.

    Note:
        The topology is shared between ``msh`` and the returned mesh.

    Args:
        msh: Input mesh.
        cmap: Coordinate element for the new geometry.

    Returns:
        A new mesh with geometry in ``cmap``.
    """
    import dolfinx.mesh as _mesh  # lazy to avoid circular import

    new_msh = _cpp.fem.interpolate_geometry(msh._cpp_object, cmap._cpp_object)
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            _mesh.to_string(new_msh.topology.cell_type),
            new_msh.geometry.cmaps[0].degree,
            basix.LagrangeVariant(new_msh.geometry.cmaps[0].variant),
            shape=(new_msh.geometry.dim,),
            dtype=new_msh.geometry.x.dtype,
        )
    )
    return _mesh.Mesh(new_msh, domain)
