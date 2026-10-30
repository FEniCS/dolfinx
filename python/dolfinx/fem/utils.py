# Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells,
# Jack S. Hale and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Utility functions for finite element computations."""

from __future__ import annotations

import typing
from collections.abc import Callable

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
from dolfinx.cpp.fem import evaluate_basis_functions as _evaluate_basis_functions
from dolfinx.cpp.fem import interpolation_coords as _interpolation_coords
from dolfinx.cpp.fem import interpolation_matrix as _interpolation_matrix
from dolfinx.cpp.fem import (
    nonmatching_interpolation_matrix as _nonmatching_interpolation_matrix,
)
from dolfinx.fem.element import CoordinateElement
from dolfinx.fem.function import FunctionSpace
from dolfinx.geometry import PointOwnershipData as _PointOwnershipData
from dolfinx.geometry import determine_point_ownership as _determine_point_ownership
from dolfinx.la import MatrixCSR as _MatrixCSR
from dolfinx.la import SparsityPattern as _SparsityPattern

if typing.TYPE_CHECKING:
    # 'dolfinx.la.SparsityPattern' is spelled out in the annotations
    # below because a bare 'SparsityPattern' is ambiguous in the
    # generated docs, matching 'dolfinx.cpp.la.SparsityPattern' too.
    # 'dolfinx.la' itself is not imported here: 'dolfinx.mesh' below
    # already binds the 'dolfinx' package name, and 'dolfinx/__init__.py'
    # imports 'la', so 'dolfinx.la.SparsityPattern' resolves without a
    # second, conflicting import style for the same module.
    import dolfinx.mesh
    from dolfinx.cpp.fem import IntegralType as IntegralType


def create_sparsity_pattern(a: dolfinx.fem.forms.Form) -> dolfinx.la.SparsityPattern:
    """Create a sparsity pattern from a bilinear form.

    Args:
        a: Bilinear form to build a sparsity pattern for.

    Returns:
        Sparsity pattern for the form ``a``.

    Note:
        The pattern is not finalised, i.e. the caller is responsible for
        calling :meth:`SparsityPattern.finalize
        <dolfinx.la.SparsityPattern.finalize>`.
    """
    return _SparsityPattern(_create_sparsity_pattern(a._cpp_object))


def build_sparsity_pattern(pattern: dolfinx.la.SparsityPattern, a: dolfinx.fem.forms.Form) -> None:
    """Build a sparsity pattern from a bilinear form.

    Args:
        pattern: The sparsity pattern to add to
        a: Bilinear form to build a sparsity pattern for.

    Note:
        The pattern is not finalised, i.e. the caller is responsible for
        calling :meth:`SparsityPattern.finalize
        <dolfinx.la.SparsityPattern.finalize>`.
    """
    _build_sparsity_pattern(pattern._cpp_object, a._cpp_object)


def create_interpolation_data(
    V_to: FunctionSpace,
    V_from: FunctionSpace,
    cells: npt.NDArray[np.int32],
    padding: float = 1e-14,
    allow_extrapolation: bool = True,
) -> _PointOwnershipData:
    """Generate data for interpolating functions on different meshes.

    Args:
        V_to: Function space to interpolate into.
        V_from: Function space to interpolate from.
        cells: Indices of the cells associated with `V_to` on which to
            interpolate into.
        padding: Absolute padding applied to the bounding box of each
            cell in `V_from`'s mesh before searching for candidate
            cells. Increasing ``padding`` increases the number of
            cells considered as candidates for an interpolation point;
            it does not by itself decide whether a point with no
            actually-containing cell is assigned an owner, which is
            controlled by ``allow_extrapolation``.
        allow_extrapolation: If ``True`` (default), a point from
            `V_to`'s mesh not actually contained in any candidate cell
            of `V_from`'s mesh is instead assigned the candidate cell
            closest to it (relevant e.g. if the two meshes do not fully
            overlap). If ``False``, such points are left unowned.

    Returns:
        Data needed to interpolation functions defined on function
        spaces on the meshes.
    """
    match V_to.mesh._cpp_object.geometry, V_to.element._cpp_object, V_from.mesh._cpp_object:
        case (
            _cpp.mesh.Geometry_float32() as geometry0,
            _cpp.fem.FiniteElement_float32() as element0,
            _cpp.mesh.Mesh_float32() as mesh1,
        ):
            return _PointOwnershipData(
                _create_interpolation_data(
                    geometry0, element0, mesh1, cells, padding, allow_extrapolation
                )
            )
        case (
            _cpp.mesh.Geometry_float64() as geometry0,
            _cpp.fem.FiniteElement_float64() as element0,
            _cpp.mesh.Mesh_float64() as mesh1,
        ):
            return _PointOwnershipData(
                _create_interpolation_data(
                    geometry0, element0, mesh1, cells, padding, allow_extrapolation
                )
            )
        case _:
            raise TypeError(
                "create_interpolation_data requires V_to and V_from to have the same dtype."
            )


def evaluate_basis_functions(
    V: FunctionSpace,
    x: npt.ArrayLike,
    cells: npt.ArrayLike,
    tol: float = 1e-6,
    maxit: int = 15,
    num_threads: int = 1,
) -> npt.NDArray[np.floating]:
    """Evaluate the basis functions of a function space at points.

    Each point is pulled back to the reference cell of the cell that
    contains it (by a Newton method if the geometry is non-affine), the
    reference basis is tabulated, transformed for the cell's
    orientation and pushed forward. This is the evaluation underlying
    :meth:`Function.eval <dolfinx.fem.Function.eval>`.

    For a blocked (e.g. vector-valued Lagrange) space, the basis of one
    block is evaluated.

    Note:
        If the element needs dof transformations and the cell
        permutations of the mesh have not yet been computed, they are
        created here, which is collective.

    Args:
        V: Function space whose basis is evaluated.
        x: Points in physical (not reference) space, with shape
            ``(num_points, 3)``.
        cells: Index of the cell containing each point, with shape
            ``(num_points,)``. Points with a negative cell index are
            skipped and their basis values are zero.
        tol: Tolerance for convergence of the Newton method for
            non-affine pull-backs.
        maxit: Maximum number of Newton iterations for non-affine
            pull-backs.
        num_threads: Number of threads. The points are split into
            contiguous chunks, one per thread. Must be >= 1.

    Returns:
        Basis values with shape ``(num_points, num_basis_functions,
        value_size)``, where ``num_basis_functions`` is the number of
        basis functions of one block on a cell, in the order of the
        cell's degrees-of-freedom, and ``value_size`` is the physical
        value size of one block.
    """
    _x = np.asarray(x, dtype=V.mesh.geometry.x.dtype)
    if _x.ndim >= 3:
        raise ValueError("x must be a 1D or 2D array of points.")
    _x = (
        _x.reshape(-1, 3)
        if _x.size == 0
        else np.reshape(_x, (_x.shape[0] if _x.ndim == 2 else 1, -1))
    )
    if _x.shape[1] != 3:
        raise ValueError("Coordinate(s) for basis evaluation must have length 3.")
    _cells = np.asarray(cells, dtype=np.int32)
    if _cells.ndim >= 2:
        raise ValueError("cells must be a 1D array of cell indices.")
    _cells = np.reshape(_cells, _cells.shape[0] if _cells.ndim == 1 else 1)
    return _evaluate_basis_functions(V._cpp_object, _x, _cells, tol, maxit, num_threads)


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
    match V0._cpp_object, V1._cpp_object:
        case (
            _cpp.fem.FunctionSpace_float32() as cpp_V0,
            _cpp.fem.FunctionSpace_float32() as cpp_V1,
        ):
            return _MatrixCSR(_discrete_curl(cpp_V0, cpp_V1))
        case (
            _cpp.fem.FunctionSpace_float64() as cpp_V0,
            _cpp.fem.FunctionSpace_float64() as cpp_V1,
        ):
            return _MatrixCSR(_discrete_curl(cpp_V0, cpp_V1))
        case _:
            raise TypeError("discrete_curl requires V0 and V1 to have the same dtype.")


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
    match space0._cpp_object, space1._cpp_object:
        case (
            _cpp.fem.FunctionSpace_float32() as cpp_space0,
            _cpp.fem.FunctionSpace_float32() as cpp_space1,
        ):
            return _MatrixCSR(_discrete_gradient(cpp_space0, cpp_space1))
        case (
            _cpp.fem.FunctionSpace_float64() as cpp_space0,
            _cpp.fem.FunctionSpace_float64() as cpp_space1,
        ):
            return _MatrixCSR(_discrete_gradient(cpp_space0, cpp_space1))
        case _:
            raise TypeError("discrete_gradient requires space0 and space1 to have the same dtype.")


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
    match space0._cpp_object, space1._cpp_object:
        case (
            _cpp.fem.FunctionSpace_float32() as cpp0,
            _cpp.fem.FunctionSpace_float32() as cpp1,
        ):
            return _MatrixCSR(_interpolation_matrix(cpp0, cpp1))
        case (
            _cpp.fem.FunctionSpace_float64() as cpp0,
            _cpp.fem.FunctionSpace_float64() as cpp1,
        ):
            return _MatrixCSR(_interpolation_matrix(cpp0, cpp1))
        case _:
            raise TypeError(
                "interpolation_matrix requires space0 and space1 to have the same dtype."
            )


#: Expansion of interpolation points into weighted points, see
#: :func:`nonmatching_interpolation_matrix`.
PointExpansion = Callable[
    [npt.NDArray[np.floating], npt.NDArray[np.int32]],
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
]


def _nonmatching_interpolation_data(
    V0: FunctionSpace,
    V1: FunctionSpace,
    cells: npt.ArrayLike | None,
    expansion: PointExpansion | None,
    padding: float,
    allow_extrapolation: bool,
) -> tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.floating], _PointOwnershipData
]:
    """Cells, row dofs, weights and point ownership across meshes.

    For a point-evaluation ``V1`` without an ``expansion``, the points
    are one per owned dof block (:func:`_point_evaluation_rows`) and the
    row dofs are returned. Otherwise the points are the interpolation
    points of every cell and the row dofs are empty.
    """
    if cells is None:
        num_cells = V1.mesh.topology.index_map(V1.mesh.topology.dim).size_local
        _cells = np.arange(num_cells, dtype=np.int32)
    else:
        _cells = np.asarray(cells, dtype=np.int32).reshape(-1)

    dtype = V0.mesh.geometry.x.dtype
    if expansion is None:
        x, _, dofs = _point_evaluation_rows(V1, _cells)
        if dofs.size > 0:
            data = _determine_point_ownership(
                V0.mesh,
                np.ascontiguousarray(x, dtype=dtype),
                padding,
                find_closest_cell=allow_extrapolation,
            )
            return _cells, dofs, np.ones(dofs.size, dtype=dtype), data

    # Interpolation points of V1 in physical space, ordered by cell
    x = _interpolation_coords(  # type: ignore[call-overload]
        V1.element._cpp_object, V1.mesh.geometry._cpp_object, _cells
    ).T
    num_points = x.shape[0]
    num_ip = V1.element.interpolation_points.shape[0]
    point_cells = np.repeat(_cells, num_ip)

    if expansion is None:
        points = x
        weights = np.ones(num_points, dtype=dtype)
    else:
        _points, _weights = expansion(x, point_cells)
        points = np.asarray(_points, dtype=dtype)
        weights = np.asarray(_weights, dtype=dtype)
        if (
            points.ndim != 3
            or points.shape[0] != num_points
            or points.shape[2] != 3
            or weights.shape != points.shape[:2]
        ):
            raise ValueError(
                "expansion must return points with shape (num_points, nq, 3) and "
                "weights with shape (num_points, nq)."
            )

    data = _determine_point_ownership(
        V0.mesh,
        np.ascontiguousarray(points.reshape(-1, 3), dtype=dtype),
        padding,
        find_closest_cell=allow_extrapolation,
    )
    return _cells, np.empty(0, dtype=np.int32), np.ascontiguousarray(weights.reshape(-1)), data


def _point_evaluation_rows(
    V1: FunctionSpace, cells: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Points, cells and owned dof blocks of point-evaluation rows.

    One point per owned dof block, from the first cell containing it, if
    the element of ``V1`` is a point evaluation (e.g. Lagrange or DG).
    All empty otherwise.
    """
    return _cpp.fem.point_evaluation_rows(V1._cpp_object, cells)  # type: ignore[arg-type]


def _nonmatching_matrix_from_data(
    V0: FunctionSpace,
    V1: FunctionSpace,
    cells: npt.NDArray[np.int32],
    dofs: npt.NDArray[np.int32],
    weights: npt.NDArray[np.floating],
    data: _PointOwnershipData,
    tol: float,
    maxit: int,
    num_threads: int,
) -> _MatrixCSR:
    """Build the matrix from precomputed interpolation data."""
    dtype = V0.mesh.geometry.x.dtype
    if V1.mesh.geometry.x.dtype != dtype:
        raise TypeError(
            "nonmatching_interpolation_matrix requires V0 and V1 to have the same dtype."
        )
    return _MatrixCSR(
        _nonmatching_interpolation_matrix(  # type: ignore[no-matching-overload]
            V0._cpp_object,
            V1._cpp_object,
            cells,
            dofs,
            weights.astype(dtype),
            data._cpp_object,
            tol,
            maxit,
            num_threads,
        )
    )


def _hierarchy_interpolation_data(
    V0: FunctionSpace,
    V1: FunctionSpace,
    parent_cells: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.floating], _PointOwnershipData
]:
    """Cells, row dofs, weights and point ownership for a hierarchy.

    :func:`_nonmatching_interpolation_data` finds the owning rank and
    cell of every point by searching a bounding-box tree. Here the mesh
    of ``V1`` is a refinement of the mesh of ``V0`` that was not
    redistributed, so each fine cell lies inside its parent cell on the
    same process, and the routing is known exactly: every point stays
    local.

    The layout required of the
    :class:`dolfinx.geometry.PointOwnershipData` is dictated by
    ``impl::scatter_values``: ``dest_owners`` must be in contiguous runs
    by rank, which holds trivially here because the only rank involved
    is this one. ``dest_cells`` indexes cells of the mesh of ``V0``.
    """
    tdim = V1.mesh.topology.dim
    num_cells = V1.mesh.topology.index_map(tdim).size_local
    _cells = np.arange(num_cells, dtype=np.int32)

    _parent = np.asarray(parent_cells, dtype=np.int32).reshape(-1)
    if _parent.size < num_cells:
        raise ValueError("parent_cells must have an entry for each owned cell of the mesh of V1.")
    # parent_cells is indexed by the cell order *before* the reordering
    # done when the refined mesh is constructed. original_cell_index maps
    # each cell of the refined mesh back to that order, as
    # refinement::transfer_cell_meshtag does.
    original = np.asarray(V1.mesh.topology._cpp_object.original_cell_index, dtype=np.int64)
    offset = V1.mesh.topology.index_map(tdim).local_range[0]
    _parent_final = _parent[(original[:num_cells] - offset).astype(np.int32)]

    coarse_map = V0.mesh.topology.index_map(V0.mesh.topology.dim)
    num_coarse = coarse_map.size_local + coarse_map.num_ghosts
    if num_cells > 0 and (_parent_final.min() < 0 or _parent_final.max() >= num_coarse):
        raise ValueError("parent_cells contains indices outside the mesh of V0.")

    # One point per owned dof block for point-evaluation elements,
    # otherwise the interpolation points of V1 ordered by cell
    x, point_cells, dofs = _point_evaluation_rows(V1, _cells)
    if dofs.size == 0:
        x = _interpolation_coords(  # type: ignore[call-overload]
            V1.element._cpp_object, V1.mesh.geometry._cpp_object, _cells
        ).T
        num_ip = V1.element.interpolation_points.shape[0]
        point_cells = np.repeat(_cells, num_ip)
    dest_cells = _parent_final[point_cells]

    dtype = V0.mesh.geometry.x.dtype
    num_points = x.shape[0]
    weights = np.ones(num_points, dtype=dtype)

    # No communication: this rank owns every point it asks about
    rank = np.int32(V1.mesh.comm.rank)
    local_ranks = np.full(num_points, rank, dtype=np.int32)
    cls = getattr(_cpp.geometry, f"PointOwnershipData_float{8 * np.dtype(dtype).itemsize}")
    data = _PointOwnershipData(
        cls(
            local_ranks,
            local_ranks,
            np.ascontiguousarray(x, dtype=dtype).reshape(-1),
            np.ascontiguousarray(dest_cells, dtype=np.int32),
        )
    )
    return _cells, dofs, weights, data


def hierarchy_interpolation_matrix(
    V0: FunctionSpace,
    V1: FunctionSpace,
    parent_cells: npt.ArrayLike,
    tol: float = 1e-6,
    maxit: int = 15,
    num_threads: int = 1,
) -> _MatrixCSR:
    """Interpolation matrix from a mesh to its refinement.

    Equivalent to :func:`nonmatching_interpolation_matrix` when the mesh
    of ``V1`` is a refinement of the mesh of ``V0``, but without the
    collision detection: ``parent_cells`` already names the cell of the
    mesh of ``V0`` containing each cell of the mesh of ``V1``, so no
    bounding-box search and no communication are needed.

    Args:
        V0: Coarse space to interpolate from.
        V1: Refined space to interpolate to.
        parent_cells: Cell of the mesh of ``V0`` containing each owned
            cell of the mesh of ``V1``, as returned by
            :func:`dolfinx.mesh.refine` with
            :attr:`dolfinx.mesh.RefinementOption.parent_cell`.
        tol: Tolerance of the Newton method for non-affine pull-backs.
        maxit: Maximum number of Newton iterations for non-affine
            pull-backs.
        num_threads: Number of threads used to evaluate the basis. Must
            be >= 1.

    Returns:
        Interpolation matrix, as for
        :func:`nonmatching_interpolation_matrix`.

    Note:
        The refinement must not have been redistributed, i.e. each
        refined cell must be on the process owning its parent. This
        holds for :func:`dolfinx.mesh.refine` with the default identity
        partitioner.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.
    """
    _cells, dofs, weights, data = _hierarchy_interpolation_data(V0, V1, parent_cells)
    return _nonmatching_matrix_from_data(
        V0, V1, _cells, dofs, weights, data, tol, maxit, num_threads
    )


def nonmatching_interpolation_matrix(
    V0: FunctionSpace,
    V1: FunctionSpace,
    cells: npt.ArrayLike | None = None,
    expansion: PointExpansion | None = None,
    padding: float = 1e-14,
    allow_extrapolation: bool = True,
    tol: float = 1e-6,
    maxit: int = 15,
    num_threads: int = 1,
) -> _MatrixCSR:
    """Create the interpolation matrix between spaces on different meshes.

    If :math:`u_0` is the degree-of-freedom vector of a function in
    ``V0``, then :math:`u_1 = A u_0` is the degree-of-freedom vector of
    its interpolation into ``V1`` on ``cells``. The meshes of ``V0`` and
    ``V1`` need not match, and may be distributed differently.

    The basis of ``V0`` is evaluated at the interpolation points of
    ``V1``, pushed forward to physical space. An ``expansion`` can
    replace each interpolation point by several weighted points, whose
    values are summed before the interpolation operator of ``V1`` is
    applied, e.g. to interpolate the average over a circle around each
    point.

    Args:
        V0: Space to interpolate from.
        V1: Space to interpolate to.
        cells: Cells of the mesh of ``V1`` to interpolate on. Defaults
            to the cells owned by this process.
        expansion: Callable ``expansion(x, cells)`` taking the
            interpolation points ``x`` with shape ``(num_points, 3)``
            and the cell of each point, and returning points with shape
            ``(num_points, nq, 3)`` and their weights with shape
            ``(num_points, nq)``. Any normalisation (e.g. division by
            the length of a circle) must be included in the weights. If
            ``None``, each interpolation point is used with weight one.
        padding: Absolute padding of the cell bounding boxes of the mesh
            of ``V0`` when searching for the cells containing the points.
        allow_extrapolation: If ``True``, a point not contained in any
            cell of the mesh of ``V0`` is assigned the closest candidate
            cell. If ``False``, such a point does not contribute.
        tol: Tolerance of the Newton method for non-affine pull-backs.
        maxit: Maximum number of Newton iterations for non-affine
            pull-backs.
        num_threads: Number of threads used to evaluate the basis.

    Returns:
        Interpolation matrix. Its rows are indexed by the index map of
        ``V1``. Its columns are indexed by the index map of ``V0``
        extended with the ghosts needed by the matrix
        (``A.index_map(1)``), so a vector to multiply must be created
        from that index map.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.
    """
    _cells, dofs, weights, data = _nonmatching_interpolation_data(
        V0, V1, cells, expansion, padding, allow_extrapolation
    )
    return _nonmatching_matrix_from_data(
        V0, V1, _cells, dofs, weights, data, tol, maxit, num_threads
    )


def compute_integration_domains(
    integral_type: IntegralType,
    topology: dolfinx.mesh.Topology,
    entities: np.ndarray,
) -> npt.NDArray[np.int32]:
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

    If ``cmap`` is discontinuous, the new geometry is discontinuous: each
    cell has its own coordinate nodes, which are not shared with
    neighbouring cells, so cell geometries can be moved independently of
    each other. This is required, e.g., for periodic meshes.

    Note:
        The topology is shared between ``msh`` and the returned mesh.

    Note:
        A discontinuous geometry has no coordinate degrees-of-freedom
        associated with the sub-entities of a cell, so functions that
        extract the geometry of a sub-entity, e.g.
        :func:`dolfinx.mesh.entities_to_geometry`, do not support it.

    Args:
        msh: Input mesh.
        cmap: Coordinate element for the new geometry.

    Returns:
        A new mesh with geometry in ``cmap``.
    """
    import dolfinx.mesh as _mesh  # lazy to avoid circular import

    new_msh: _cpp.mesh.Mesh_float32 | _cpp.mesh.Mesh_float64
    match msh._cpp_object, cmap._cpp_object:
        case (
            _cpp.mesh.Mesh_float32() as cpp_msh,
            _cpp.fem.CoordinateElement_float32() as cpp_cmap,
        ):
            new_msh = _cpp.fem.interpolate_geometry(cpp_msh, cpp_cmap)
        case (
            _cpp.mesh.Mesh_float64() as cpp_msh,
            _cpp.fem.CoordinateElement_float64() as cpp_cmap,
        ):
            new_msh = _cpp.fem.interpolate_geometry(cpp_msh, cpp_cmap)
        case _:
            raise TypeError("interpolate_geometry requires msh and cmap to have the same dtype.")
    new_cmap = new_msh.geometry.cmaps[0]
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            _mesh.to_string(new_msh.topology.cell_type),
            new_cmap.degree,
            basix.LagrangeVariant(new_cmap.variant),
            discontinuous=new_cmap.is_discontinuous,
            shape=(new_msh.geometry.dim,),
            dtype=new_msh.geometry.x.dtype,
        )
    )
    return _mesh.Mesh(new_msh, domain)
