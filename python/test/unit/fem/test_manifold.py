# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for finite element spaces on manifolds (``gdim > tdim``).

A finite element basis is tabulated on the reference cell and pushed
forward to the physical cell. For a Piola-mapped element the push-forward
uses the Jacobian ``J`` of shape ``(gdim, tdim)``, so the *physical* value
shape of, e.g., Raviart-Thomas on a triangle is ``(gdim,)`` while its
*reference* value shape is ``(tdim,)``. The two coincide only when
``gdim == tdim``, which is why every test here is parametrised over
``gdim``: the ``gdim == 2`` case is the control that must pass both
before and after the fix.

See https://github.com/FEniCS/dolfinx/issues/3619.
"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    create_interpolation_data,
    discrete_gradient,
    form,
    functionspace,
    interpolation_matrix,
)
from dolfinx.graph import adjacencylist
from dolfinx.mesh import (
    CellType,
    GhostMode,
    cell_normals,
    compute_midpoints,
    create_mesh,
    create_submesh,
    create_unit_cube,
    exterior_facet_indices,
)

# Tangent vectors of the embedding plane used by ``plane_mesh``. Every
# cell of the mesh lies in this plane, so any constant combination of
# these is tangential to every cell.
TANGENTS = {2: (np.array([1.0, 0.0]), np.array([0.0, 1.0]))}
TANGENTS[3] = (np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))


def plane_mesh(n, gdim, dtype=default_real_type, mixed_orientation=False):
    """Triangulate the unit square with ``2 * n**2`` cells.

    For ``gdim == 3`` the square is embedded in R^3 as the plane
    ``z = x``, giving a flat manifold whose cells all share the tangent
    plane spanned by ``TANGENTS[3]``.

    Every cell's vertices run anticlockwise, so all cell normals agree.
    With ``mixed_orientation`` every other cell is reversed instead.
    """
    if MPI.COMM_WORLD.rank == 0:
        s = np.linspace(0.0, 1.0, n + 1)
        xs, ys = np.meshgrid(s, s, indexing="ij")
        x = np.column_stack([xs.ravel(), ys.ravel()])
        if gdim == 3:
            x = np.column_stack([x, x[:, 0]])

        def node(i, j):
            return i * (n + 1) + j

        cells = np.array(
            [
                c
                for i in range(n)
                for j in range(n)
                for c in (
                    [node(i, j), node(i + 1, j), node(i, j + 1)],
                    [node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)],
                )
            ],
            dtype=np.int64,
        )
        if mixed_orientation:
            cells[::2] = cells[::2][:, [0, 2, 1]]
    else:
        x = np.zeros((0, gdim))
        cells = np.zeros((0, 3), dtype=np.int64)

    domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(gdim,), dtype=dtype))
    return create_mesh(MPI.COMM_WORLD, cells, domain, x.astype(dtype))


def tangential_constant(gdim):
    """A constant vector lying in the plane of every cell."""
    t0, t1 = TANGENTS[gdim]
    return 0.3 * t0 + 0.7 * t1


def constant_callable(c):
    """Interpolation callable returning the constant ``c`` at every point."""
    return lambda x: np.tile(np.asarray(c).reshape(-1, 1), (1, x.shape[1]))


def l2_error(mesh, e):
    """Global ``||e||`` of a UFL expression over ``mesh``."""
    norm2 = mesh.comm.allreduce(
        assemble_scalar(form(ufl.inner(e, e) * ufl.dx, dtype=default_real_type)), op=MPI.SUM
    )
    return np.sqrt(abs(norm2))


def tol(mesh):
    """Tolerance for a quantity that is exact up to rounding."""
    return 1.0e3 * np.finfo(mesh.geometry.x.dtype).eps


# Piola-mapped elements on a triangle: (family, degree, value rank). The
# physical value shape is (gdim,) for rank 1 and (gdim, gdim) for rank 2.
PIOLA_ELEMENTS = [
    ("RT", 1, 1),
    ("RT", 2, 1),
    ("BDM", 1, 1),
    ("N1curl", 1, 1),
    ("N1curl", 2, 1),
    ("N2curl", 1, 1),
    ("Regge", 0, 2),
    ("HHJ", 0, 2),
]


def piola_id(spec):
    return f"{spec[0]}{spec[1]}"


# -----------------------------------------------------------------------
# Value shapes
# -----------------------------------------------------------------------


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", PIOLA_ELEMENTS, ids=piola_id)
def test_piola_element_value_shape(gdim, spec):
    """``FiniteElement.value_shape`` must be the physical value shape.

    On a manifold it differs from the reference value shape that Basix
    tabulates, and it must agree with the UFL function space, which
    derives it from the pullback and the geometric dimension.
    """
    family, degree, rank = spec
    mesh = plane_mesh(1, gdim)
    el = element(family, "triangle", degree, dtype=default_real_type)
    V = functionspace(mesh, el)
    assert V.value_shape == (gdim,) * rank
    assert tuple(V.element.value_shape) == V.value_shape


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize(
    "shape, symmetry",
    [((2,), None), ((3,), None), ((5,), None), ((3, 3), None), ((2, 2), True), ((3, 3), True)],
)
def test_blocked_value_shape_independent_of_gdim(gdim, shape, symmetry):
    """A blocked element's value shape is whatever the user asked for.

    Blocked elements are built from scalar, identity-mapped base
    elements, so the geometric dimension must not enter: a 5-vector or a
    3x3 tensor space on a 2D mesh keeps its shape. This guards against a
    fix that rewrites trailing axes to ``gdim`` unconditionally.
    """
    mesh = plane_mesh(1, gdim)
    V = functionspace(
        mesh,
        element("Lagrange", "triangle", 1, shape=shape, symmetry=symmetry, dtype=default_real_type),
    )

    assert V.value_shape == shape
    assert tuple(V.element.value_shape) == shape


@pytest.mark.parametrize("gdim", [2, 3])
def test_scalar_value_shape(gdim):
    mesh = plane_mesh(1, gdim)
    el = element("Lagrange", "triangle", 1, dtype=default_real_type)
    V = functionspace(mesh, el)
    assert V.value_shape == ()
    assert tuple(V.element.value_shape) == ()


# -----------------------------------------------------------------------
# Interpolation
# -----------------------------------------------------------------------


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", [s for s in PIOLA_ELEMENTS if s[2] == 1], ids=piola_id)
def test_interpolate_callable(gdim, spec):
    """Interpolate a tangential constant field from a callable.

    The callable returns ``gdim`` components (one per physical
    direction). A vector-valued Piola element of any degree reproduces a
    constant field that lies in the plane of the cell exactly, so the
    interpolant must equal it.

    This is the second reproducer of issue #3619.
    """
    family, degree, _ = spec
    mesh = plane_mesh(2, gdim)
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))

    c = tangential_constant(gdim)
    w = Function(V, dtype=default_real_type)
    w.interpolate(constant_callable(c))

    assert l2_error(mesh, w - ufl.as_vector(c)) < tol(mesh)


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", PIOLA_ELEMENTS, ids=piola_id)
def test_interpolate_piola_to_dg(gdim, spec):
    """Interpolate a Piola-mapped Function into a (blocked) DG space.

    ``DG_k`` of the same degree contains the Piola space, so the
    interpolation is exact and the two Functions must agree pointwise.
    This exercises ``interpolate_nonmatching_maps`` (the map types
    differ), which both checks and sizes its buffers from the element
    value shapes.

    This is the first reproducer of issue #3619.
    """
    family, degree, rank = spec
    mesh = plane_mesh(2, gdim)
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))
    Q = functionspace(
        mesh,
        element("DG", "triangle", max(degree, 1), shape=(gdim,) * rank, dtype=default_real_type),
    )

    w = Function(V, dtype=default_real_type)
    w.x.array[:] = np.random.default_rng(seed=7).random(w.x.array.shape)
    q = Function(Q, dtype=default_real_type)
    q.interpolate(w)

    assert l2_error(mesh, w - q) < tol(mesh)


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("families", [("RT", "BDM"), ("N1curl", "N2curl")])
def test_interpolate_same_map(gdim, families):
    """Interpolate between two elements sharing a map type.

    ``RT_1`` is contained in ``BDM_1`` and ``N1curl_1`` in ``N2curl_1``,
    so this is exact. Takes the ``interpolate_same_map`` branch.
    """
    mesh = plane_mesh(2, gdim)
    V = functionspace(mesh, element(families[0], "triangle", 1, dtype=default_real_type))
    W = functionspace(mesh, element(families[1], "triangle", 1, dtype=default_real_type))

    w = Function(V, dtype=default_real_type)
    w.x.array[:] = np.random.default_rng(seed=11).random(w.x.array.shape)
    v = Function(W, dtype=default_real_type)
    v.interpolate(w)

    assert l2_error(mesh, w - v) < tol(mesh)


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", [s for s in PIOLA_ELEMENTS if s[2] == 1], ids=piola_id)
def test_interpolate_expression(gdim, spec):
    """Interpolate an ``Expression`` into a Piola-mapped space.

    ``Expression`` carries the UFL (physical) value shape, so on a
    manifold it is compared against the element's value shape, which
    must be physical too.
    """
    family, degree, _ = spec
    mesh = plane_mesh(2, gdim)
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))
    Q = functionspace(mesh, element("DG", "triangle", 1, shape=(gdim,), dtype=default_real_type))

    c = tangential_constant(gdim)
    g = Function(Q, dtype=default_real_type)
    g.interpolate(constant_callable(c))

    w = Function(V, dtype=default_real_type)
    w.interpolate(Expression(2 * g, V.element.interpolation_points, dtype=default_real_type))

    assert l2_error(mesh, w - 2 * ufl.as_vector(c)) < tol(mesh)


@pytest.mark.parametrize("gdim", [2, 3])
def test_interpolate_nonmatching_meshes(gdim):
    """Interpolate a Piola-mapped Function between two manifold meshes.

    Both meshes triangulate the same embedded plane, so a tangential
    constant is representable on both and the round trip is exact. This
    path evaluates the source Function at arbitrary physical points
    (``Function::eval``) before interpolating the result.
    """
    mesh0 = plane_mesh(2, gdim)
    mesh1 = plane_mesh(3, gdim)
    V0 = functionspace(mesh0, element("RT", "triangle", 1, dtype=default_real_type))
    V1 = functionspace(mesh1, element("RT", "triangle", 1, dtype=default_real_type))

    c = tangential_constant(gdim)
    u0 = Function(V0, dtype=default_real_type)
    u0.interpolate(constant_callable(c))

    cells = np.arange(mesh1.topology.index_map(mesh1.topology.dim).size_local, dtype=np.int32)
    data = create_interpolation_data(V1, V0, cells, padding=1e-6)
    u1 = Function(V1, dtype=default_real_type)
    u1.interpolate_nonmatching(u0, cells, data)

    assert l2_error(mesh1, u1 - ufl.as_vector(c)) < tol(mesh1)


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", [s for s in PIOLA_ELEMENTS if s[2] == 1], ids=piola_id)
def test_eval_piola(gdim, spec):
    """``Function.eval`` returns ``gdim`` components on a manifold.

    ``Function::eval`` pushes the reference basis forward, so its output
    buffer must be sized with the physical value size.
    """
    family, degree, _ = spec
    mesh = plane_mesh(1, gdim)
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))

    c = tangential_constant(gdim)
    w = Function(V)
    w.interpolate(constant_callable(c))

    # Centroid of cell 0, which is interior to it by construction.
    # ``Function.eval`` always takes points padded to three components.
    x = np.mean(mesh.geometry.x[mesh.geometry.dofmaps[0][0]], axis=0).reshape(1, 3)
    values = np.ravel(w.eval(x, np.array([0], dtype=np.int32)))

    assert values.shape == (gdim,)
    np.testing.assert_allclose(values, c, atol=tol(mesh))


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", [s for s in PIOLA_ELEMENTS if s[2] == 1], ids=piola_id)
def test_eval_piola_is_tangential(gdim, spec):
    """Every value of a Piola-mapped field lies in the plane of its cell.

    The degrees of freedom are set directly, so this exercises
    ``Function::eval`` without relying on interpolation. On a manifold
    the push-forward produces ``gdim`` components; a buffer sized with
    the reference value size instead leaves the trailing component at
    zero, which is detected here as a non-tangential result.
    """
    family, degree, _ = spec
    mesh = plane_mesh(1, gdim)
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))

    w = Function(V)
    w.x.array[:] = np.random.default_rng(seed=13).random(w.x.array.shape)

    x = np.mean(mesh.geometry.x[mesh.geometry.dofmaps[0][0]], axis=0).reshape(1, 3)
    values = np.ravel(w.eval(x, np.array([0], dtype=np.int32)))

    assert values.shape == (gdim,)
    if gdim == 3:
        normal = np.cross(*TANGENTS[3])
        normal /= np.linalg.norm(normal)
        assert abs(np.dot(values, normal)) < tol(mesh)


# -----------------------------------------------------------------------
# Discrete operators
# -----------------------------------------------------------------------


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("gdim", [2, 3])
def test_interpolation_matrix(gdim):
    """The interpolation operator DG -> N1curl must act as interpolation.

    Applying the matrix to a tangential constant must give the
    coefficients of that same constant. ``interpolation_matrix`` sizes
    its push-forward buffers from the element value sizes, so on a
    manifold it silently builds the wrong operator rather than raising.
    """
    mesh = plane_mesh(2, gdim)
    V = functionspace(mesh, element("N1curl", "triangle", 1, dtype=default_real_type))
    Q = functionspace(mesh, element("DG", "triangle", 1, shape=(gdim,), dtype=default_real_type))

    c = tangential_constant(gdim)
    g = Function(Q, dtype=default_real_type)
    g.interpolate(constant_callable(c))

    w = Function(V, dtype=default_real_type)
    w.x.array[:] = interpolation_matrix(Q, V).to_dense() @ g.x.array

    assert l2_error(mesh, w - ufl.as_vector(c)) < tol(mesh)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("gdim", [2, 3])
def test_discrete_gradient(gdim):
    """``discrete_gradient`` maps a P1 field to the N1curl gradient.

    Regression guard: this operator is built purely from degrees of
    freedom, so it is expected to be correct on a manifold already.
    """
    mesh = plane_mesh(2, gdim)
    W = functionspace(mesh, element("Lagrange", "triangle", 1, dtype=default_real_type))
    V = functionspace(mesh, element("N1curl", "triangle", 1, dtype=default_real_type))

    u = Function(W, dtype=default_real_type)
    u.interpolate(lambda x: 2.0 * x[0] - 3.0 * x[1])

    w = Function(V, dtype=default_real_type)
    w.x.array[:] = discrete_gradient(W, V).to_dense() @ u.x.array

    assert l2_error(mesh, w - ufl.grad(u)) < tol(mesh)


# -----------------------------------------------------------------------
# Cell orientations
# -----------------------------------------------------------------------

# Vector-valued elements, and whether each contains a linear tangential
# field (otherwise only a constant one).
ORIENTED_ELEMENTS = [
    ("RT", 1, False),
    ("RT", 2, True),
    ("BDM", 1, True),
    ("N1curl", 1, False),
    ("N1curl", 2, True),
    ("N2curl", 1, True),
]


def oriented_id(spec):
    label = "linear_tangential_field" if spec[2] else "constant"
    return f"{spec[0]}{spec[1]}-{label}"


def tangential_field(mesh, linear):
    """A field tangential to ``plane_mesh``, as a callable and as UFL.

    Constant, or linear in the plane coordinates ``(x, y)``.
    """
    t0, t1 = TANGENTS[mesh.geometry.dim]

    def coefficients(x0, x1, zero):
        return (0.3 + 0.5 * x1, 0.7 - 0.4 * x0) if linear else (0.3 + zero, 0.7 + zero)

    def f(x):
        a, b = coefficients(x[0], x[1], np.zeros(x.shape[1]))
        return np.outer(t0, a) + np.outer(t1, b)

    # A mesh-bound zero, so that the constant field still has a mesh to
    # compile an Expression on
    X = ufl.SpatialCoordinate(mesh)
    a, b = coefficients(X[0], X[1], Constant(mesh, default_real_type(0.0)))
    return f, a * ufl.as_vector(t0) + b * ufl.as_vector(t1)


def reversed_cells(mesh):
    """Whether the orientation of each owned and ghost cell is reversed.

    Read from the bit ``create_cell_orientations`` sets in the cell
    permutation info.
    """
    return (mesh.topology.get_cell_permutation_info() >> 31).astype(bool)


def cube_surface(ghost_mode, cell_type=CellType.tetrahedron):
    """The boundary of a cube, as a facet submesh.

    Its cells keep the vertex order of the cube's facets, so their
    normals point both inwards and outwards. They are triangles for a
    tetrahedral and quadrilaterals for a hexahedral cube.
    """
    cube = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, cell_type=cell_type, ghost_mode=ghost_mode)
    cube.topology.create_connectivity(2, 3)
    return create_submesh(cube, 2, exterior_facet_indices(cube.topology))[0]


def mobius_mesh(n, m, dtype=default_real_type):
    """A Moebius strip, ``n`` cells around and ``m`` across."""
    if MPI.COMM_WORLD.rank == 0:
        width = 0.4

        def node(i, j):
            # The half twist: column n is column 0, upside down
            return i * (m + 1) + j if i < n else m - j

        u = np.repeat(2 * np.pi * np.arange(n) / n, m + 1)
        v = np.tile(np.linspace(-width, width, m + 1), n)
        r = 1 + v * np.cos(u / 2)
        x = np.column_stack([r * np.cos(u), r * np.sin(u), v * np.sin(u / 2)])
        cells = np.array(
            [
                c
                for i in range(n)
                for j in range(m)
                for c in (
                    [node(i, j), node(i + 1, j), node(i, j + 1)],
                    [node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)],
                )
            ],
            dtype=np.int64,
        )
    else:
        x = np.zeros((0, 3))
        cells = np.zeros((0, 3), dtype=np.int64)
    domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(3,), dtype=dtype))
    return create_mesh(MPI.COMM_WORLD, cells, domain, x.astype(dtype))


def t_joint_mesh(dests, dtype=default_real_type):
    """Three triangles sharing one edge, cell ``i`` on rank ``dests[i]``.

    The ranks are taken modulo the number of ranks. The mesh is not
    ghosted, so a rank sees only the cells of the edge that it owns.
    """
    if MPI.COMM_WORLD.rank == 0:
        x = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]])
        cells = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    else:
        x = np.zeros((0, 3))
        cells = np.zeros((0, 3), dtype=np.int64)

    def partitioner(comm, nparts, dual_graph, cell_weights, edge_weights, ghosting):
        ranks = np.array(dests[: dual_graph.num_nodes], dtype=np.int32) % comm.size
        return adjacencylist(ranks)

    domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(3,), dtype=dtype))
    return create_mesh(
        MPI.COMM_WORLD, cells, domain, x.astype(dtype), partitioner, max_facet_to_cell_links=3
    )


@pytest.mark.parametrize("gdim", [2, 3])
@pytest.mark.parametrize("spec", ORIENTED_ELEMENTS, ids=oriented_id)
def test_interpolate_on_mixed_cell_orientations(gdim, spec):
    """Piola-mapped fields on a mesh whose cell normals disagree.

    Every other cell is reversed. In 2D the signed Jacobian determinant
    orients every cell, and covariant fields (N1curl, N2curl) do not
    depend on it: both are controls. A contravariant field (RT, BDM) on
    the manifold uses the unsigned pseudo-determinant, so neighbouring
    cells disagree about the direction of the flux they share until the
    cells are oriented.
    """
    family, degree, linear = spec
    mesh = plane_mesh(2, gdim, mixed_orientation=True)
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))
    f, f_ufl = tangential_field(mesh, linear)

    w = Function(V, dtype=default_real_type)
    if gdim == 3 and family in ("RT", "BDM"):
        w.interpolate(f)
        assert l2_error(mesh, w - f_ufl) > 0.1, "the reversed cells should show"
        mesh.topology.create_cell_orientations()

    # Interpolation of a callable, and of a compiled expression
    w.interpolate(f)
    assert l2_error(mesh, w - f_ufl) < tol(mesh)
    w_expr = Function(V, dtype=default_real_type)
    w_expr.interpolate(Expression(f_ufl, V.element.interpolation_points, dtype=default_real_type))
    assert l2_error(mesh, w_expr - f_ufl) < tol(mesh)

    # Function.eval, which pushes the basis forward in C++
    num_cells = mesh.topology.index_map(2).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    midpoints = compute_midpoints(mesh, 2, cells)
    values = w.eval(midpoints, cells).reshape(num_cells, gdim)
    np.testing.assert_allclose(values, f(midpoints.T).T, atol=tol(mesh))


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("family, degree", [("RT", 1), ("RT", 2), ("BDM", 1)])
def test_divergence_theorem_on_a_closed_surface(family, degree, ghost_mode):
    """``int div(w) dx = 0`` on a closed surface, for any conforming ``w``.

    Summed over the cells, the flux through each edge cancels between
    the two cells sharing it, provided they agree about its direction.
    The degrees-of-freedom are set from their global index, so the field
    does not depend on the partition.
    """
    surface = cube_surface(ghost_mode)
    V = functionspace(surface, element(family, "triangle", degree, dtype=default_real_type))
    w = Function(V, dtype=default_real_type)
    imap = V.dofmap.index_map
    indices = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    w.x.array[:] = np.sin(imap.local_to_global(indices).astype(default_real_type) + 0.3)

    def integral(e):
        return surface.comm.allreduce(
            assemble_scalar(form(e * ufl.dx, dtype=default_real_type)), op=MPI.SUM
        )

    rounding = tol(surface) * integral(abs(ufl.div(w)))
    assert abs(integral(ufl.div(w))) > 1e2 * rounding, "the reversed cells should show"

    surface.topology.create_cell_orientations()
    num_owned = surface.topology.index_map(2).size_local
    num_reversed = surface.comm.allreduce(
        int(np.sum(reversed_cells(surface)[:num_owned])), op=MPI.SUM
    )
    num_cells = surface.topology.index_map(2).size_global
    assert 0 < num_reversed < num_cells, "the test needs cells of both orientations"
    assert abs(integral(ufl.div(w))) < rounding


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_cell_orientations_agree_with_the_outward_normal(ghost_mode, cell_type):
    """On a closed surface the computed orientation is the outward one or its opposite.

    It comes from the vertex orders alone, so it is fixed only up to one
    sign per connected surface, which has to be the same on every rank,
    ghost cells included. Triangle and quadrilateral surfaces run their
    edges in different directions.
    """
    surface = cube_surface(ghost_mode, cell_type)
    surface.topology.create_cell_orientations()
    cell_map = surface.topology.index_map(2)
    cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    own_outward = (
        np.einsum(
            "ci,ci->c",
            cell_normals(surface, 2, cells).reshape(-1, 3),
            compute_midpoints(surface, 2, cells) - 0.5,
        )
        > 0
    )
    # Outward once the orientation is applied
    outward = own_outward != reversed_cells(surface)
    comm = surface.comm
    all_outward = comm.allreduce(bool(np.all(outward)), op=MPI.LAND)
    all_inward = comm.allreduce(not bool(np.any(outward)), op=MPI.LAND)
    assert all_outward or all_inward


def test_cell_orientations_undo_mixed_vertex_orders():
    """The cells that ``plane_mesh`` reverses are flagged opposite to the others.

    With ``mixed_orientation`` the even original cells are reversed.
    Which of the two groups is flagged depends on the partitioning.
    """
    mesh = plane_mesh(2, 3, mixed_orientation=True)
    mesh.topology.create_cell_orientations()
    num_owned = mesh.topology.index_map(2).size_local
    even = np.asarray(mesh.topology.original_cell_index[:num_owned]) % 2 == 0
    flipped = reversed_cells(mesh)[:num_owned]
    comm = mesh.comm
    even_flagged = comm.allreduce(bool(np.all(flipped == even)), op=MPI.LAND)
    odd_flagged = comm.allreduce(bool(np.all(flipped != even)), op=MPI.LAND)
    assert even_flagged or odd_flagged


def test_cell_orientations_do_not_change_other_elements():
    """Orientations change no element off a manifold, nor covariant ones on it."""
    for gdim, family in [(2, "RT"), (3, "N1curl")]:
        mesh = plane_mesh(2, gdim, mixed_orientation=True)
        V = functionspace(mesh, element(family, "triangle", 1, dtype=default_real_type))
        f, _ = tangential_field(mesh, False)
        before = Function(V)
        before.interpolate(f)
        mesh.topology.create_cell_orientations()
        assert mesh.comm.allreduce(bool(np.any(reversed_cells(mesh))), op=MPI.LOR)
        after = Function(V)
        after.interpolate(f)
        np.testing.assert_array_equal(after.x.array, before.x.array)


def test_cell_orientations_refuse_a_moebius_strip():
    """A non-orientable surface has no consistent orientation, so it is refused."""
    mesh = mobius_mesh(24, 4)
    with pytest.raises(RuntimeError, match="not orientable"):
        mesh.topology.create_cell_orientations()


@pytest.mark.parametrize("dests", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 2)])
def test_cell_orientations_refuse_a_t_joint(dests):
    """An edge shared by three cells cannot be oriented, so it is refused.

    ``dests`` places the cells on ranks, so that in parallel no rank owns
    all cells of the edge: two on one rank and one on another, or one on
    each of three ranks.
    """
    mesh = t_joint_mesh(dests)
    with pytest.raises(RuntimeError, match="more than two cells"):
        mesh.topology.create_cell_orientations()


def test_cell_orientations_need_a_surface():
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    with pytest.raises(ValueError, match="surface mesh"):
        mesh.topology.create_cell_orientations()
