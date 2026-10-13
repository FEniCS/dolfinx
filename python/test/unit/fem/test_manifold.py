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
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, has_debug
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    assemble_vector,
    create_interpolation_data,
    discrete_gradient,
    form,
    functionspace,
    interpolation_matrix,
)
from dolfinx.graph import adjacencylist
from dolfinx.la import InsertMode, vector
from dolfinx.mesh import (
    CellType,
    GhostMode,
    cell_num_entities,
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


def plane_mesh(
    n,
    gdim,
    dtype=default_real_type,
    mixed_orientation=False,
    orient=True,
    ghost_mode=GhostMode.none,
):
    """Triangulate the unit square with ``2 * n**2`` cells.

    For ``gdim == 3`` the square is embedded in R^3 as the plane
    ``z = x``, giving a flat manifold whose cells all share the tangent
    plane spanned by ``TANGENTS[3]``.

    Every cell's vertices run anticlockwise, so all cell normals agree.
    With ``mixed_orientation`` every other cell is reversed instead.
    With ``orient`` the facet signs are computed, which H(div) spaces
    on the manifold need.
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
    mesh = create_mesh(MPI.COMM_WORLD, cells, domain, x.astype(dtype), ghost_mode=ghost_mode)
    if orient:
        mesh.topology.create_facet_orientations()
    return mesh


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


def assert_on_all_ranks(comm, check, *args, **kwargs):
    """Call ``check(*args, **kwargs)``, and fail on every rank if it fails on any.

    A check of rank-local data can fail on some ranks only. Those would
    move on to the next test while the others wait in a collective
    call, so a parallel run would hang instead of failing.
    """
    try:
        check(*args, **kwargs)
        error = None
    except AssertionError as e:
        error = e
    failed = [r for r, f in enumerate(comm.allgather(error is not None)) if f]
    if error is not None:
        raise error
    assert not failed, f"Check failed on rank(s) {failed}"


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
def test_piola_element_value_shape(gdim, spec) -> None:
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
def test_blocked_value_shape_independent_of_gdim(gdim, shape, symmetry) -> None:
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
def test_scalar_value_shape(gdim) -> None:
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
def test_interpolate_callable(gdim, spec) -> None:
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
def test_interpolate_piola_to_dg(gdim, spec) -> None:
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
def test_interpolate_same_map(gdim, families) -> None:
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
def test_interpolate_expression(gdim, spec) -> None:
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
def test_interpolate_nonmatching_meshes(gdim) -> None:
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
def test_eval_piola(gdim, spec) -> None:
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
def test_eval_piola_is_tangential(gdim, spec) -> None:
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
def test_interpolation_matrix(gdim) -> None:
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
def test_discrete_gradient(gdim) -> None:
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


def num_signed(mesh):
    """Number of signed facets of the owned cells, over all ranks."""
    # A rank may hold no cells, so the number of facets per cell is taken
    # from the cell type rather than from the size of the signs
    num_owned = mesh.topology.index_map(2).size_local
    signs = mesh.topology.facet_orientations()
    per_cell = cell_num_entities(mesh.topology.cell_type, 1)
    local = int(np.sum(signs[: num_owned * per_cell] < 0))
    return mesh.comm.allreduce(local, op=MPI.SUM)


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


def cube_surface(ghost_mode, cell_type=CellType.tetrahedron, dtype=default_real_type):
    """The boundary of a cube, as a facet submesh.

    Its cells keep the vertex order of the cube's facets, so their
    normals point both inwards and outwards. They are triangles for a
    tetrahedral and quadrilaterals for a hexahedral cube. The submesh
    inherits the ghost cells of the cube, if any.
    """
    cube = create_unit_cube(
        MPI.COMM_WORLD, 3, 3, 3, cell_type=cell_type, ghost_mode=ghost_mode, dtype=dtype
    )
    cube.topology.create_connectivity(2, 3)
    return create_submesh(cube, 2, exterior_facet_indices(cube.topology))[0]


def mobius_mesh(n, m, dtype=default_real_type, ghost_mode=GhostMode.none):
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
    return create_mesh(MPI.COMM_WORLD, cells, domain, x.astype(dtype), ghost_mode=ghost_mode)


def t_joint_mesh(dests, ghosted, dtype=default_real_type):
    """Three triangles sharing one edge, cell ``i`` on rank ``dests[i]``.

    The ranks are taken modulo the number of ranks. If ``ghosted``, each
    cell is ghosted on the ranks of the other two, so every rank holding
    the edge sees all three of its cells; otherwise no rank sees more
    than its own.
    """
    if MPI.COMM_WORLD.rank == 0:
        x = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]])
        cells = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    else:
        x = np.zeros((0, 3))
        cells = np.zeros((0, 3), dtype=np.int64)

    def partitioner(comm, nparts, dual_graph, cell_weights, edge_weights, ghosting):
        # All cells share the edge, so each is ghosted on the ranks of the
        # others. The owner comes first.
        ranks = [r % comm.size for r in dests[: dual_graph.num_nodes]]
        rows = [[r, *sorted(set(ranks) - {r})] if ghosted else [r] for r in ranks]
        offsets = np.cumsum([0, *(len(row) for row in rows)], dtype=np.int32)
        return adjacencylist(np.array([r for row in rows for r in row], dtype=np.int32), offsets)

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
    facets they share are signed.
    """
    family, degree, linear = spec
    mesh = plane_mesh(2, gdim, mixed_orientation=True, orient=False)
    contravariant_on_manifold = gdim == 3 and family in ("RT", "BDM")
    if contravariant_on_manifold:
        mesh.topology.create_facet_orientations()
        assert num_signed(mesh) > 0, "the reversed cells should be signed"
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))
    f, f_ufl = tangential_field(mesh, linear)

    w = Function(V, dtype=default_real_type)

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
    assert_on_all_ranks(
        mesh.comm, np.testing.assert_allclose, values, f(midpoints.T).T, atol=tol(mesh)
    )


@pytest.mark.parametrize("degree, linear", [(1, False), (2, True)])
def test_interpolate_discontinuous_rt_on_mixed_cell_orientations(degree, linear):
    """Discontinuous RT on a manifold needs no cell orientation.

    Every other cell is reversed. With no continuity between cells, the
    field on each cell follows from that cell's basis alone, so the
    space can be created and interpolated into without an orientation.
    Computing one afterwards leaves the degrees-of-freedom unchanged.
    """
    mesh = plane_mesh(2, 3, mixed_orientation=True, orient=False)
    V = functionspace(
        mesh, element("RT", "triangle", degree, discontinuous=True, dtype=default_real_type)
    )
    f, f_ufl = tangential_field(mesh, linear)

    # Interpolation of a callable, and of a compiled expression
    w = Function(V, dtype=default_real_type)
    w.interpolate(f)
    assert l2_error(mesh, w - f_ufl) < tol(mesh)
    w_expr = Function(V, dtype=default_real_type)
    w_expr.interpolate(Expression(f_ufl, V.element.interpolation_points, dtype=default_real_type))
    assert l2_error(mesh, w_expr - f_ufl) < tol(mesh)

    # Function.eval, which pushes the basis forward in C++
    num_cells = mesh.topology.index_map(2).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    midpoints = compute_midpoints(mesh, 2, cells)
    values = w.eval(midpoints, cells).reshape(num_cells, 3)
    assert_on_all_ranks(
        mesh.comm, np.testing.assert_allclose, values, f(midpoints.T).T, atol=tol(mesh)
    )

    # The signs are ignored
    mesh.topology.create_facet_orientations()
    assert num_signed(mesh) > 0
    w_oriented = Function(V, dtype=default_real_type)
    w_oriented.interpolate(f)
    assert_on_all_ranks(mesh.comm, np.testing.assert_array_equal, w_oriented.x.array, w.x.array)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize(
    "cell_type, family, degree",
    [
        (CellType.tetrahedron, "RT", 1),
        (CellType.tetrahedron, "RT", 2),
        (CellType.tetrahedron, "BDM", 1),
        (CellType.hexahedron, "RTCF", 1),
        (CellType.hexahedron, "RTCF", 2),
        (CellType.hexahedron, "BDMCF", 1),
    ],
)
def test_divergence_theorem_on_a_closed_surface(cell_type, family, degree, ghost_mode, dtype):
    """``int div(phi) dx = 0`` on a closed surface, for every basis function.

    Equivalently ``int div(w) dx = 0`` for any conforming ``w``. The flux
    of a basis function through an edge cancels between the two cells
    sharing it, provided they agree about its direction. Where they do
    not, it adds up to ``+-2``. Checking each basis function rather than
    one field avoids cancellation between edges. The surface of a
    hexahedral cube has quadrilateral cells, which run their edges in
    other directions than triangles.
    """
    surface = cube_surface(ghost_mode, cell_type, dtype)
    surface.topology.create_facet_orientations()

    # The facets of the submesh are not consistently ordered, so plenty
    # of them carry a sign, which is what the test is here to exercise
    assert num_signed(surface) > 0, "the test needs facets of both orientations"

    V = functionspace(surface, element(family, surface.basix_cell(), degree, dtype=dtype))
    divergence = form(ufl.div(ufl.TestFunction(V)) * ufl.dx, dtype=dtype)
    num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # ``max_i |int div(phi_i) dx|`` over all basis functions. A basis
    # function's flux through an edge is O(1), so its rounding is O(tol).
    b = assemble_vector(divergence)
    b.scatter_reverse(InsertMode.add)
    local_max = np.max(np.abs(b.array[:num_owned_dofs]), initial=0)
    assert surface.comm.allreduce(local_max, op=MPI.MAX) < tol(surface)


# Basix's edges of each cell, and whether walking round the cell in its
# vertex order runs edge i from its first to its second vertex
CELL_EDGES = {
    "triangle": [(1, 2), (0, 2), (0, 1)],
    "quadrilateral": [(0, 1), (0, 2), (1, 3), (2, 3)],
}
EDGE_RUNS_FORWARD = [True, False, True, False]


@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_facet_orientations_follow_the_vertex_orders(cell_type, ghost_mode):
    """The signs are recomputed from the vertex orders, independently.

    A facet is signed on the cell with the larger global index exactly
    when its two cells run it in the same direction. That is checked on
    every facet whose two cells this rank holds. Ghost cells must carry
    the signs of their owner, which is checked by sending the owner's.
    """
    surface = cube_surface(ghost_mode, cell_type)
    surface.topology.create_facet_orientations()
    topology = surface.topology
    signs = topology.facet_orientations()

    topology.create_connectivity(2, 1)
    topology.create_connectivity(1, 2)
    c_to_v = topology.connectivity(2, 0)
    c_to_e = topology.connectivity(2, 1)
    e_to_c = topology.connectivity(1, 2)
    cell_map = topology.index_map(2)
    num_owned = cell_map.size_local
    num_cells = num_owned + cell_map.num_ghosts
    vertex_map = topology.index_map(0)
    num_vertices = vertex_map.size_local + vertex_map.num_ghosts
    global_cell = cell_map.local_to_global(np.arange(num_cells, dtype=np.int32))
    global_vertex = vertex_map.local_to_global(np.arange(num_vertices, dtype=np.int32))
    edges = CELL_EDGES[topology.cell_name()]
    per_cell = len(edges)
    assert signs.dtype == np.int8
    assert signs.shape == (num_cells * per_cell,)

    def runs_up(c, i):
        p, q = edges[i] if EDGE_RUNS_FORWARD[i] else edges[i][::-1]
        v = c_to_v.links(c)
        return global_vertex[v[p]] < global_vertex[v[q]]

    checked = 0
    wrong = []
    for c in range(num_cells):
        for i, e in enumerate(c_to_e.links(c)):
            cells = e_to_c.links(e)
            if len(cells) != 2:
                continue  # The other cell is on another rank
            d = cells[1] if cells[0] == c else cells[0]
            j = list(c_to_e.links(d)).index(e)
            signed = global_cell[c] > global_cell[d] and runs_up(c, i) == runs_up(d, j)
            if signs[c * per_cell + i] != (-1 if signed else 1):
                wrong.append((int(global_cell[c]), i))
            checked += 1
    assert_on_all_ranks(surface.comm, np.testing.assert_equal, wrong, [])
    assert surface.comm.allreduce(checked, op=MPI.SUM) > 0

    # Ghosts carry their owner's signs
    packed = np.sum((signs.reshape(num_cells, per_cell) < 0) << np.arange(per_cell), axis=1)
    owners = vector(cell_map, dtype=np.int32)
    owners.array[:num_owned] = packed[:num_owned]
    owners.scatter_forward()
    assert_on_all_ranks(
        surface.comm,
        np.testing.assert_array_equal,
        owners.array[num_owned:],
        packed[num_owned:],
    )


def test_facet_orientations_before_and_after():
    """All facets carry sign ``1`` until they are signed, then ``1`` or ``-1``."""
    mesh = plane_mesh(2, 3, mixed_orientation=True, orient=False)
    mesh.topology.create_cell_permutations()
    cell_map = mesh.topology.index_map(2)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    signs = mesh.topology.facet_orientations()
    assert signs.dtype == np.int8
    assert signs.shape == (3 * num_cells,)
    assert_on_all_ranks(mesh.comm, np.testing.assert_array_equal, signs, 1)

    mesh.topology.create_facet_orientations()
    signs = mesh.topology.facet_orientations()
    assert signs.shape == (3 * num_cells,)
    assert_on_all_ranks(mesh.comm, np.testing.assert_array_equal, np.abs(signs), 1)
    assert num_signed(mesh) > 0


def test_facet_orientations_do_not_change_other_elements():
    """Orientations change no element off a manifold, nor covariant ones on it."""
    for gdim, family in [(2, "RT"), (3, "N1curl")]:
        mesh = plane_mesh(2, gdim, mixed_orientation=True, orient=False)
        V = functionspace(mesh, element(family, "triangle", 1, dtype=default_real_type))
        f, _ = tangential_field(mesh, False)
        before = Function(V, dtype=default_real_type)
        before.interpolate(f)
        mesh.topology.create_facet_orientations()
        assert num_signed(mesh) > 0
        after = Function(V, dtype=default_real_type)
        after.interpolate(f)
        assert_on_all_ranks(mesh.comm, np.testing.assert_array_equal, after.x.array, before.x.array)


@pytest.mark.parametrize("family,degree", [("RT", 1), ("RT", 2), ("BDM", 1)])
def test_moebius_strip_is_conforming(family, degree):
    """A non-orientable surface is made conforming one facet at a time.

    No per-cell orientation makes every pair of neighbours agree on a
    Möbius strip, so the facets left disagreeing carry a sign of their
    own. The test is that the normal component, which is what
    H(div)-conformity constrains, has no jump across any interior facet.
    """
    # The dS diagnostic needs the cell across each process boundary
    mesh = mobius_mesh(24, 4, ghost_mode=GhostMode.shared_facet)
    mesh.topology.create_facet_orientations()
    V = functionspace(mesh, element(family, "triangle", degree, dtype=default_real_type))

    u = Function(V, dtype=default_real_type)
    u.x.array[:] = np.arange(1, u.x.array.size + 1)
    u.x.scatter_forward()

    n = ufl.FacetNormal(mesh)
    jump = ufl.jump(u, n)
    jump2 = mesh.comm.allreduce(
        assemble_scalar(form(ufl.inner(jump, jump) * ufl.dS, dtype=default_real_type)),
        op=MPI.SUM,
    )
    size = mesh.comm.allreduce(
        assemble_scalar(form(ufl.inner(u, u) * ufl.dx, dtype=default_real_type)), op=MPI.SUM
    )
    assert np.sqrt(abs(jump2)) < 1e3 * tol(mesh) * np.sqrt(abs(size))


@pytest.mark.parametrize("ghosted", [True, False])
@pytest.mark.parametrize("dests", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 2)])
def test_facet_orientations_refuse_a_t_joint(dests, ghosted):
    """An edge shared by three cells cannot be oriented, so it is refused.

    ``dests`` places the cells on ranks, so that in parallel no rank owns
    all cells of the edge: two on one rank and one on another, or one on
    each of three ranks. Without ghosts no rank sees all three cells, so
    the count has to be combined across ranks.

    Every rank holding the edge raises. In Release builds a rank without
    it returns normally, as the error is not reduced over ranks there.
    """
    mesh = t_joint_mesh(dests, ghosted)
    cell_map = mesh.topology.index_map(2)
    if has_debug or cell_map.size_local + cell_map.num_ghosts > 0:
        with pytest.raises(RuntimeError, match="more than two cells"):
            mesh.topology.create_facet_orientations()
    else:
        mesh.topology.create_facet_orientations()


def test_function_space_needs_facet_orientations():
    """H(div) spaces on a manifold need an orientation, other spaces do not."""

    def make(family, degree=1, **kwargs):
        return element(family, "triangle", degree, dtype=default_real_type, **kwargs)

    mesh = plane_mesh(1, 3, orient=False)
    assert not mesh.topology.has_facet_orientations()
    needs = [make("RT"), make("BDM"), mixed_element([make("RT"), make("DG", degree=0)])]
    for e in needs:
        with pytest.raises(RuntimeError, match="create_facet_orientations"):
            functionspace(mesh, e)
    for e in [make("N1curl"), make("RT", discontinuous=True), make("Lagrange")]:
        functionspace(mesh, e)
    functionspace(plane_mesh(1, 2, orient=False), make("RT"))  # Not a manifold

    mesh.topology.create_facet_orientations()
    assert mesh.topology.has_facet_orientations()
    for e in needs:
        functionspace(mesh, e)
    functionspace(mesh, needs[2]).sub(0).collapse()
    assert plane_mesh(1, 3).topology.has_facet_orientations()


def test_facet_orientations_need_a_surface():
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    with pytest.raises(ValueError, match="surface mesh"):
        mesh.topology.create_facet_orientations()
    mesh.topology.create_cell_permutations()
    with pytest.raises(ValueError, match="surface mesh"):
        mesh.topology.facet_orientations()
