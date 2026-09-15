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
    Expression,
    Function,
    assemble_scalar,
    create_interpolation_data,
    discrete_gradient,
    form,
    functionspace,
    interpolation_matrix,
)
from dolfinx.mesh import create_mesh

# Tangent vectors of the embedding plane used by ``plane_mesh``. Every
# cell of the mesh lies in this plane, so any constant combination of
# these is tangential to every cell.
TANGENTS = {2: (np.array([1.0, 0.0]), np.array([0.0, 1.0]))}
TANGENTS[3] = (np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))


def plane_mesh(n, gdim, dtype=default_real_type):
    """Triangulate the unit square with ``2 * n**2`` cells.

    For ``gdim == 3`` the square is embedded in R^3 as the plane
    ``z = x``, giving a flat manifold whose cells all share the tangent
    plane spanned by ``TANGENTS[3]``.
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
    norm2 = mesh.comm.allreduce(assemble_scalar(form(ufl.inner(e, e) * ufl.dx)), op=MPI.SUM)
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
    w = Function(V)
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

    w = Function(V)
    w.x.array[:] = np.random.default_rng(seed=7).random(w.x.array.shape)
    q = Function(Q)
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

    w = Function(V)
    w.x.array[:] = np.random.default_rng(seed=11).random(w.x.array.shape)
    v = Function(W)
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
    g = Function(Q)
    g.interpolate(constant_callable(c))

    w = Function(V)
    w.interpolate(Expression(2 * g, V.element.interpolation_points))

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
    u0 = Function(V0)
    u0.interpolate(constant_callable(c))

    cells = np.arange(mesh1.topology.index_map(mesh1.topology.dim).size_local, dtype=np.int32)
    data = create_interpolation_data(V1, V0, cells, padding=1e-6)
    u1 = Function(V1)
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
    g = Function(Q)
    g.interpolate(constant_callable(c))

    w = Function(V)
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

    u = Function(W)
    u.interpolate(lambda x: 2.0 * x[0] - 3.0 * x[1])

    w = Function(V)
    w.x.array[:] = discrete_gradient(W, V).to_dense() @ u.x.array

    assert l2_error(mesh, w - ufl.grad(u)) < tol(mesh)
