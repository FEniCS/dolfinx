# Copyright (C) 2011 Johan Hake
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the FunctionSpace class"""
import pytest

import basix.finite_element
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import create_unit_cube, create_mesh
from ufl import (FiniteElement, TestFunction, TrialFunction, VectorElement,
                 grad, triangle, Mesh, Cell)
from ufl.log import UFLException
import numpy as np
from mpi4py import MPI


@pytest.fixture
def mesh():
    return create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, ('Lagrange', 1))


@pytest.fixture
def W(mesh):
    return VectorFunctionSpace(mesh, ('Lagrange', 1))


@pytest.fixture
def Q(mesh):
    W = VectorElement('Lagrange', mesh.ufl_cell(), 1)
    V = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    return FunctionSpace(mesh, W * V)


@pytest.fixture
def f(V):
    return Function(V)


@pytest.fixture
def V2(f):
    return f.function_space


@pytest.fixture
def g(W):
    return Function(W)


@pytest.fixture
def W2(g):
    return g.function_space


def test_python_interface(V, V2, W, W2, Q):
    # Test Python interface of cpp generated FunctionSpace
    assert isinstance(V, FunctionSpace)
    assert isinstance(W, FunctionSpace)
    assert isinstance(V2, FunctionSpace)
    assert isinstance(W2, FunctionSpace)

    assert V.ufl_cell() == V2.ufl_cell()
    assert W.ufl_cell() == W2.ufl_cell()
    assert V.element == V2.element
    assert W.element == W2.element
    assert V.ufl_element() == V2.ufl_element()
    assert W.ufl_element() == W2.ufl_element()
    assert W is W2
    assert V is V2


def test_component(V, W, Q):
    assert not W.component()
    assert not V.component()
    assert W.sub(0).component()[0] == 0
    assert W.sub(1).component()[0] == 1
    assert Q.sub(0).component()[0] == 0
    assert Q.sub(1).component()[0] == 1


def test_equality(V, V2, W, W2):
    assert V == V  # /NOSONAR
    assert V == V2
    assert W == W
    assert W == W2


def test_sub(Q, W):
    X = Q.sub(0)

    assert W.dofmap.dof_layout.num_dofs == X.dofmap.dof_layout.num_dofs
    for dim, entity_count in enumerate([4, 6, 4, 1]):
        assert W.dofmap.dof_layout.num_entity_dofs(dim) == X.dofmap.dof_layout.num_entity_dofs(dim)
        assert W.dofmap.dof_layout.num_entity_closure_dofs(dim) == X.dofmap.dof_layout.num_entity_closure_dofs(dim)
        for i in range(entity_count):
            assert len(W.dofmap.dof_layout.entity_dofs(dim, i)) \
                == len(X.dofmap.dof_layout.entity_dofs(dim, i)) \
                == len(X.dofmap.dof_layout.entity_dofs(dim, 0))
            assert len(W.dofmap.dof_layout.entity_closure_dofs(dim, i)) \
                == len(X.dofmap.dof_layout.entity_closure_dofs(dim, i)) \
                == len(X.dofmap.dof_layout.entity_closure_dofs(dim, 0))

    assert W.dofmap.dof_layout.block_size == X.dofmap.dof_layout.block_size
    assert W.dofmap.bs * len(W.dofmap.cell_dofs(0)) == len(X.dofmap.cell_dofs(0))

    assert W.element.num_sub_elements == X.element.num_sub_elements
    assert W.element.space_dimension == X.element.space_dimension
    assert W.element.value_shape == X.element.value_shape
    assert W.element.interpolation_points().shape == X.element.interpolation_points().shape
    assert W.element == X.element


def test_inclusion(V, Q):
    assert V.contains(V)
    assert not Q.contains(V)

    assert Q.contains(Q)
    assert Q.contains(Q.sub(0))
    assert Q.contains(Q.sub(1))
    assert Q.contains(Q.sub(0).sub(0))
    assert Q.contains(Q.sub(0).sub(1))

    assert not Q.sub(0).contains(Q)
    assert Q.sub(0).contains(Q.sub(0))
    assert not Q.sub(0).contains(Q.sub(1))
    assert Q.sub(0).contains(Q.sub(0).sub(0))
    assert Q.sub(0).contains(Q.sub(0).sub(1))

    assert not Q.sub(1).contains(Q)
    assert not Q.sub(1).contains(Q.sub(0))
    assert Q.sub(1).contains(Q.sub(1))
    assert not Q.sub(1).contains(Q.sub(0).sub(0))
    assert not Q.sub(1).contains(Q.sub(0).sub(1))

    assert not Q.sub(0).sub(0).contains(Q)
    assert not Q.sub(0).sub(0).contains(Q.sub(0))
    assert not Q.sub(0).sub(0).contains(Q.sub(1))
    assert Q.sub(0).sub(0).contains(Q.sub(0).sub(0))
    assert not Q.sub(0).sub(0).contains(Q.sub(0).sub(1))


def test_not_equal(W, V, W2, V2):
    assert W != V
    assert W2 != V2


def test_clone(W):
    assert W.clone() is not W


def test_collapse(W, V):
    Vs = W.sub(2)
    with pytest.raises(RuntimeError):
        Function(Vs)
    assert Vs.dofmap.cell_dofs(0)[0] != V.dofmap.cell_dofs(0)[0]

    # Collapse the space it should now be the same as V
    Vc = Vs.collapse()[0]
    assert Vc.dofmap.cell_dofs(0)[0] == V.dofmap.cell_dofs(0)[0]
    f0 = Function(V)
    f1 = Function(Vc)
    assert f0.vector.getSize() == f1.vector.getSize()


def test_argument_equality(mesh, V, V2, W, W2):
    """Placed this test here because it's mainly about detecting differing
    function spaces"""
    mesh2 = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)
    V3 = FunctionSpace(mesh2, ("Lagrange", 1))
    W3 = VectorFunctionSpace(mesh2, ("Lagrange", 1))

    for TF in (TestFunction, TrialFunction):
        v = TF(V)
        v2 = TF(V2)
        v3 = TF(V3)
        assert v == v2
        assert v2 == v
        assert V != V3
        assert V2 != V3
        assert not v == v3
        assert not v2 == v3
        assert v != v3
        assert v2 != v3
        assert v != v3
        assert v2 != v3

        w = TF(W)
        w2 = TF(W2)
        w3 = TF(W3)
        assert w == w2
        assert w2 == w
        assert w != w3
        assert w2 != w3

        assert v != w
        assert w != v

        s1 = set((v, w))
        s2 = set((v2, w2))
        s3 = set((v, v2, w, w2))
        assert len(s1) == 2
        assert len(s2) == 2
        assert len(s3) == 2
        assert s1 == s2
        assert s1 == s3
        assert s2 == s3

        # Test that the dolfinx implementation of Argument.__eq__ is
        # triggered when comparing ufl expressions
        assert grad(v) == grad(v2)
        assert grad(v) != grad(v3)


def test_cell_mismatch(mesh):
    """Test that cell mismatch raises early enough from UFL"""
    element = FiniteElement("P", triangle, 1)
    with pytest.raises(UFLException):
        FunctionSpace(mesh, element)


def test_basix_element(V, W, Q, V2):
    for V_ in (V, W, V2):
        e = V_.element.basix_element
        assert isinstance(e, basix.finite_element.FiniteElement)

    # Mixed spaces do not yet return a basix element
    with pytest.raises(RuntimeError):
        e = Q.element.basix_element


@pytest.mark.skip_in_parallel
def test_vector_function_space_cell_type():
    """Test that the UFL element cell of a vector function
    space is correct on meshes where gdim > tdim"""
    comm = MPI.COMM_WORLD
    gdim = 2

    # Create a mesh containing a single interval living in 2D
    cell = Cell("interval", geometric_dimension=gdim)
    domain = Mesh(VectorElement("Lagrange", cell, 1))
    cells = np.array([[0, 1]], dtype=np.int64)
    x = np.array([[0., 0.], [1., 1.]])
    mesh = create_mesh(comm, cells, x, domain)

    # Create functions space over mesh, and check element cell
    # is correct
    V = VectorFunctionSpace(mesh, ('Lagrange', 1))
    assert V.ufl_element().cell() == cell
