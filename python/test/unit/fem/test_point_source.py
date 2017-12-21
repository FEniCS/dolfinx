"""Unit tests for PointSources"""

# Copyright (C) 2016 Ettie Unwin
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
from dolfin import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
                    Point, MPI, FunctionSpace, TestFunction,
                    TrialFunction, assemble, Constant, PointSource,
                    dx, Cell, VectorFunctionSpace, near,
                    vertex_to_dof_map, vertices, dot, FiniteElement,
                    Function, VectorElement, MixedElement)

meshes = [UnitIntervalMesh(10), UnitSquareMesh(10, 10), UnitCubeMesh(4, 4, 4)]
data = [(UnitIntervalMesh(10), Point(0.5)),
        (UnitSquareMesh(10, 10), Point(0.5, 0.5)),
        (UnitCubeMesh(3, 3, 3), Point(0.5, 0.5, 0.5))]


@pytest.mark.parametrize("mesh, point", data)
def test_pointsource_vector_node(mesh, point):
    """Tests point source when given constructor PointSource(V, point,
    mag) with a vector and when placed at a node for 1D, 2D and
    3D. Global points given to constructor from rank 0 processor.

    """

    rank = MPI.rank(mesh.mpi_comm())
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(Constant(0.0)*v*dx)
    if rank == 0:
        ps = PointSource(V, point, 10.0)
    else:
        ps = PointSource(V, [])
    ps.apply(b)

    # Checks array sums to correct value
    b_sum = b.sum()
    assert round(b_sum - 10.0) == 0

    # Checks point source is added to correct part of the array
    v2d = vertex_to_dof_map(V)
    for v in vertices(mesh):
        if near(v.midpoint().distance(point), 0.0):
            ind = v2d[v.index()]
            if ind < len(b.get_local()):
                assert np.round(b.get_local()[ind]-10.0) == 0


@pytest.mark.parametrize("mesh", meshes)
def test_pointsource_vector(mesh):
    """Tests point source when given constructor PointSource(V, point,
    mag) with a vector that isn't placed at a node for 1D, 2D and
    3D. Global points given to constructor from rank 0 processor

    """

    cell = Cell(mesh, 0)
    point = cell.midpoint()
    rank = MPI.rank(mesh.mpi_comm())

    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(Constant(0.0)*v*dx)
    if rank == 0:
        ps = PointSource(V, point, 10.0)
    else:
        ps = PointSource(V, [])
    ps.apply(b)

    # Checks array sums to correct value
    b_sum = b.sum()
    assert round(b_sum - 10.0) == 0


@pytest.mark.parametrize("mesh, point", data)
def test_pointsource_vector_fs(mesh, point):
    """Tests point source when given constructor PointSource(V, point,
    mag) with a vector for a vector function space that isn't placed
    at a node for 1D, 2D and 3D. Global points given to constructor
    from rank 0 processor.

    """

    rank = MPI.rank(mesh.mpi_comm())
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(dot(Constant([0.0]*mesh.geometry().dim()), v)*dx)
    if rank == 0:
        ps = PointSource(V, point, 10.0)
    else:
        ps = PointSource(V, [])
    ps.apply(b)

    # Checks array sums to correct value
    b_sum = b.sum()
    assert round(b_sum - 10.0*V.num_sub_spaces()) == 0

    # Checks point source is added to correct part of the array
    v2d = vertex_to_dof_map(V)
    for v in vertices(mesh):
        if near(v.midpoint().distance(point), 0.0):
            for spc_idx in range(V.num_sub_spaces()):
                ind = v2d[v.index()*V.num_sub_spaces() + spc_idx]
                if ind < len(b.get_local()):
                    assert np.round(b.get_local()[ind] - 10.0) == 0


@pytest.mark.parametrize("mesh, point", data)
def test_pointsource_mixed_space(mesh, point):
    """Tests point source when given constructor PointSource(V, point,
    mag) with a vector for a mixed function space that isn't placed at
    a node for 1D, 2D and 3D. Global points given to constructor from
    rank 0 processor.

    """

    rank = MPI.rank(mesh.mpi_comm())
    ele1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    ele2 = FiniteElement("DG", mesh.ufl_cell(), 2)
    ele3 = VectorElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, MixedElement([ele1, ele2, ele3]))
    value_dimension = V.element().value_dimension(0)
    v = TestFunction(V)
    b = assemble(dot(Constant([0.0]*value_dimension), v)*dx)
    if rank == 0:
        ps = PointSource(V, point, 10.0)
    else:
        ps = PointSource(V, [])
    ps.apply(b)

    # Checks array sums to correct value
    b_sum = b.sum()
    assert round(b_sum - 10.0*value_dimension) == 0


def test_point_outside():
    """Tests point source fails if given a point outside the domain."""
    mesh = UnitIntervalMesh(10)
    point = Point(1.2)
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    assemble(Constant(0.0)*v*dx)
    # Runtime Error is only produced on one process which causes the
    # whole function to fail but makes this test hang in parallel.
    with pytest.raises(RuntimeError):
        PointSource(V, point, 10.0)


@pytest.mark.parametrize("mesh, point", data)
def test_pointsource_matrix(mesh, point):
    """Tests point source when given constructor PointSource(V, point,
    mag) with a matrix and when placed at a node for 1D, 2D and
    3D. Global points given to constructor from rank 0 processor.

    """

    rank = MPI.rank(mesh.mpi_comm())
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V)
    A = assemble(Constant(0.0)*u*v*dx)
    if rank == 0:
        ps = PointSource(V, point, 10.0)
    else:
        ps = PointSource(V, [])
    ps.apply(A)

    # Checks array sums to correct value
    a_sum = MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
    assert round(a_sum - 10.0) == 0

    # Checks point source is added to correct part of the array
    A.get_diagonal(w.vector())
    v2d = vertex_to_dof_map(V)
    for v in vertices(mesh):
        if near(v.midpoint().distance(point), 0.0):
            ind = v2d[v.index()]
            if ind < len(A.array()):
                assert np.round(w.vector()[ind] - 10.0) == 0


# FIXME: Edit this test to have V1 != V2 when that is implemented
@pytest.mark.parametrize("mesh, point", data)
def test_pointsource_matrix_second_constructor(mesh, point):
    """Tests point source when given different constructor PointSource(V1,
    V2, point, mag) with a matrix and when placed at a node for 1D, 2D
    and 3D. Global points given to constructor from rank 0
    processor. Currently only implemented if V1=V2.

    """

    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 1)

    rank = MPI.rank(mesh.mpi_comm())
    u, v = TrialFunction(V1), TestFunction(V2)
    w = Function(V1)
    A = assemble(Constant(0.0)*u*v*dx)
    if rank == 0:
        ps = PointSource(V1, V2, point, 10.0)
    else:
        ps = PointSource(V1, V2, [])
    ps.apply(A)

    # Checks array sums to correct value
    a_sum = MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
    assert round(a_sum - 10.0) == 0

    # Checks point source is added to correct part of the array
    A.get_diagonal(w.vector())
    v2d = vertex_to_dof_map(V1)
    for v in vertices(mesh):
        if near(v.midpoint().distance(point), 0.0):
            ind = v2d[v.index()]
            if ind < len(A.array()):
                assert np.round(w.vector()[ind] - 10.0) == 0


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_vector_node(mesh):
    """Tests point source when given constructor PointSource(V, V, point,
    mag) with a matrix when points placed at 3 node for 1D, 2D and
    3D. Global points given to constructor from rank 0 processor.

    """

    point = [0.0, 0.5, 1.0]
    dim = mesh.geometry().dim()
    rank = MPI.rank(mesh.mpi_comm())
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(Constant(0.0)*v*dx)

    source = []
    point_coords = np.zeros(dim)
    for p in point:
        for i in range(dim):
            point_coords[i-1] = p
        if rank == 0:
            source.append((Point(point_coords), 10.0))
    ps = PointSource(V, source)
    ps.apply(b)

    # Checks b sums to correct value
    b_sum = b.sum()
    assert round(b_sum - len(point)*10.0) == 0

    # Checks values added to correct part of vector
    mesh_coords = V.tabulate_dof_coordinates()
    for p in point:
        for i in range(dim):
            point_coords[i] = p

        j = 0
        for i in range(len(mesh_coords)//(dim)):
            mesh_coords_check = mesh_coords[j:j + dim - 1]
            if np.array_equal(point_coords, mesh_coords_check) is True:
                assert np.round(b.array()[j//(dim)]-10.0) == 0.0
            j += dim


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_vector_node_local(mesh):
    """Tests point source when given constructor PointSource(V, V, point,
    mag) with a matrix when points placed at 3 node for 1D, 2D and
    3D. Local points given to constructor.

    """

    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(Constant(0.0)*v*dx)

    source = []
    point_coords = mesh.coordinates()[0]
    source.append((Point(point_coords), 10.0))
    ps = PointSource(V, source)
    ps.apply(b)

    # Checks b sums to correct value
    size = MPI.size(mesh.mpi_comm())
    b_sum = b.sum()
    assert round(b_sum - size*10.0) == 0


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_vector(mesh):
    """Tests point source PointSource(V, source) for mulitple point
    sources applied to a vector for 1D, 2D and 3D. Global points given
    to constructor from rank 0 processor.

    """

    c_ids = [0, 1, 2]
    rank = MPI.rank(mesh.mpi_comm())
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    b = assemble(Constant(0.0)*v*dx)

    source = []
    if rank == 0:
        for c_id in c_ids:
            cell = Cell(mesh, c_id)
            point = cell.midpoint()
            source.append((point, 10.0))
    ps = PointSource(V, source)
    ps.apply(b)

    # Checks b sums to correct value
    b_sum = b.sum()
    assert round(b_sum - len(c_ids)*10.0) == 0


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_matrix_node(mesh):
    """Tests point source when given constructor PointSource(V, source)
    with a matrix when points placed at 3 nodes for 1D, 2D and
    3D. Global points given to constructor from rank 0 processor.

    """

    point = [0.0, 0.5, 1.0]
    rank = MPI.rank(mesh.mpi_comm())
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V)
    A = assemble(Constant(0.0)*u*v*dx)
    dim = mesh.geometry().dim()

    source = []

    point_coords = np.zeros(dim)
    for p in point:
        for i in range(dim):
            point_coords[i-1] = p
        if rank == 0:
            source.append((Point(point_coords), 10.0))
    ps = PointSource(V, source)
    ps.apply(A)

    # Checks matrix sums to correct value.
    A.get_diagonal(w.vector())
    a_sum = MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
    assert round(a_sum - len(point)*10) == 0

    # Check if coordinates are in portion of mesh and if so check that
    # diagonal components sum to the correct value.
    mesh_coords = V.tabulate_dof_coordinates()
    for p in point:
        for i in range(dim):
            point_coords[i-1] = p

        j = 0
        for i in range(len(mesh_coords)//(dim)):
            mesh_coords_check = mesh_coords[j:j+dim-1]
            if np.array_equal(point_coords, mesh_coords_check) is True:
                assert np.round(w.vector()[j//(dim)]-10.0) == 0.0
            j += dim


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_matrix_node_local(mesh):
    """Tests point source when given constructor PointSource(V, V, point,
    mag) with a matrix when points placed at 3 node for 1D, 2D and
    3D. Local points given to constructor.

    """

    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V)
    A = assemble(Constant(0.0)*u*v*dx)

    source = []
    point_coords = mesh.coordinates()[0]
    source.append((Point(point_coords), 10.0))
    ps = PointSource(V, source)
    ps.apply(A)

    # Checks matrix sums to correct value.
    A.get_diagonal(w.vector())
    size = MPI.size(mesh.mpi_comm())
    a_sum = MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
    assert round(a_sum - size*10.0) == 0


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_matrix_node_vector_fs(mesh):
    """Tests point source applied to a matrix with given constructor
    PointSource(V, source) and a vector function space when points
    placed at 3 vertices for 1D, 2D and 3D. Global points given to
    constructor from rank 0 processor.

    """

    point = [0.0, 0.5, 1.0]
    rank = MPI.rank(mesh.mpi_comm())
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    u, v = TrialFunction(V), TestFunction(V)
    w = Function(V)
    A = assemble(Constant(0.0)*dot(u, v)*dx)
    dim = mesh.geometry().dim()

    source = []
    point_coords = np.zeros(dim)
    for p in point:
        for i in range(dim):
            point_coords[i - 1] = p
        if rank == 0:
            source.append((Point(point_coords), 10.0))
    ps = PointSource(V, source)
    ps.apply(A)

    # Checks array sums to correct value
    A.get_diagonal(w.vector())
    a_sum = MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
    assert round(a_sum - 2*len(point)*10) == 0

    # Check if coordinates are in portion of mesh and if so check that
    # diagonal components sum to the correct value.
    mesh_coords = V.tabulate_dof_coordinates()
    for p in point:
        for i in range(dim):
            point_coords[i] = p

        j = 0
        for i in range(len(mesh_coords)//(dim)):
            mesh_coords_check = mesh_coords[j:j+dim-1]
            if np.array_equal(point_coords, mesh_coords_check) is True:
                assert np.round(w.vector()[j//(dim)] - 10.0) == 0.0
            j += dim


@pytest.mark.parametrize("mesh", meshes)
def test_multi_ps_matrix(mesh):
    """Tests point source PointSource(V, source) for mulitple point
    sources applied to a matrix for 1D, 2D and 3D. Global points given
    to constructor from rank 0 processor.

    """

    c_ids = [0, 1, 2]
    rank = MPI.rank(mesh.mpi_comm())
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(Constant(0.0)*dot(u, v)*dx)

    source = []
    if rank == 0:
        for c_id in c_ids:
            cell = Cell(mesh, c_id)
            point = cell.midpoint()
            source.append((point, 10.0))
    ps = PointSource(V, source)
    ps.apply(A)

    # Checks b sums to correct value
    a_sum = MPI.sum(mesh.mpi_comm(), np.sum(A.array()))
    assert round(a_sum - 2*len(c_ids)*10) == 0
