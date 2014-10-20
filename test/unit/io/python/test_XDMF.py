#!/usr/bin/env py.test

# Copyright (C) 2012 Garth N. Wells
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
import os
from dolfin import *
from dolfin_utils.test import skip_if_not_HDF5, skip_in_parallel, fixture, temppath


@skip_if_not_HDF5
def test_save_and_load_1d_mesh(temppath):
    filename = os.path.join(temppath, "mesh.xdmf")
    mesh = UnitIntervalMesh(32)

    file = File(filename)
    file << mesh
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mesh
    del file

    mesh2 = Mesh(filename)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)

@skip_if_not_HDF5
def test_save_and_load_2d_mesh(temppath):
    filename = os.path.join(temppath, "mesh_2D.xdmf")
    mesh = UnitSquareMesh(32, 32)

    file = File(filename)
    file << mesh
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mesh
    del file

    mesh2 = Mesh(filename)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)

@skip_if_not_HDF5
def test_save_and_load_3d_mesh(temppath):
    filename = os.path.join(temppath, "mesh_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)

    file = File(filename)
    file << mesh
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mesh
    del file

    mesh2 = Mesh(filename)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)

@skip_if_not_HDF5
def test_save_1d_scalar(temppath):
    filename1 = os.path.join(temppath, "u1.xdmf")
    filename2 = os.path.join(temppath, "u1_.xdmf")
    mesh = UnitIntervalMesh(32)
    V = FunctionSpace(mesh, "Lagrange", 2) # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    file = File(filename1)
    file << u
    del file

    file = XDMFFile(mesh.mpi_comm(), filename2)
    file << u
    del file

@skip_if_not_HDF5
def test_save_2d_scalar(temppath):
    filename = os.path.join(temppath, "u2.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "Lagrange", 2)  # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    file = File(mesh.mpi_comm(), filename)
    file << u
    del file
    file = XDMFFile(mesh.mpi_comm(), filename)
    file << u
    del file

@skip_if_not_HDF5
def test_save_3d_scalar(temppath):
    filename = os.path.join(temppath, "u3.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    u.vector()[:] = 1.0
    file = File(mesh.mpi_comm(), filename)
    file << u
    del file
    file = XDMFFile(mesh.mpi_comm(), filename)
    file << u
    del file

@skip_if_not_HDF5
def test_save_2d_vector(temppath):
    filename = os.path.join(temppath, "u_2dv.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    c = Constant((1.0, 2.0))
    u.interpolate(c)
    file = File(mesh.mpi_comm(), filename)
    file << u
    del file
    file = XDMFFile(mesh.mpi_comm(), filename)
    file << u
    del file

@skip_if_not_HDF5
def test_save_3d_vector(temppath):
    filename = os.path.join(temppath, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(1, 1, 1)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 1))
    c = Constant((1.0, 2.0, 3.0))
    u.interpolate(c)
    file = File(mesh.mpi_comm(), filename)
    file << u
    del file
    file = XDMFFile(mesh.mpi_comm(), filename)
    file << u
    del file

@skip_if_not_HDF5
def test_save_3d_vector_series(temppath):
    filename = os.path.join(temppath, "u_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    file = File(mesh.mpi_comm(), filename)

    u.vector()[:] = 1.0
    file << (u, 0.1)

    u.vector()[:] = 2.0
    file << (u, 0.2)

    u.vector()[:] = 3.0
    file << (u, 0.3)

    del file

    file = XDMFFile(mesh.mpi_comm(), filename)

    u.vector()[:] = 1.0
    file << (u, 0.1)

    u.vector()[:] = 2.0
    file << (u, 0.2)

    u.vector()[:] = 3.0
    file << (u, 0.3)

    del file

@skip_if_not_HDF5
def test_save_2d_tensor(temppath):
    filename = os.path.join(temppath, "tensor.xdmf")
    mesh = UnitSquareMesh(16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    file = File(mesh.mpi_comm(), filename)
    file << u
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << u
    del file

@skip_if_not_HDF5
def test_save_3d_tensor(temppath):
    filename = os.path.join(temppath, "u3t.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    file = File(mesh.mpi_comm(), filename)
    file << u
    del file
    file = XDMFFile(mesh.mpi_comm(), filename)
    file << u
    del file

@skip_if_not_HDF5
def test_save_1d_mesh(temppath):
    filename = os.path.join(temppath, "mf_1D.xdmf")
    mesh = UnitIntervalMesh(32)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_2D_cell_function(temppath):
    filename = os.path.join(temppath, "mf_2D.xdmf")
    mesh = UnitSquareMesh(32, 32)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_3D_cell_function(temppath):
    mesh = UnitCubeMesh(8, 8, 8)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()
    filename = os.path.join(temppath, "mf_3D.xdmf")

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_2D_facet_function(temppath):
    mesh = UnitSquareMesh(32, 32)
    mf = FacetFunction("size_t", mesh)
    for facet in facets(mesh):
        mf[facet] = facet.index()
    filename = os.path.join(temppath, "mf_facet_2D.xdmf")

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_3D_facet_function(temppath):
    mesh = UnitCubeMesh(8, 8, 8)
    mf = FacetFunction("size_t", mesh)
    for facet in facets(mesh):
        mf[facet] = facet.index()
    filename = os.path.join(temppath, "mf_facet_3D.xdmf")

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_3D_edge_function(temppath):
    mesh = UnitCubeMesh(8, 8, 8)
    mf = EdgeFunction("size_t", mesh)
    for edge in edges(mesh):
        mf[edge] = edge.index()

    file = File(mesh.mpi_comm(), os.path.join(temppath, "mf_edge_3D.xdmf"))
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), os.path.join(temppath, "mf_edge_3D.xdmf"))
    file << mf
    del file

@skip_if_not_HDF5
def test_save_2D_vertex_function(temppath):
    mesh = UnitSquareMesh(32, 32)
    mf = VertexFunction("size_t", mesh)
    for vertex in vertices(mesh):
        mf[vertex] = vertex.index()
    filename = os.path.join(temppath, "mf_vertex_2D.xdmf")

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_3D_vertex_function(temppath):
    filename = os.path.join(temppath, "mf_vertex_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    mf = VertexFunction("size_t", mesh)
    for vertex in vertices(mesh):
        mf[vertex] = vertex.index()

    file = File(mesh.mpi_comm(), filename)
    file << mf
    del file

    file = XDMFFile(mesh.mpi_comm(), filename)
    file << mf
    del file

@skip_if_not_HDF5
def test_save_points_2D(temppath):
    import numpy
    mesh = UnitSquareMesh(16, 16)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(temppath, "points_2D.xdmf"))
    file.write(points)
    del file

    file = XDMFFile(mesh.mpi_comm(), os.path.join(temppath, "points_values_2D.xdmf"))
    file.write(points, vals)
    del file

@skip_if_not_HDF5
def test_save_points_3D(temppath):
    import numpy
    mesh = UnitCubeMesh(4, 4, 4)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(temppath, "points_3D.xdmf"))
    file.write(points)
    del file

    file = XDMFFile(mesh.mpi_comm(), os.path.join(temppath, "points_values_3D.xdmf"))
    file.write(points, vals)
    del file
