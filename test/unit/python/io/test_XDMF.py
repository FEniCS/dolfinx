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
from dolfin_utils.test import skip_if_not_HDF5, skip_in_parallel, fixture, tempdir

# Currently supported XDMF file encoding
encodings = (XDMFFile.Encoding_HDF5, XDMFFile.Encoding_ASCII)

# Catch the cases here where we expect dolfin to throw and error
# E.g: 
#   XDMF ASCII in parallel
#   No HDF5 support and requested HDF5 encoding
def ensure_errors_for_special_cases(func):
    def fails_decorator(tempdir, encoding):
        if not has_hdf5() and encoding == XDMFFile.Encoding_HDF5:
            with pytest.raises(Exception):
                fname = func.__name__
                func(tempdir, encoding)
        elif encoding == XDMFFile.Encoding_ASCII and MPI.size(mpi_comm_world()) > 1:
            with pytest.raises(RuntimeError):
                fname = func.__name__
                func(tempdir, encoding)
        else:
            func(tempdir, encoding)
    return fails_decorator

@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_and_load_1d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitIntervalMesh(32)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mesh, encoding)
    del file

    mesh2 = Mesh()
    file = XDMFFile(mpi_comm_world(), filename)
    file.read(mesh2, False)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_and_load_2d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh_2D.xdmf")
    mesh = UnitSquareMesh(32, 32)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mesh, encoding)
    del file

    mesh2 = Mesh()
    file = XDMFFile(mpi_comm_world(), filename)
    file.read(mesh2, False)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_and_load_3d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mesh, encoding)
    del file

    mesh2 = Mesh()
    file = XDMFFile(mpi_comm_world(), filename)
    file.read(mesh2, False)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_1d_scalar(tempdir, encoding):
    filename1 = os.path.join(tempdir, "u1.xdmf")
    filename2 = os.path.join(tempdir, "u1_.xdmf")
    mesh = UnitIntervalMesh(32)
    V = FunctionSpace(mesh, "Lagrange", 2)  # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename2)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_2d_scalar(tempdir, encoding):
    filename = os.path.join(tempdir, "u2.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "Lagrange", 2)  # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3d_scalar(tempdir, encoding):
    filename = os.path.join(tempdir, "u3.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_2d_vector(tempdir, encoding):
    filename = os.path.join(tempdir, "u_2dv.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    c = Constant((1.0, 2.0))
    u.interpolate(c)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3d_vector(tempdir, encoding):
    filename = os.path.join(tempdir, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(1, 1, 1)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 1))
    c = Constant((1.0, 2.0, 3.0))
    u.interpolate(c)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3d_vector_series(tempdir, encoding):
    filename = os.path.join(tempdir, "u_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))

    file = XDMFFile(mesh.mpi_comm(), filename)

    u.vector()[:] = 1.0
    file.write(u, 0.1, encoding)

    u.vector()[:] = 2.0
    file.write(u, 0.2, encoding)

    u.vector()[:] = 3.0
    file.write(u, 0.3, encoding)

    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_2d_tensor(tempdir, encoding):
    filename = os.path.join(tempdir, "tensor.xdmf")
    mesh = UnitSquareMesh(16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3d_tensor(tempdir, encoding):
    filename = os.path.join(tempdir, "u3t.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_1d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mf_1D.xdmf")
    mesh = UnitIntervalMesh(32)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_2D_cell_function(tempdir, encoding):
    filename = os.path.join(tempdir, "mf_2D.xdmf")
    mesh = UnitSquareMesh(32, 32)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3D_cell_function(tempdir, encoding):
    mesh = UnitCubeMesh(8, 8, 8)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()
    filename = os.path.join(tempdir, "mf_3D.xdmf")

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_2D_facet_function(tempdir, encoding):
    mesh = UnitSquareMesh(32, 32)
    mf = FacetFunction("size_t", mesh)
    for facet in facets(mesh):
        mf[facet] = facet.index()
    filename = os.path.join(tempdir, "mf_facet_2D.xdmf")

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3D_facet_function(tempdir, encoding):
    mesh = UnitCubeMesh(8, 8, 8)
    mf = FacetFunction("size_t", mesh)
    for facet in facets(mesh):
        mf[facet] = facet.index()
    filename = os.path.join(tempdir, "mf_facet_3D.xdmf")

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3D_edge_function(tempdir, encoding):
    mesh = UnitCubeMesh(8, 8, 8)
    mf = EdgeFunction("size_t", mesh)
    for edge in edges(mesh):
        mf[edge] = edge.index()

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "mf_edge_3D.xdmf"))
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_2D_vertex_function(tempdir, encoding):
    mesh = UnitSquareMesh(32, 32)
    mf = VertexFunction("size_t", mesh)
    for vertex in vertices(mesh):
        mf[vertex] = vertex.index()
    filename = os.path.join(tempdir, "mf_vertex_2D.xdmf")

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_3D_vertex_function(tempdir, encoding):
    filename = os.path.join(tempdir, "mf_vertex_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    mf = VertexFunction("size_t", mesh)
    for vertex in vertices(mesh):
        mf[vertex] = vertex.index()

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_points_2D(tempdir, encoding):
    import numpy
    mesh = UnitSquareMesh(16, 16)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_2D.xdmf"))
    if encoding == XDMFFile.Encoding_ASCII:
        with pytest.raises(RuntimeError):
            file.write(points, encoding)
    else:
        file.write(points, encoding)
    del file

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir,
                                                  "points_values_2D.xdmf"))
    if encoding == XDMFFile.Encoding_ASCII:
        with pytest.raises(RuntimeError):
            file.write(points, encoding)
    else:
        file.write(points, vals, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_points_3D(tempdir, encoding):
    import numpy
    mesh = UnitCubeMesh(4, 4, 4)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_3D.xdmf"))
    if encoding == XDMFFile.Encoding_ASCII:
        with pytest.raises(RuntimeError):
            file.write(points, encoding)
    else:
        file.write(points, encoding)
    del file

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_values_3D.xdmf"))
    if encoding == XDMFFile.Encoding_ASCII:
        with pytest.raises(RuntimeError):
            file.write(points, encoding)
    else:
        file.write(points, vals, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@ensure_errors_for_special_cases
def test_save_mesh_value_collection(tempdir, encoding):
    mesh = UnitCubeMesh(4, 4, 4)
    tdim = mesh.topology().dim()

    meshfn = CellFunction("size_t", mesh, 0)
    meshfn.rename("volume_marker", "Volume Markers")
    for c in cells(mesh):
        if c.midpoint().y() > 0.1:
            meshfn[c] = 1
        if c.midpoint().y() > 0.9:
            meshfn[c] = 2

    for mvc_dim in range(0, tdim + 1):
        mvc = MeshValueCollection("size_t", mesh, 2)
        mvc.rename("dim_%d_marker" % mvc_dim, "BC")
        mesh.init(mvc_dim, tdim)
        for e in cpp.entities(mesh, mvc_dim):
            if (e.midpoint().x() > 0.5):
                mvc.set_value(e.index(), 1)

        xdmf = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "mvc_%d.xdmf"
                                                      % mvc_dim))
        xdmf.parameters['time_series'] = False
        xdmf.write(meshfn, encoding)
        xdmf.write(mvc, encoding)
