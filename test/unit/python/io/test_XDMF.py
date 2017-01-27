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
from dolfin_utils.test import skip_in_parallel, fixture, tempdir, skip_if_not_HDF5

# Supported XDMF file encoding
encodings = (XDMFFile.Encoding_HDF5, XDMFFile.Encoding_ASCII)

# Data types supported in templating
data_types = (('int', int), ('size_t', int), ('double', float), ('bool', bool))


def invalid_config(encoding):
    return (not has_hdf5() and encoding == XDMFFile.Encoding_HDF5) \
        or (encoding == XDMFFile.Encoding_ASCII and MPI.size(mpi_comm_world()) > 1)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_1d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitIntervalMesh(32)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mesh, encoding)
    del file

    mesh2 = Mesh()
    file = XDMFFile(mpi_comm_world(), filename)
    file.read(mesh2)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh_2D.xdmf")
    mesh = UnitSquareMesh(32, 32)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mesh, encoding)
    del file

    mesh2 = Mesh()
    file = XDMFFile(mpi_comm_world(), filename)
    file.read(mesh2)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_3d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mesh, encoding)
    del file

    mesh2 = Mesh()
    file = XDMFFile(mpi_comm_world(), filename)
    file.read(mesh2)
    assert mesh.size_global(0) == mesh2.size_global(0)
    dim = mesh.topology().dim()
    assert mesh.size_global(dim) == mesh2.size_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_scalar(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
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
def test_save_2d_scalar(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u2.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "Lagrange", 2)  # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_scalar(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u3.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_vector(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
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
def test_save_3d_vector(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(2, 2, 2)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 1))
    c = Constant((1.0, 2.0, 3.0))
    u.interpolate(c)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector_series(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
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
def test_save_2d_tensor(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "tensor.xdmf")
    mesh = UnitSquareMesh(16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_tensor(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u3t.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(u, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mf_1D.xdmf")
    mesh = UnitIntervalMesh(32)
    mf = CellFunction("size_t", mesh)
    for cell in cells(mesh):
        mf[cell] = cell.index()

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_cell_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    filename = os.path.join(tempdir, "mf_2D_%s.xdmf" % dtype_str)
    mesh = UnitSquareMesh(32, 32)
    mf = CellFunction(dtype_str, mesh)
    mf.rename("cells", "cells")
    for cell in cells(mesh):
        mf[cell] = dtype(cell.index())

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file

    mf_in = CellFunction(dtype_str, mesh)
    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mf_in, "cells")
    del xdmf

    diff = 0
    for cell in cells(mesh):
        diff += (mf_in[cell] - mf[cell])
    assert diff == 0


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_cell_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mf = CellFunction(dtype_str, mesh)
    mf.rename("cells", "cells")
    for cell in cells(mesh):
        mf[cell] = dtype(cell.index())
    filename = os.path.join(tempdir, "mf_3D_%s.xdmf" % dtype_str)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file

    mf_in = CellFunction(dtype_str, mesh)
    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mf_in, "cells")
    del xdmf

    diff = 0
    for cell in cells(mesh):
        diff += (mf_in[cell] - mf[cell])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_facet_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitSquareMesh(32, 32)
    mf = FacetFunction(dtype_str, mesh)
    mf.rename("facets", "facets")

    if (MPI.size(mesh.mpi_comm()) == 1):
        for facet in facets(mesh):
            mf[facet] = dtype(facet.index())
    else:
        for facet in facets(mesh):
            mf[facet] = dtype(facet.global_index())
    filename = os.path.join(tempdir, "mf_facet_2D_%s.xdmf" % dtype_str)

    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.write(mf, encoding)
    del xdmf

    mf_in = FacetFunction(dtype_str, mesh)
    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mf_in, "facets")
    del xdmf

    diff = 0
    for facet in facets(mesh):
        diff += (mf_in[facet] - mf[facet])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_facet_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mf = FacetFunction(dtype_str, mesh)
    mf.rename("facets", "facets")

    if (MPI.size(mesh.mpi_comm()) == 1):
        for facet in facets(mesh):
            mf[facet] = dtype(facet.index())
    else:
        for facet in facets(mesh):
            mf[facet] = dtype(facet.global_index())
    filename = os.path.join(tempdir, "mf_facet_3D_%s.xdmf" % dtype_str)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file

    mf_in = FacetFunction(dtype_str, mesh)
    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mf_in, "facets")
    del xdmf

    diff = 0
    for facet in facets(mesh):
        diff += (mf_in[facet] - mf[facet])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_edge_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mf = EdgeFunction(dtype_str, mesh)
    mf.rename("edges", "edges")
    for edge in edges(mesh):
        mf[edge] = dtype(edge.index())

    filename = os.path.join(tempdir, "mf_edge_3D_%s.xdmf" % dtype_str)
    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_vertex_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitSquareMesh(32, 32)
    mf = VertexFunction(dtype_str, mesh)
    mf.rename("vertices", "vertices")
    for vertex in vertices(mesh):
        mf[vertex] = dtype(vertex.global_index())
    filename = os.path.join(tempdir, "mf_vertex_2D_%s.xdmf" % dtype_str)

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file

    mf_in = VertexFunction(dtype_str, mesh)
    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mf_in, "vertices")
    del xdmf

    diff = 0
    for v in vertices(mesh):
        diff += (mf_in[v] - mf[v])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_vertex_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    filename = os.path.join(tempdir, "mf_vertex_3D_%s.xdmf" % dtype_str)
    mesh = UnitCubeMesh(8, 8, 8)
    mf = VertexFunction(dtype_str, mesh)
    for vertex in vertices(mesh):
        mf[vertex] = dtype(vertex.index())

    file = XDMFFile(mesh.mpi_comm(), filename)
    file.write(mf, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.skipif(True, reason="Missing swig typemap")
def test_save_points_2D(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    import numpy
    mesh = UnitSquareMesh(16, 16)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_2D.xdmf"))
    file.write(points, encoding)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir,
                                                  "points_values_2D.xdmf"))
    file.write(points, vals, encoding)
    del file


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.skipif(True, reason="Missing swig typemap")
def test_save_points_3D(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    import numpy
    mesh = UnitCubeMesh(4, 4, 4)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_3D.xdmf"))
    file.write(points, encoding)

    file = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_values_3D.xdmf"))
    file.write(points, vals, encoding)
    del file

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_mesh_value_collection(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(4, 4, 4)
    tdim = mesh.topology().dim()

    meshfn = CellFunction(dtype_str, mesh, 0)
    meshfn.rename("volume_marker", "Volume Markers")
    for c in cells(mesh):
        if c.midpoint().y() > 0.1:
            meshfn[c] = dtype(1)
        if c.midpoint().y() > 0.9:
            meshfn[c] = dtype(2)

    for mvc_dim in range(0, tdim + 1):
        mvc = MeshValueCollection(dtype_str, mesh, mvc_dim)
        tag = "dim_%d_marker" % mvc_dim
        mvc.rename(tag, "BC")
        mesh.init(mvc_dim, tdim)
        for e in cpp.entities(mesh, mvc_dim):
            if (e.midpoint().x() > 0.5):
                mvc.set_value(e.index(), dtype(1))

        filename = os.path.join(tempdir, "mvc_%d.xdmf" % mvc_dim)

        xdmf = XDMFFile(mesh.mpi_comm(), filename)
        xdmf.write(meshfn, encoding)
        xdmf.write(mvc, encoding)

        del xdmf

        xdmf = XDMFFile(mesh.mpi_comm(), filename)
        mvc = MeshValueCollection(dtype_str, mesh)
        xdmf.read(mvc, tag)

@skip_in_parallel
@pytest.mark.parametrize("encoding", encodings)
@skip_if_not_HDF5
def test_quadratic_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")
    mesh = UnitDiscMesh(mpi_comm_world(), 2, 2, 2)
    Q = FunctionSpace(mesh, "CG", 1)
    u = Function(Q)
    u.interpolate(Constant(1.0))
    xdmf = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "quadratic1.xdmf"))
    xdmf.write(u)

    Q = FunctionSpace(mesh, "CG", 2)
    u = Function(Q)
    u.interpolate(Constant(1.0))
    xdmf = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "quadratic2.xdmf"))
    xdmf.write(u)

    xdmf = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "qmesh.xdmf"))
    xdmf.write(mesh)
    c0 = mesh.coordinates()

    mesh = Mesh()
    xdmf = XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "qmesh.xdmf"))
    xdmf.read(mesh)
    c1 = mesh.coordinates()

    assert (c0 - c1).sum() == 0.0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_append_and_load_mesh_functions(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    meshes = [UnitSquareMesh(32, 32), UnitCubeMesh(8, 8, 8)]

    for mesh in meshes:
        dim = mesh.topology().dim()

        vf = VertexFunction(dtype_str, mesh)
        vf.rename("vertices", "vertices")
        ff = FacetFunction(dtype_str, mesh)
        ff.rename("facets", "facets")
        cf = CellFunction(dtype_str, mesh)
        cf.rename("cells", "cells")

        if (MPI.size(mesh.mpi_comm()) == 1):
            for vertex in vertices(mesh):
                vf[vertex] = dtype(vertex.index())
            for facet in facets(mesh):
                ff[facet] = dtype(facet.index())
            for cell in cells(mesh):
                cf[cell] = dtype(cell.index())
        else:
            for vertex in vertices(mesh):
                vf[vertex] = dtype(vertex.global_index())
            for facet in facets(mesh):
                ff[facet] = dtype(facet.global_index())
            for cell in cells(mesh):
                cf[cell] = dtype(cell.global_index())

        filename = os.path.join(tempdir, "appended_mf_%dD.xdmf" % dim)

        xdmf = XDMFFile(mesh.mpi_comm(), filename)
        xdmf.write(mesh)
        xdmf.write(vf, encoding)
        xdmf.write(ff, encoding)
        xdmf.write(cf, encoding)
        del xdmf

        xdmf = XDMFFile(mesh.mpi_comm(), filename)

        vf_in = VertexFunction(dtype_str, mesh)
        xdmf.read(vf_in, "vertices")
        ff_in = FacetFunction(dtype_str, mesh)
        xdmf.read(ff_in, "facets")
        cf_in = CellFunction(dtype_str, mesh)
        xdmf.read(cf_in, "cells")
        del xdmf

        diff = 0
        for vertex in vertices(mesh):
            diff += (vf_in[vertex] - vf[vertex])
        for facet in facets(mesh):
            diff += (ff_in[facet] - ff[facet])
        for cell in cells(mesh):
            diff += (cf_in[cell] - cf[cell])
        assert diff == 0


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_append_and_load_mesh_value_collections(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.xfail("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mesh.init()
    for d in range(mesh.geometry().dim() + 1):
        mesh.init_global(d)

    mvc_v = MeshValueCollection(dtype_str, mesh, 0)
    mvc_v.rename("vertices", "vertices")
    mvc_e = MeshValueCollection(dtype_str, mesh, 1)
    mvc_e.rename("edges", "edges")
    mvc_f = MeshValueCollection(dtype_str, mesh, 2)
    mvc_f.rename("facets", "facets")
    mvc_c = MeshValueCollection(dtype_str, mesh, 3)
    mvc_c.rename("cells", "cells")

    mvcs = [mvc_v, mvc_e, mvc_f, mvc_c]

    filename = os.path.join(tempdir, "appended_mvcs.xdmf")
    xdmf = XDMFFile(mesh.mpi_comm(), filename)

    for mvc in mvcs:
        for ent in entities(mesh, mvc.dim()):
            assert(mvc.set_value(ent.index(), dtype(ent.global_index())))
        xdmf.write(mvc)

    del xdmf

    mvc_v_in = MeshValueCollection(dtype_str, mesh, 0)
    mvc_e_in = MeshValueCollection(dtype_str, mesh, 1)
    mvc_f_in = MeshValueCollection(dtype_str, mesh, 2)
    mvc_c_in = MeshValueCollection(dtype_str, mesh, 3)

    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mvc_v_in, "vertices")
    xdmf.read(mvc_e_in, "edges")
    xdmf.read(mvc_f_in, "facets")
    xdmf.read(mvc_c_in, "cells")

    mvcs_in = [mvc_v_in, mvc_e_in, mvc_f_in, mvc_c_in]

    for (mvc, mvc_in) in zip(mvcs, mvcs_in):
        mf = MeshFunction(dtype_str, mesh, mvc)
        mf_in = MeshFunction(dtype_str, mesh, mvc_in)

        diff = 0
        for ent in entities(mesh, mf.dim()):
            diff += (mf_in[ent] - mf[ent])
        assert(diff == 0)
