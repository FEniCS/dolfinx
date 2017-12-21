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
from dolfin_utils.test import skip_in_parallel, fixture, tempdir


# Supported XDMF file encoding
encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)

# Data types supported in templating
data_types = (('int', int), ('size_t', int), ('double', float), ('bool', bool))

# Finite elements tested
fe_1d_shapes = ["interval"]
fe_2d_shapes = ["triangle"]
fe_3d_shapes = ["tetrahedron"]
fe_families = ["CG", "DG"]
fe_degrees = [0, 1, 3]
mesh_tdims = [1, 2, 3]
mesh_ns = [4, 11]

# Meshes tested
def mesh_factory(tdim, n):
    if tdim == 1:
        return UnitIntervalMesh(n)
    elif tdim == 2:
        return UnitSquareMesh(n, n)
    elif tdim == 3:
        return UnitCubeMesh(n, n, n)


def invalid_config(encoding):
    return (not has_hdf5() and encoding == XDMFFile.Encoding.HDF5) \
        or (encoding == XDMFFile.Encoding.ASCII and MPI.size(MPI.comm_world) > 1) \
        or (not has_hdf5_parallel() and MPI.size(MPI.comm_world) > 1)


def invalid_fe(fe_family, fe_degree):
    return (fe_family == "CG" and fe_degree == 0)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in
    parallell

    """
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_1d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitIntervalMesh(32)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mesh, encoding)

    mesh2 = Mesh()
    with XDMFFile(MPI.comm_world, filename) as file:
        file.read(mesh2)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology().dim()
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh_2D.xdmf")
    mesh = UnitSquareMesh(32, 32)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mesh, encoding)

    mesh2 = Mesh()
    with XDMFFile(MPI.comm_world, filename) as file:
        file.read(mesh2)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology().dim()
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_quad_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh_2D_quad.xdmf")
    mesh = UnitSquareMesh.create(32, 32, CellType.Type.quadrilateral)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mesh, encoding)

    mesh2 = Mesh()
    with XDMFFile(MPI.comm_world, filename) as file:
        file.read(mesh2)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology().dim()
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_3d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mesh_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mesh, encoding)

    mesh2 = Mesh()
    with XDMFFile(MPI.comm_world, filename) as file:
        file.read(mesh2)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology().dim()
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_scalar(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename1 = os.path.join(tempdir, "u1.xdmf")
    filename2 = os.path.join(tempdir, "u1_.xdmf")
    mesh = UnitIntervalMesh(32)
    V = FunctionSpace(mesh, "Lagrange", 2)  # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    with XDMFFile(mesh.mpi_comm(), filename2) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("fe_degree", fe_degrees)
@pytest.mark.parametrize("fe_family", fe_families)
@pytest.mark.parametrize("mesh_tdim", mesh_tdims)
@pytest.mark.parametrize("mesh_n", mesh_ns)
def test_save_and_checkpoint_scalar(tempdir, encoding, fe_degree, fe_family,
                                    mesh_tdim, mesh_n):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    if invalid_fe(fe_family, fe_degree):
        pytest.skip("Trivial finite element")

    filename = os.path.join(tempdir, "u1_checkpoint.xdmf")
    mesh = mesh_factory(mesh_tdim, mesh_n)
    FE = FiniteElement(fe_family, mesh.ufl_cell(), fe_degree)
    V = FunctionSpace(mesh, FE)
    u_in = Function(V)
    u_out = Function(V)

    u_out.interpolate(Expression("x[0]", degree=1))

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write_checkpoint(u_out, "u_out", 0, encoding)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.read_checkpoint(u_in, "u_out", 0)

    result = u_in.vector() - u_out.vector()
    assert all([near(x, 0.0) for x in result.get_local()])


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("fe_degree", fe_degrees)
@pytest.mark.parametrize("fe_family", fe_families)
@pytest.mark.parametrize("mesh_tdim", mesh_tdims)
@pytest.mark.parametrize("mesh_n", mesh_ns)
def test_save_and_checkpoint_vector(tempdir, encoding, fe_degree, fe_family,
                                    mesh_tdim, mesh_n):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    if invalid_fe(fe_family, fe_degree):
        pytest.skip("Trivial finite element")

    filename = os.path.join(tempdir, "u2_checkpoint.xdmf")
    mesh = mesh_factory(mesh_tdim, mesh_n)
    FE = VectorElement(fe_family, mesh.ufl_cell(), fe_degree)
    V = FunctionSpace(mesh, FE)
    u_in = Function(V)
    u_out = Function(V)

    if mesh.geometry().dim() == 1:
        u_out.interpolate(Expression(("x[0]", ), degree=1))
    elif mesh.geometry().dim() == 2:
        u_out.interpolate(Expression(("x[0]*x[1]", "x[0]"), degree=2))
    elif mesh.geometry().dim() == 3:
        u_out.interpolate(Expression(("x[0]*x[1]", "x[0]", "x[2]"), degree=2))

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write_checkpoint(u_out, "u_out", 0, encoding)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.read_checkpoint(u_in, "u_out", 0)

    result = u_in.vector() - u_out.vector()
    assert all([near(x, 0.0) for x in result.get_local()])


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_checkpoint_timeseries(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    mesh = UnitSquareMesh(16, 16)
    filename = os.path.join(tempdir, "u2_checkpoint.xdmf")
    FE = FiniteElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, FE)

    times = [0.5, 0.2, 0.1]
    u_out = [None]*len(times)
    u_in = [None]*len(times)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        for i, p in enumerate(times):
            u_out[i] = interpolate(Expression("x[0]*p", p=p, degree=1), V)
            file.write_checkpoint(u_out[i], "u_out", p, encoding)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        for i, p in enumerate(times):
            u_in[i] = Function(V)
            file.read_checkpoint(u_in[i], "u_out", i)

    for i, p in enumerate(times):
        result = u_in[i].vector() - u_out[i].vector()
        assert all([near(x, 0.0) for x in result.get_local()])

    # test reading last
    with XDMFFile(mesh.mpi_comm(), filename) as file:
        u_in_last = Function(V)
        file.read_checkpoint(u_in_last, "u_out", -1)

    result = u_out[-1].vector() - u_in_last.vector()
    assert all([near(x, 0.0) for x in result.get_local()])


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_scalar(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u2.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, "Lagrange", 2)  # FIXME: This randomly hangs in parallel
    u = Function(V)
    u.vector()[:] = 1.0

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_scalar(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u3.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    u.vector()[:] = 1.0

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_vector(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u_2dv.xdmf")
    mesh = UnitSquareMesh(16, 16)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    u = Function(V)
    c = Constant((1.0, 2.0))
    u.interpolate(c)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(2, 2, 2)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 1))
    c = Constant((1.0, 2.0, 3.0))
    u.interpolate(c)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector_series(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u_3D.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        u.vector()[:] = 1.0
        file.write(u, 0.1, encoding)

        u.vector()[:] = 2.0
        file.write(u, 0.2, encoding)

        u.vector()[:] = 3.0
        file.write(u, 0.3, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_tensor(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "tensor.xdmf")
    mesh = UnitSquareMesh(16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_tensor(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "u3t.xdmf")
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(u, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    filename = os.path.join(tempdir, "mf_1D.xdmf")
    mesh = UnitIntervalMesh(32)
    mf = MeshFunction("size_t", mesh, mesh.topology().dim())
    for cell in cells(mesh):
        mf[cell] = cell.index()

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_cell_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    filename = os.path.join(tempdir, "mf_2D_%s.xdmf" % dtype_str)
    mesh = UnitSquareMesh(32, 32)
    mf = MeshFunction(dtype_str, mesh, mesh.topology().dim())
    mf.rename("cells", "cells")
    for cell in cells(mesh):
        mf[cell] = dtype(cell.index())

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)

    mf_in = MeshFunction(dtype_str, mesh, mesh.topology().dim())
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.read(mf_in, "cells")

    diff = 0
    for cell in cells(mesh):
        diff += (mf_in[cell] - mf[cell])
    assert diff == 0


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_cell_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mf = MeshFunction(dtype_str, mesh, mesh.topology().dim())
    mf.rename("cells", "cells")
    for cell in cells(mesh):
        mf[cell] = dtype(cell.index())
    filename = os.path.join(tempdir, "mf_3D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)

    mf_in = MeshFunction(dtype_str, mesh, mesh.topology().dim())
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.read(mf_in, "cells")

    diff = 0
    for cell in cells(mesh):
        diff += (mf_in[cell] - mf[cell])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_facet_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitSquareMesh(32, 32)
    mf = MeshFunction(dtype_str, mesh, mesh.topology().dim()-1)
    mf.rename("facets", "facets")

    if (MPI.size(mesh.mpi_comm()) == 1):
        for facet in facets(mesh):
            mf[facet] = dtype(facet.index())
    else:
        for facet in facets(mesh):
            mf[facet] = dtype(facet.global_index())
    filename = os.path.join(tempdir, "mf_facet_2D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.write(mf, encoding)

    mf_in = MeshFunction(dtype_str, mesh, mesh.topology().dim()-1)
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.read(mf_in, "facets")

    diff = 0
    for facet in facets(mesh):
        diff += (mf_in[facet] - mf[facet])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_facet_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mf = MeshFunction(dtype_str, mesh, mesh.topology().dim()-1)
    mf.rename("facets", "facets")

    if (MPI.size(mesh.mpi_comm()) == 1):
        for facet in facets(mesh):
            mf[facet] = dtype(facet.index())
    else:
        for facet in facets(mesh):
            mf[facet] = dtype(facet.global_index())
    filename = os.path.join(tempdir, "mf_facet_3D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)

    mf_in = MeshFunction(dtype_str, mesh, mesh.topology().dim()-1)
    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.read(mf_in, "facets")

    diff = 0
    for facet in facets(mesh):
        diff += (mf_in[facet] - mf[facet])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_edge_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(8, 8, 8)
    mf = MeshFunction(dtype_str, mesh, 1)
    mf.rename("edges", "edges")
    for edge in edges(mesh):
        mf[edge] = dtype(edge.index())

    filename = os.path.join(tempdir, "mf_edge_3D_%s.xdmf" % dtype_str)
    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_vertex_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitSquareMesh(32, 32)
    mf = MeshFunction(dtype_str, mesh, 0)
    mf.rename("vertices", "vertices")
    for vertex in vertices(mesh):
        mf[vertex] = dtype(vertex.global_index())
    filename = os.path.join(tempdir, "mf_vertex_2D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)

    mf_in = MeshFunction(dtype_str, mesh, 0)
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.read(mf_in, "vertices")

    diff = 0
    for v in vertices(mesh):
        diff += (mf_in[v] - mf[v])
    assert diff == 0

@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_vertex_function(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    filename = os.path.join(tempdir, "mf_vertex_3D_%s.xdmf" % dtype_str)
    mesh = UnitCubeMesh(8, 8, 8)
    mf = MeshFunction(dtype_str, mesh, 0)
    for vertex in vertices(mesh):
        mf[vertex] = dtype(vertex.index())

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        file.write(mf, encoding)

@pytest.mark.parametrize("encoding", encodings)
def test_save_points_2D(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    import numpy
    mesh = UnitSquareMesh(16, 16)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_2D.xdmf")) as file:
        file.write(points, encoding)

    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir,
                  "points_values_2D.xdmf")) as file:
        file.write(points, vals, encoding)


@pytest.mark.parametrize("encoding", encodings)
def test_save_points_3D(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    import numpy
    mesh = UnitCubeMesh(4, 4, 4)
    points, values = [], []
    for v in vertices(mesh):
        points.append(v.point())
        values.append(v.point().norm())
    vals = numpy.array(values)

    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_3D.xdmf")) as file:
        file.write(points, encoding)

    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "points_values_3D.xdmf")) as file:
        file.write(points, vals, encoding)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_mesh_value_collection(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    mesh = UnitCubeMesh(4, 4, 4)
    tdim = mesh.topology().dim()

    meshfn = MeshFunction(dtype_str, mesh, mesh.topology().dim(), False)
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
        for e in entities(mesh, mvc_dim):
            if (e.midpoint().x() > 0.5):
                mvc.set_value(e.index(), dtype(1))

        filename = os.path.join(tempdir, "mvc_%d.xdmf" % mvc_dim)

        with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
            xdmf.write(meshfn, encoding)
            xdmf.write(mvc, encoding)


        with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
            mvc = MeshValueCollection(dtype_str, mesh)
            xdmf.read(mvc, tag)


@skip_in_parallel
@pytest.mark.parametrize("encoding", encodings)
def test_quadratic_mesh(tempdir, encoding):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")
    mesh = UnitDiscMesh.create(MPI.comm_world, 2, 2, 2)
    Q = FunctionSpace(mesh, "CG", 1)
    u = Function(Q)
    u.interpolate(Constant(1.0))
    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "quadratic1.xdmf")) as xdmf:
        xdmf.write(u)

    Q = FunctionSpace(mesh, "CG", 2)
    u = Function(Q)
    u.interpolate(Constant(1.0))
    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "quadratic2.xdmf")) as xdmf:
        xdmf.write(u)

    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "qmesh.xdmf")) as xdmf:
        xdmf.write(mesh)
    c0 = mesh.coordinates()

    mesh = Mesh()
    with XDMFFile(mesh.mpi_comm(), os.path.join(tempdir, "qmesh.xdmf")) as xdmf:
        xdmf.read(mesh)
    c1 = mesh.coordinates()

    assert (c0 - c1).sum() == 0.0


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_append_and_load_mesh_functions(tempdir, encoding, data_type):
    if invalid_config(encoding):
        pytest.skip("XDMF unsupported in current configuration")

    dtype_str, dtype = data_type

    meshes = [UnitSquareMesh(32, 32), UnitCubeMesh(8, 8, 8)]

    for mesh in meshes:
        dim = mesh.topology().dim()

        vf = MeshFunction(dtype_str, mesh, 0)
        vf.rename("vertices", "vertices")
        ff = MeshFunction(dtype_str, mesh, mesh.topology().dim()-1)
        ff.rename("facets", "facets")
        cf = MeshFunction(dtype_str, mesh, mesh.topology().dim())
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

        with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
            xdmf.write(mesh)
            xdmf.write(vf, encoding)
            xdmf.write(ff, encoding)
            xdmf.write(cf, encoding)

        with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
            vf_in = MeshFunction(dtype_str, mesh, 0)
            xdmf.read(vf_in, "vertices")
            ff_in = MeshFunction(dtype_str, mesh, mesh.topology().dim()-1)
            xdmf.read(ff_in, "facets")
            cf_in = MeshFunction(dtype_str, mesh, mesh.topology().dim())
            xdmf.read(cf_in, "cells")

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
        pytest.skip("XDMF unsupported in current configuration")

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
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        for mvc in mvcs:
            for ent in entities(mesh, mvc.dim()):
                assert(mvc.set_value(ent.index(), dtype(ent.global_index())))
            xdmf.write(mvc)

    mvc_v_in = MeshValueCollection(dtype_str, mesh, 0)
    mvc_e_in = MeshValueCollection(dtype_str, mesh, 1)
    mvc_f_in = MeshValueCollection(dtype_str, mesh, 2)
    mvc_c_in = MeshValueCollection(dtype_str, mesh, 3)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
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
