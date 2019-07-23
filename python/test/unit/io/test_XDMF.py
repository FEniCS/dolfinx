# Copyright (C) 2012 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy
import pytest
from dolfin_utils.test.fixtures import tempdir

from dolfin import (MPI, Cell, Facet, Function, FunctionSpace, MeshEntities,
                    MeshFunction, MeshValueCollection, TensorFunctionSpace,
                    UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
                    VectorFunctionSpace, Vertex, cpp, has_petsc_complex,
                    interpolate)
from dolfin.cpp.mesh import CellType
from dolfin.io import XDMFFile
from ufl import FiniteElement, VectorElement

assert (tempdir)

# Supported XDMF file encoding
if MPI.size(MPI.comm_world) > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)

# Data types supported in templating
data_types = (('int', int), ('size_t', int), ('double', float))

# Finite elements tested
fe_1d_shapes = ["interval"]
fe_2d_shapes = ["triangle"]
fe_3d_shapes = ["tetrahedron"]
fe_families = ["CG", "DG"]
fe_degrees = [0, 1, 3]
mesh_tdims = [1, 2, 3]
mesh_ns = [6, 10]


def mesh_factory(tdim, n):
    if tdim == 1:
        return UnitIntervalMesh(MPI.comm_world, n)
    elif tdim == 2:
        return UnitSquareMesh(MPI.comm_world, n, n)
    elif tdim == 3:
        return UnitCubeMesh(MPI.comm_world, n, n, n)


def invalid_fe(fe_family, fe_degree):
    return (fe_family == "CG" and fe_degree == 0)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


@pytest.mark.parametrize("encoding", encodings)
def test_multiple_datasets(tempdir, encoding):
    mesh = UnitSquareMesh(MPI.comm_world, 2, 2)
    cf0 = MeshFunction('size_t', mesh, 2, 11)
    cf0.name = 'cf0'
    cf1 = MeshFunction('size_t', mesh, 2, 22)
    cf1.name = 'cf1'
    filename = os.path.join(tempdir, "multiple_mf.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
        xdmf.write(mesh)
        xdmf.write(cf0)
        xdmf.write(cf1)
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        mesh = xdmf.read_mesh(cpp.mesh.GhostMode.none)
        cf0 = xdmf.read_mf_size_t(mesh, "cf0")
        cf1 = xdmf.read_mf_size_t(mesh, "cf1")
    assert (cf0.values[0] == 11 and cf1.values[0] == 22)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_1d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFile(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh(cpp.mesh.GhostMode.none)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology.dim
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh_2D.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFile(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh(cpp.mesh.GhostMode.none)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology.dim
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_quad_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh_2D_quad.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32, CellType.quadrilateral)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFile(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh(cpp.mesh.GhostMode.none)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology.dim
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_3d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh_3D.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFile(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh(cpp.mesh.GhostMode.none)
    assert mesh.num_entities_global(0) == mesh2.num_entities_global(0)
    dim = mesh.topology.dim
    assert mesh.num_entities_global(dim) == mesh2.num_entities_global(dim)


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_scalar(tempdir, encoding):
    filename2 = os.path.join(tempdir, "u1_.xdmf")
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    # FIXME: This randomly hangs in parallel
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename2, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("fe_degree", fe_degrees)
@pytest.mark.parametrize("fe_family", fe_families)
@pytest.mark.parametrize("mesh_tdim", mesh_tdims)
@pytest.mark.parametrize("mesh_n", mesh_ns)
def test_save_and_checkpoint_scalar(tempdir, encoding, fe_degree, fe_family,
                                    mesh_tdim, mesh_n):
    if invalid_fe(fe_family, fe_degree):
        pytest.skip("Trivial finite element")

    filename = os.path.join(tempdir, "u1_checkpoint.xdmf")
    mesh = mesh_factory(mesh_tdim, mesh_n)
    FE = FiniteElement(fe_family, mesh.ufl_cell(), fe_degree)
    V = FunctionSpace(mesh, FE)
    u_in = Function(V)
    u_out = Function(V)

    if has_petsc_complex:
        def expr_eval(values, x):
            values[:, 0] = x[:, 0] + 1.0j * x[:, 0]
        u_out.interpolate(expr_eval)
    else:
        def expr_eval(values, x):
            values[:, 0] = x[:, 0]
        u_out.interpolate(expr_eval)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write_checkpoint(u_out, "u_out", 0)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        u_in = file.read_checkpoint(V, "u_out", 0)

    u_in.vector().axpy(-1.0, u_out.vector())
    assert u_in.vector().norm() < 1.0e-12


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("fe_degree", fe_degrees)
@pytest.mark.parametrize("fe_family", fe_families)
@pytest.mark.parametrize("mesh_tdim", mesh_tdims)
@pytest.mark.parametrize("mesh_n", mesh_ns)
def test_save_and_checkpoint_vector(tempdir, encoding, fe_degree, fe_family,
                                    mesh_tdim, mesh_n):
    if invalid_fe(fe_family, fe_degree):
        pytest.skip("Trivial finite element")

    filename = os.path.join(tempdir, "u2_checkpoint.xdmf")
    mesh = mesh_factory(mesh_tdim, mesh_n)
    FE = VectorElement(fe_family, mesh.ufl_cell(), fe_degree)
    V = FunctionSpace(mesh, FE)
    u_in = Function(V)
    u_out = Function(V)

    if has_petsc_complex:
        if mesh.geometry.dim == 1:
            def expr_eval(values, x):
                values[:, 0] = x[:, 0] + 1.0j * x[:, 0]
            u_out.interpolate(expr_eval)

        elif mesh.geometry.dim == 2:
            def expr_eval(values, x):
                values[:, 0] = 1.0j * x[:, 0] * x[:, 1]
                values[:, 1] = x[:, 0] + 1.0j * x[:, 0]
            u_out.interpolate(expr_eval)

        elif mesh.geometry.dim == 3:
            def expr_eval(values, x):
                values[:, 0] = x[:, 0] * x[:, 1]
                values[:, 1] = x[:, 0] + 1.0j * x[:, 0]
                values[:, 2] = x[:, 2]
            u_out.interpolate(expr_eval)
    else:
        if mesh.geometry.dim == 1:
            def expr_eval(values, x):
                values[:, 0] = x[:, 0]
            u_out.interpolate(expr_eval)

        elif mesh.geometry.dim == 2:
            def expr_eval(values, x):
                values[:, 0] = x[:, 0] * x[:, 1]
                values[:, 1] = x[:, 0]
            u_out.interpolate(expr_eval)

        elif mesh.geometry.dim == 3:
            def expr_eval(values, x):
                values[:, 0] = x[:, 0] * x[:, 1]
                values[:, 1] = x[:, 0]
                values[:, 2] = x[:, 2]
            u_out.interpolate(expr_eval)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write_checkpoint(u_out, "u_out", 0)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        u_in = file.read_checkpoint(V, "u_out", 0)

    u_in.vector().axpy(-1.0, u_out.vector())
    assert u_in.vector().norm() < 1.0e-12


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_checkpoint_timeseries(tempdir, encoding):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    filename = os.path.join(tempdir, "u2_checkpoint.xdmf")
    FE = FiniteElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, FE)

    times = [0.5, 0.2, 0.1]
    u_out = [None] * len(times)
    u_in = [None] * len(times)

    p = 0.0

    def expr_eval(values, x):
        values[:, 0] = x[:, 0] * p

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        for i, p in enumerate(times):
            u_out[i] = interpolate(expr_eval, V)
            file.write_checkpoint(u_out[i], "u_out", p)

    with XDMFFile(mesh.mpi_comm(), filename) as file:
        for i, p in enumerate(times):
            u_in[i] = file.read_checkpoint(V, "u_out", i)

    for i, p in enumerate(times):
        u_in[i].vector().axpy(-1.0, u_out[i].vector())
        assert u_in[i].vector().norm() < 1.0e-12

    # test reading last
    with XDMFFile(mesh.mpi_comm(), filename) as file:
        u_in_last = file.read_checkpoint(V, "u_out", -1)

    u_out[-1].vector().axpy(-1.0, u_in_last.vector())
    assert u_out[-1].vector().norm() < 1.0e-12


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_scalar(tempdir, encoding):
    filename = os.path.join(tempdir, "u2.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    # FIXME: This randomly hangs in parallel
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_scalar(tempdir, encoding):
    filename = os.path.join(tempdir, "u3.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_vector(tempdir, encoding):
    filename = os.path.join(tempdir, "u_2dv.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector(tempdir, encoding):
    filename = os.path.join(tempdir, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector_series(tempdir, encoding):
    filename = os.path.join(tempdir, "u_3D.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        u.vector().set(1.0 + (1j if has_petsc_complex else 0))
        file.write(u, 0.1)
        u.vector().set(2.0 + (2j if has_petsc_complex else 0))
        file.write(u, 0.2)
        u.vector().set(3.0 + (3j if has_petsc_complex else 0))
        file.write(u, 0.3)


@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_tensor(tempdir, encoding):
    filename = os.path.join(tempdir, "tensor.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_tensor(tempdir, encoding):
    filename = os.path.join(tempdir, "u3t.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector().set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mf_1D.xdmf")
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    mf = MeshFunction("size_t", mesh, mesh.topology.dim, 0)

    mf.values[:] = numpy.arange(mesh.num_entities(1))

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_cell_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    filename = os.path.join(tempdir, "mf_2D_%s.xdmf" % dtype_str)
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    mf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
    mf.name = "cells"

    mf.values[:] = numpy.arange(mesh.num_entities(2), dtype=dtype)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh, "cells")

    diff = mf_in.values - mf.values
    assert numpy.all(diff == 0)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_cell_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    mf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
    mf.name = "cells"

    mf.values[:] = numpy.arange(mesh.num_entities(3), dtype=dtype)

    filename = os.path.join(tempdir, "mf_3D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh, "cells")

    diff = mf_in.values - mf.values
    assert numpy.all(diff == 0)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_facet_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    tdim = mesh.topology.dim
    mf = MeshFunction(dtype_str, mesh, tdim - 1, 0)
    mf.name = "facets"

    tdim = mesh.topology.dim
    if (MPI.size(mesh.mpi_comm()) == 1):
        for i in range(mesh.num_entities(tdim - 1)):
            mf.values[i] = dtype(i)
    else:
        for i in range(mesh.num_entities(tdim - 1)):
            f = Facet(mesh, i)
            mf.values[i] = dtype(f.global_index())
    filename = os.path.join(tempdir, "mf_facet_2D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
        xdmf.write(mf)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh, "facets")

    diff = mf_in.values - mf.values
    assert numpy.all(diff == 0)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_facet_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    tdim = mesh.topology.dim
    mf = MeshFunction(dtype_str, mesh, tdim - 1, 0)
    mf.name = "facets"

    if (MPI.size(mesh.mpi_comm()) == 1):
        for i in range(mesh.num_entities(tdim - 1)):
            mf.values[i] = dtype(i)
    else:
        for i in range(mesh.num_entities(tdim - 1)):
            f = Facet(mesh, i)
            mf.values[i] = dtype(f.global_index())
    filename = os.path.join(tempdir, "mf_facet_3D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
        xdmf.write(mf)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh, "facets")

    diff = mf_in.values - mf.values
    assert numpy.all(diff == 0)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_edge_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    mf = MeshFunction(dtype_str, mesh, 1, 0)
    mf.name = "edges"

    mf.values[:] = numpy.arange(mesh.num_entities(1), dtype=dtype)

    filename = os.path.join(tempdir, "mf_edge_3D_%s.xdmf" % dtype_str)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_vertex_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    mf = MeshFunction(dtype_str, mesh, 0, 0)
    mf.name = "vertices"
    for v in range(mesh.num_entities(0)):
        mf.values[v] = dtype(Vertex(mesh, v).global_index())
    filename = os.path.join(tempdir, "mf_vertex_2D_%s.xdmf" % dtype_str)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh, "vertices")

    diff = mf_in.values - mf.values
    assert numpy.all(diff == 0)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_vertex_function(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    filename = os.path.join(tempdir, "mf_vertex_3D_%s.xdmf" % dtype_str)
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    mf = MeshFunction(dtype_str, mesh, 0, 0)
    for v in range(mesh.num_entities(0)):
        mf.values[v] = dtype(v)

    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)


@pytest.mark.parametrize("encoding", encodings)
def test_save_points_2D(tempdir, encoding):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    points = mesh.geometry.points
    vals = numpy.linalg.norm(points, axis=1)
    with XDMFFile(
            mesh.mpi_comm(),
            os.path.join(tempdir, "points_2D.xdmf"),
            encoding=encoding) as file:
        file.write(points)
    with XDMFFile(
            mesh.mpi_comm(),
            os.path.join(tempdir, "points_values_2D.xdmf"),
            encoding=encoding) as file:
        file.write(points, vals)


@pytest.mark.parametrize("encoding", encodings)
def test_save_points_3D(tempdir, encoding):
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    points = mesh.geometry.points
    vals = numpy.linalg.norm(points, axis=1)
    with XDMFFile(
            mesh.mpi_comm(),
            os.path.join(tempdir, "points_3D.xdmf"),
            encoding=encoding) as file:
        file.write(points)
    with XDMFFile(
            mesh.mpi_comm(),
            os.path.join(tempdir, "points_values_3D.xdmf"),
            encoding=encoding) as file:
        file.write(points, vals)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_mesh_value_collection(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4)
    tdim = mesh.topology.dim
    meshfn = MeshFunction(dtype_str, mesh, mesh.topology.dim, False)
    meshfn.name = "volume_marker"
    for i in range(mesh.num_cells()):
        c = Cell(mesh, i)
        if cpp.mesh.midpoint(c)[1] > 0.1:
            meshfn.values[c.index()] = dtype(1)
        if cpp.mesh.midpoint(c)[1] > 0.9:
            meshfn.values[c.index()] = dtype(2)

    for mvc_dim in range(0, tdim + 1):
        mvc = MeshValueCollection(dtype_str, mesh, mvc_dim)
        tag = "dim_{}_marker".format(mvc_dim)
        mvc.name = tag
        mesh.create_connectivity(mvc_dim, tdim)
        for e in MeshEntities(mesh, mvc_dim):
            if (cpp.mesh.midpoint(e)[0] > 0.5):
                mvc.set_value(e.index(), dtype(1))

        filename = os.path.join(tempdir, "mvc_{}.xdmf".format(mvc_dim))

        with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
            xdmf.write(meshfn)
            xdmf.write(mvc)

        with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
            read_function = getattr(xdmf, "read_mvc_" + dtype_str)
            mvc = read_function(mesh, tag)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_append_and_load_mesh_functions(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    meshes = [
        UnitSquareMesh(MPI.comm_world, 12, 12),
        UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    ]

    for mesh in meshes:
        dim = mesh.topology.dim

        vf = MeshFunction(dtype_str, mesh, 0, 0)
        vf.name = "vertices"
        ff = MeshFunction(dtype_str, mesh, mesh.topology.dim - 1, 0)
        ff.name = "facets"
        cf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
        cf.name = "cells"

        if (MPI.size(mesh.mpi_comm()) == 1):
            for v in range(mesh.num_entities(0)):
                vf.values[v] = dtype(v)
            for f in range(mesh.num_entities(dim - 1)):
                ff.values[f] = dtype(f)
            for c in range(mesh.num_entities(dim)):
                cf.values[c] = dtype(c)
        else:
            for v in range(mesh.num_entities(0)):
                vf.values[v] = dtype(Vertex(mesh, v).global_index())
            for f in range(mesh.num_entities(dim - 1)):
                ff.values[f] = dtype(Facet(mesh, f).global_index())
            for c in range(mesh.num_entities(dim)):
                cf.values[c] = dtype(Cell(mesh, c).global_index())

        filename = os.path.join(tempdir, "appended_mf_%dD.xdmf" % dim)

        with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
            xdmf.write(mesh)
            xdmf.write(vf)
            xdmf.write(ff)
            xdmf.write(cf)

        with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
            read_function = getattr(xdmf, "read_mf_" + dtype_str)
            vf_in = read_function(mesh, "vertices")
            ff_in = read_function(mesh, "facets")
            cf_in = read_function(mesh, "cells")

        diff_vf = vf_in.values - vf.values
        diff_ff = ff_in.values - ff.values
        diff_cf = cf_in.values - cf.values

        assert numpy.all(diff_vf == 0)
        assert numpy.all(diff_ff == 0)
        assert numpy.all(diff_cf == 0)


@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_append_and_load_mesh_value_collections(tempdir, encoding, data_type):
    dtype_str, dtype = data_type
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    mesh.create_connectivity_all()
    for d in range(mesh.geometry.dim + 1):
        mesh.create_global_indices(d)

    mvc_v = MeshValueCollection(dtype_str, mesh, 0)
    mvc_v.name = "vertices"
    mvc_e = MeshValueCollection(dtype_str, mesh, 1)
    mvc_e.name = "edges"
    mvc_f = MeshValueCollection(dtype_str, mesh, 2)
    mvc_f.name = "facets"
    mvc_c = MeshValueCollection(dtype_str, mesh, 3)
    mvc_c.name = "cells"

    mvcs = [mvc_v, mvc_e, mvc_f, mvc_c]

    filename = os.path.join(tempdir, "appended_mvcs.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        for mvc in mvcs:
            for ent in MeshEntities(mesh, mvc.dim):
                assert (mvc.set_value(ent.index(), dtype(ent.global_index())))
            xdmf.write(mvc)

    mvc_v_in = MeshValueCollection(dtype_str, mesh, 0)
    mvc_e_in = MeshValueCollection(dtype_str, mesh, 1)
    mvc_f_in = MeshValueCollection(dtype_str, mesh, 2)
    mvc_c_in = MeshValueCollection(dtype_str, mesh, 3)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mvc_" + dtype_str)
        mvc_v_in = read_function(mesh, "vertices")
        mvc_e_in = read_function(mesh, "edges")
        mvc_f_in = read_function(mesh, "facets")
        mvc_c_in = read_function(mesh, "cells")

    mvcs_in = [mvc_v_in, mvc_e_in, mvc_f_in, mvc_c_in]

    for (mvc, mvc_in) in zip(mvcs, mvcs_in):
        mf = MeshFunction(dtype_str, mesh, mvc, 0)
        mf_in = MeshFunction(dtype_str, mesh, mvc_in, 0)

        diff = mf_in.values - mf.values
        assert numpy.all(diff == 0)


def test_xdmf_timeseries_write_to_closed_hdf5_using_with(tempdir):
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    V = FunctionSpace(mesh, ("CG", 1))
    u = Function(V)

    filename = os.path.join(tempdir, "time_series_closed_append.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        xdmf.write(u, float(0.0))

    xdmf.write(u, float(1.0))
    xdmf.close()

    with xdmf:
        xdmf.write(u, float(2.0))
