# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
from dolfin_utils.test.fixtures import tempdir
from dolfin_utils.test.skips import skip_in_parallel

from dolfin import (MPI, Function, FunctionSpace, MeshFunction,
                    TensorFunctionSpace, UnitCubeMesh, UnitIntervalMesh,
                    UnitSquareMesh, VectorFunctionSpace)
from dolfin.io import VTKFile

assert (tempdir)


# VTK file options
@pytest.fixture
def file_options():
    return ["ascii", "base64", "compressed"]


@pytest.fixture
def mesh_function_types():
    return ["size_t", "int", "double"]


@pytest.fixture
def type_conv():
    return dict(size_t=int, int=int, double=float)


@pytest.fixture(scope="function")
def tempfile(tempdir, request):
    return os.path.join(tempdir, request.function.__name__)


def test_save_1d_meshfunctions(tempfile, mesh_function_types, file_options,
                               type_conv):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    for d in range(mesh.topology.dim + 1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology.dim - d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd").write(mf)
            f = VTKFile(tempfile + "mf.pvd")
            f.write(mf, 0.)
            f.write(mf, 1.)
            # for file_option in file_options:
            #     VTKFile(tempfile + "mf.pvd", file_option).write(mf)


def test_save_2d_meshfunctions(tempfile, mesh_function_types, file_options,
                               type_conv):
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    for d in range(mesh.topology.dim + 1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology.dim - d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd").write(mf)
            f = VTKFile(tempfile + "mf.pvd")
            f.write(mf, 0.)
            f.write(mf, 1.)
            # for file_option in file_options:
            #     VTKFile(tempfile + "mf.pvd", file_option).write(mf)


def test_save_3d_meshfunctions(tempfile, mesh_function_types, file_options,
                               type_conv):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    for d in range(mesh.topology.dim + 1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology.dim - d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd").write(mf)
            f = VTKFile(tempfile + "mf.pvd")
            f.write(mf, 0.)
            f.write(mf, 1.)
            # for file_option in file_options:
            #     VTKFile(tempfile + "mf.pvd", file_option).write(mf)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_1d_mesh(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    VTKFile(tempfile + "mesh.pvd").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_mesh(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    VTKFile(tempfile + "mesh.pvd").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_mesh(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    VTKFile(tempfile + "mesh.pvd").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_1d_scalar(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_scalar(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_scalar(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="FFC fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_vector(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_vector(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_vector(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="FFC fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_tensor(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_tensor(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_tensor(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)
