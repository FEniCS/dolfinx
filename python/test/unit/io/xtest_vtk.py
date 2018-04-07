# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
import os
from dolfin_utils.test import skip_in_parallel, fixture, tempdir

# VTK file options
@fixture
def file_options():
    return ["ascii", "base64", "compressed"]

@fixture
def mesh_function_types():
    return ["size_t", "int", "double", "bool"]

@fixture
def type_conv():
    return dict(size_t=int, int=int, double=float, bool=bool)

@pytest.fixture(scope="function")
def tempfile(tempdir, request):
    return os.path.join(tempdir, request.function.__name__)

def test_save_1d_meshfunctions(tempfile,
                                mesh_function_types, file_options, type_conv):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    for d in range(mesh.topology.dim+1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology.dim-d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd", "ascii").write(mf)
            f = VTKFile(tempfile + "mf.pvd", "ascii")
            f.write(mf, 0.)
            f.write(mf, 1.)
            for file_option in file_options:
                VTKFile(tempfile + "mf.pvd", file_option).write(mf)

def test_save_2d_meshfunctions(tempfile,
                                mesh_function_types, file_options, type_conv):
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    for d in range(mesh.topology.dim+1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology.dim-d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd", "ascii").write(mf)
            f = VTKFile(tempfile + "mf.pvd", "ascii")
            f.write(mf, 0.)
            f.write(mf, 1.)
            for file_option in file_options:
                VTKFile(tempfile + "mf.pvd", file_option).write(mf)

def test_save_3d_meshfunctions(tempfile,
                                mesh_function_types, file_options, type_conv):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    for d in range(mesh.topology.dim+1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology.dim-d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd", "ascii").write(mf)
            f = VTKFile(tempfile + "mf.pvd", "ascii")
            f.write(mf, 0.)
            f.write(mf, 1.)
            for file_option in file_options:
                VTKFile(tempfile + "mf.pvd", file_option).write(mf)

def test_save_1d_mesh(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    VTKFile(tempfile + "mesh.pvd", "ascii").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd", "ascii")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)

def test_save_2d_mesh(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    VTKFile(tempfile + "mesh.pvd", "ascii").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd", "ascii")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)

def test_save_3d_mesh(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    VTKFile(tempfile + "mesh.pvd", "ascii").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd", "ascii")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)

def test_save_1d_scalar(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd", "ascii")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

def test_save_2d_scalar(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd", "ascii")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

def test_save_3d_scalar(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd", "ascii")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

@pytest.mark.xfail(reason="FFC fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_vector(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

def test_save_2d_vector(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

def test_save_3d_vector(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

@pytest.mark.xfail(reason="FFC fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_tensor(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

def test_save_2d_tensor(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd", "ascii")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)

def test_save_3d_tensor(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd", "ascii").write(u)
    f = VTKFile(tempfile + "u.pvd", "ascii")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)
