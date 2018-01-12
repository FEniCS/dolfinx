# Copyright (C) 2011 Garth N. Wells
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
    for d in range(mesh.topology().dim()+1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology().dim()-d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd", "ascii") << mf
            f = VTKFile(tempfile + "mf.pvd", "ascii")
            f << (mf, 0.)
            f << (mf, 1.)
            for file_option in file_options:
                VTKFile(tempfile + "mf.pvd", file_option) << mf

def test_save_2d_meshfunctions(tempfile,
                                mesh_function_types, file_options, type_conv):
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    for d in range(mesh.topology().dim()+1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology().dim()-d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd") << mf
            f = VTKFile(tempfile + "mf.pvd")
            f << (mf, 0.)
            f << (mf, 1.)
            for file_option in file_options:
                VTKFile(tempfile + "mf.pvd", file_option) << mf

def test_save_3d_meshfunctions(tempfile,
                                mesh_function_types, file_options, type_conv):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    for d in range(mesh.topology().dim()+1):
        for t in mesh_function_types:
            mf = MeshFunction(t, mesh, mesh.topology().dim()-d, type_conv[t](1))
            VTKFile(tempfile + "mf.pvd") << mf
            f = VTKFile(tempfile + "mf.pvd")
            f << (mf, 0.)
            f << (mf, 1.)
            for file_option in file_options:
                VTKFile(tempfile + "mf.pvd", file_option) << mf

def test_save_1d_mesh(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    VTKFile(tempfile + "mesh.pvd") << mesh
    f = VTKFile(tempfile + "mesh.pvd")
    f << (mesh, 0.)
    f << (mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option) << mesh

def test_save_2d_mesh(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 32, 32)
    VTKFile(tempfile + "mesh.pvd") << mesh
    f = VTKFile(tempfile + "mesh.pvd")
    f << (mesh, 0.)
    f << (mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option) << mesh

def test_save_3d_mesh(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    VTKFile(tempfile + "mesh.pvd") << mesh
    f = VTKFile(tempfile + "mesh.pvd")
    f << (mesh, 0.)
    f << (mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option) << mesh

def test_save_1d_scalar(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

def test_save_2d_scalar(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

def test_save_3d_scalar(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

@pytest.mark.xfail(reason="FFC fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_vector(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

def test_save_2d_vector(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

def test_save_3d_vector(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

@pytest.mark.xfail(reason="FFC fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_tensor(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

def test_save_2d_tensor(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u

def test_save_3d_tensor(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    VTKFile(tempfile + "u.pvd") << u
    f = VTKFile(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option) << u
