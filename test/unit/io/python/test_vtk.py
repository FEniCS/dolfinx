#!/usr/bin/env py.test

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
#
# First added:  2011-05-18
# Last changed:

import pytest
from dolfin import *
import os
from dolfin_utils.test import skip_in_parallel, fixture

# VTK file options
@fixture
def file_options():
    return ["ascii", "base64", "compressed"]

@fixture
def mesh_functions():
    return [CellFunction, FacetFunction, FaceFunction, EdgeFunction, VertexFunction]

@fixture
def mesh_function_types():
    return ["size_t", "int", "double", "bool"]

@fixture
def type_conv():
    return dict(size_t=int, int=int, double=float, bool=bool)

# create an output folder
@fixture
def temppath():
    filedir = os.path.dirname(os.path.abspath(__file__))
    basename = os.path.basename(__file__).replace(".py", "_data")
    temppath = os.path.join(filedir, basename)
    if not os.path.exists(temppath):
        os.mkdir(temppath)
    return temppath

@pytest.fixture(scope="function")
def tempfile(temppath, request):
    return os.path.join(temppath, request.function.__name__)

def test_save_1d_meshfunctions(tempfile, mesh_functions, \
                                mesh_function_types, file_options, type_conv):
    mesh = UnitIntervalMesh(32)
    for F in mesh_functions:
       if F in [FaceFunction, EdgeFunction]: continue
       for t in mesh_function_types:
            mf = F(t, mesh, type_conv[t](1))
            File(tempfile + "mf.pvd") << mf
            f = File(tempfile + "mf.pvd")
            f << (mf, 0.)
            f << (mf, 1.)
            for file_option in file_options:
                File(tempfile + "mf.pvd", file_option) << mf

def test_save_2d_meshfunctions(tempfile, mesh_functions, \
                                mesh_function_types, file_options, type_conv):
    mesh = UnitSquareMesh(32, 32)
    for F in mesh_functions:
        for t in mesh_function_types:
            mf = F(t, mesh, type_conv[t](1))
            File(tempfile + "mf.pvd") << mf
            f = File(tempfile + "mf.pvd")
            f << (mf, 0.)
            f << (mf, 1.)
            for file_option in file_options:
                File(tempfile + "mf.pvd", file_option) << mf

def test_save_3d_meshfunctions(tempfile, mesh_functions, \
                                mesh_function_types, file_options, type_conv):
    mesh = UnitCubeMesh(8, 8, 8)
    for F in mesh_functions:
        for t in mesh_function_types:
            mf = F(t, mesh, type_conv[t](1))
            File(tempfile + "mf.pvd") << mf
            f = File(tempfile + "mf.pvd")
            f << (mf, 0.)
            f << (mf, 1.)
            for file_option in file_options:
                File(tempfile + "mf.pvd", file_option) << mf

def test_save_1d_mesh(tempfile, file_options):
    mesh = UnitIntervalMesh(32)
    File(tempfile + "mesh.pvd") << mesh
    f = File(tempfile + "mesh.pvd")
    f << (mesh, 0.)
    f << (mesh, 1.)
    for file_option in file_options:
        File(tempfile + "mesh.pvd", file_option) << mesh

def test_save_2d_mesh(tempfile, file_options):
    mesh = UnitSquareMesh(32, 32)
    File(tempfile + "mesh.pvd") << mesh
    f = File(tempfile + "mesh.pvd")
    f << (mesh, 0.)
    f << (mesh, 1.)
    for file_option in file_options:
        File(tempfile + "mesh.pvd", file_option) << mesh

def test_save_3d_mesh(tempfile, file_options):
    mesh = UnitCubeMesh(8, 8, 8)
    File(tempfile + "mesh.pvd") << mesh
    f = File(tempfile + "mesh.pvd")
    f << (mesh, 0.)
    f << (mesh, 1.)
    for file_option in file_options:
        File(tempfile + "mesh.pvd", file_option) << mesh 

def test_save_1d_scalar(tempfile, file_options):
    mesh = UnitIntervalMesh(32)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u

def test_save_2d_scalar(tempfile, file_options):
    mesh = UnitSquareMesh(16, 16)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u

def test_save_3d_scalar(tempfile, file_options):
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u

# FFC fails for vector spaces in 1D
#@pytest.skipIf(MPI.size(mpi_comm_world()) > 1, "Skipping unit test(s) not working in parallel")
#def test_save_1d_vector(tempfile, file_options):
#    mesh = UnitIntervalMesh(32)
#    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
#    u.vector()[:] = 1.0
#    File(tempfile + "u.pvd") << u
#    for file_option in file_options:
#        File(tempfile + "u.pvd", file_option) << u

def test_save_2d_vector(tempfile, file_options):
    mesh = UnitSquareMesh(16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u

def test_save_3d_vector(tempfile, file_options):
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u

# FFC fails for tensor spaces in 1D
#@pskip_in_parallel
#def test_save_1d_tensor(tempfile, file_options):
#    mesh = UnitIntervalMesh(32)
#    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
#    u.vector()[:] = 1.0
#    File(tempfile + "u.pvd") << u
#    for file_option in file_options:
#        File(tempfile + "u.pvd", file_option) << u

def test_save_2d_tensor(tempfile, file_options):
    mesh = UnitSquareMesh(16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u

def test_save_3d_tensor(tempfile, file_options):
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File(tempfile + "u.pvd") << u
    f = File(tempfile + "u.pvd")
    f << (u, 0.)
    f << (u, 1.)
    for file_option in file_options:
        File(tempfile + "u.pvd", file_option) << u
