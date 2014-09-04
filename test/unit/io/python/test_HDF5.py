#!/usr/bin/env py.test

"""Unit tests for the HDF5 io library"""

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
#
# Modified by Chris Richardson 2013

import pytest
import os
from dolfin import *

# create an output folder
@pytest.fixture(scope="module")
def temppath():
    filedir = os.path.dirname(os.path.abspath(__file__))
    basename = os.path.basename(__file__).replace(".py", "_data")
    temppath = os.path.join(filedir, basename, "")
    if not os.path.exists(temppath):
        os.mkdir(temppath)
    return temppath

skip_if_no_hdf5 = pytest.mark.skipif(not has_hdf5(),
                                     reason="Skipping unit test(s) depending on HDF5.")

@skip_if_no_hdf5
def test_save_vector(temppath):
    filename = os.path.join(temppath, "x.h5")
    x = Vector(mpi_comm_world(), 305)
    x[:] = 1.0
    vector_file = HDF5File(x.mpi_comm(), filename, "w")
    vector_file.write(x, "/my_vector")

@skip_if_no_hdf5
def test_save_and_read_vector(temppath):
    filename = os.path.join(temppath, "vector.h5")

    # Write to file
    x = Vector(mpi_comm_world(), 305)
    x[:] = 1.2
    vector_file = HDF5File(x.mpi_comm(), filename, "w")
    vector_file.write(x, "/my_vector")
    del vector_file

    # Read from file
    y = Vector()
    vector_file = HDF5File(x.mpi_comm(), filename, "r")
    vector_file.read(y, "/my_vector", False)
    assert y.size() == x.size()
    assert (x - y).norm("l1") == 0.0

@skip_if_no_hdf5
def test_save_and_read_meshfunction_2D(temppath):
    filename = os.path.join(temppath, "meshfn-2d.h5")

    # Write to file
    mesh = UnitSquareMesh(20, 20)
    mf_file = HDF5File(mesh.mpi_comm(), filename, "w")

    # save meshfuns to compare when reading back
    meshfunctions = []
    for i in range(0,3):
        mf = MeshFunction('double', mesh, i)
        # NB choose a value to set which will be the same
        # on every process for each entity
        for cell in entities(mesh, i):
            mf[cell] = cell.midpoint()[0]
        meshfunctions.append(mf)
        mf_file.write(mf, "/meshfunction/meshfun%d" % i)

    del mf_file

    # Read back from file
    mf_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for i in range(0,3):
        mf2 = MeshFunction('double', mesh, i)
        mf_file.read(mf2, "/meshfunction/meshfun%d" % i)
        for cell in entities(mesh, i):
            assert meshfunctions[i][cell] == mf2[cell]

@skip_if_no_hdf5
def test_save_and_read_meshfunction_3D(temppath):
    filename = os.path.join(temppath, "meshfn-3d.h5")

    # Write to file
    mesh = UnitCubeMesh(10, 10, 10)
    mf_file = HDF5File(mesh.mpi_comm(), filename, "w")

    # save meshfuns to compare when reading back
    meshfunctions = []
    for i in range(0,4):
        mf = MeshFunction('double', mesh, i)
        # NB choose a value to set which will be the same
        # on every process for each entity
        for cell in entities(mesh, i):
            mf[cell] = cell.midpoint()[0]
        meshfunctions.append(mf)
        mf_file.write(mf, "/meshfunction/group/%d/meshfun"%i)

    del mf_file

    # Read back from file
    mf_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for i in range(0,4):
        mf2 = MeshFunction('double', mesh, i)
        mf_file.read(mf2, "/meshfunction/group/%d/meshfun"%i)
        for cell in entities(mesh, i):
            assert meshfunctions[i][cell] == mf2[cell]

@skip_if_no_hdf5
def test_save_and_read_mesh_value_collection(temppath):
    filename = os.path.join(temppath, "mesh_value_collection.h5")
    mesh = UnitCubeMesh(5, 5, 5)

    # Writ to file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    for dim in range(mesh.topology().dim()):
        mvc = MeshValueCollection("size_t", mesh, dim)
        for i, cell in enumerate(entities(mesh, dim)):
            mvc.set_value(cell.index(), i)
        hdf5_file.write(mvc, "/mesh_value_collection_%d" % dim)
    del hdf5_file

    # Read from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for dim in range(mesh.topology().dim()):
        mvc = MeshValueCollection("size_t", mesh, dim)
        hdf5_file.read(mvc, "/mesh_value_collection_%d" % dim)

@skip_if_no_hdf5
def test_save_and_read_function(temppath):
    filename = os.path.join(temppath, "function.h5")

    mesh = UnitSquareMesh(10, 10)
    Q = FunctionSpace(mesh, "CG", 3)
    F0 = Function(Q)
    F1 = Function(Q)
    E = Expression("x[0]")
    F0.interpolate(E)

    # Save to HDF5 File

    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    hdf5_file.write(F0, "/function")
    del hdf5_file

    #Read back from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    hdf5_file.read(F1, "/function")
    result = F0.vector() - F1.vector()
    assert len(result.array().nonzero()[0]) == 0

@skip_if_no_hdf5
def test_save_and_read_mesh_2D(temppath):
    filename = os.path.join(temppath, "mesh2d.h5")

    # Write to file
    mesh0 = UnitSquareMesh(20, 20)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    del mesh_file

    # Read from file
    mesh1 = Mesh()
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh_file.read(mesh1, "/my_mesh", False)

    assert mesh0.size_global(0) == mesh1.size_global(0)
    dim = mesh0.topology().dim()
    assert mesh0.size_global(dim) == mesh1.size_global(dim)

@skip_if_no_hdf5
def test_save_and_read_mesh_3D(temppath):
    filename = os.path.join(temppath, "mesh3d.h5")

    # Write to file
    mesh0 = UnitCubeMesh(10, 10, 10)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    del mesh_file

    # Read from file
    mesh1 = Mesh()
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh_file.read(mesh1, "/my_mesh", False)

    assert mesh0.size_global(0) == mesh1.size_global(0)
    dim = mesh0.topology().dim()
    assert mesh0.size_global(dim) == mesh1.size_global(dim)
