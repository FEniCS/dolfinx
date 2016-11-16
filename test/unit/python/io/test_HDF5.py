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
from dolfin_utils.test import skip_if_not_HDF5, fixture, tempdir


@skip_if_not_HDF5
def test_save_vector(tempdir):
    filename = os.path.join(tempdir, "x.h5")
    x = Vector(mpi_comm_world(), 305)
    x[:] = 1.0
    with HDF5File(x.mpi_comm(), filename, "w") as vector_file:
        vector_file.write(x, "/my_vector")

@skip_if_not_HDF5
def test_save_and_read_vector(tempdir):
    filename = os.path.join(tempdir, "vector.h5")

    # Write to file
    x = Vector(mpi_comm_world(), 305)
    x[:] = 1.2
    with HDF5File(x.mpi_comm(), filename, "w") as vector_file:
        vector_file.write(x, "/my_vector")

    # Read from file
    y = Vector()
    with HDF5File(x.mpi_comm(), filename, "r") as vector_file:
        vector_file.read(y, "/my_vector", False)
        assert y.size() == x.size()
        assert (x - y).norm("l1") == 0.0

@skip_if_not_HDF5
def test_save_and_read_meshfunction_2D(tempdir):
    filename = os.path.join(tempdir, "meshfn-2d.h5")

    # Write to file
    mesh = UnitSquareMesh(20, 20)
    with HDF5File(mesh.mpi_comm(), filename, "w") as mf_file:

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

    # Read back from file
    with HDF5File(mesh.mpi_comm(), filename, "r") as mf_file:
        for i in range(0,3):
            mf2 = MeshFunction('double', mesh, i)
            mf_file.read(mf2, "/meshfunction/meshfun%d" % i)
            for cell in entities(mesh, i):
                assert meshfunctions[i][cell] == mf2[cell]

@skip_if_not_HDF5
def test_save_and_read_meshfunction_3D(tempdir):
    filename = os.path.join(tempdir, "meshfn-3d.h5")

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
    mf_file.close()

    # Read back from file
    mf_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for i in range(0,4):
        mf2 = MeshFunction('double', mesh, i)
        mf_file.read(mf2, "/meshfunction/group/%d/meshfun"%i)
        for cell in entities(mesh, i):
            assert meshfunctions[i][cell] == mf2[cell]
    mf_file.close()

@skip_if_not_HDF5
def test_save_and_read_mesh_value_collection(tempdir):
    ndiv = 5
    filename = os.path.join(tempdir, "mesh_value_collection.h5")
    mesh = UnitCubeMesh(ndiv, ndiv, ndiv)

    point2list = lambda p : [ p.x(), p.y(), p.z() ]

    # write to file
    with HDF5File(mesh.mpi_comm(), filename, 'w') as f :
        for dim in range(mesh.topology().dim()) :
            mvc = MeshValueCollection("size_t", mesh, dim)
            mesh.init(dim)
            for e in entities(mesh, dim) :
                # this can be easily computed to the check the value
                val = int(ndiv*sum(point2list(e.midpoint()))) + 1
                mvc.set_value(e.index(), val)
            f.write(mvc, "/mesh_value_collection_{}".format(dim))

    # read from file
    with HDF5File(mesh.mpi_comm(), filename, 'r') as f :
        for dim in range(mesh.topology().dim()) :
            mvc = MeshValueCollection("size_t", mesh, dim)
            f.read(mvc, "/mesh_value_collection_{}".format(dim))
            # check the values
            for (cell, lidx), val in mvc.values().items() :
                eidx = Cell(mesh, cell).entities(dim)[lidx]
                mid = point2list(MeshEntity(mesh, dim, eidx).midpoint())
                assert val == int(ndiv*sum(mid)) + 1

@skip_if_not_HDF5
def test_save_and_read_mesh_value_collection_with_only_one_marked_entity(tempdir):
    ndiv = 5
    filename = os.path.join(tempdir, "mesh_value_collection.h5")
    mesh = UnitCubeMesh(ndiv, ndiv, ndiv)
    mvc = MeshValueCollection("size_t", mesh, 3)
    mesh.init(3)
    if MPI.rank(mesh.mpi_comm()) == 0:
        mvc.set_value(0, 1)

    # write to file
    with HDF5File(mesh.mpi_comm(), filename, 'w') as f :
        f.write(mvc, "/mesh_value_collection")

    # read from file
    with HDF5File(mesh.mpi_comm(), filename, 'r') as f :
        mvc = MeshValueCollection("size_t", mesh, 3)
        f.read(mvc, "/mesh_value_collection")
        assert MPI.sum(mesh.mpi_comm(), mvc.size()) == 1
        if MPI.rank(mesh.mpi_comm()) == 0:
            assert mvc.get_value(0, 0) == 1

@skip_if_not_HDF5
def test_save_and_read_function(tempdir):
    filename = os.path.join(tempdir, "function.h5")

    mesh = UnitSquareMesh(10, 10)
    Q = FunctionSpace(mesh, "CG", 3)
    F0 = Function(Q)
    F1 = Function(Q)
    E = Expression("x[0]", degree=1)
    F0.interpolate(E)

    # Save to HDF5 File

    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    hdf5_file.write(F0, "/function")
    hdf5_file.close()

    #Read back from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    hdf5_file.read(F1, "/function")
    result = F0.vector() - F1.vector()
    assert len(result.array().nonzero()[0]) == 0
    hdf5_file.close()

@skip_if_not_HDF5
def test_save_and_read_mesh_2D(tempdir):
    filename = os.path.join(tempdir, "mesh2d.h5")

    # Write to file
    mesh0 = UnitSquareMesh(20, 20)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    mesh_file.close()

    # Read from file
    mesh1 = Mesh()
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh_file.read(mesh1, "/my_mesh", False)
    mesh_file.close()

    assert mesh0.size_global(0) == mesh1.size_global(0)
    dim = mesh0.topology().dim()
    assert mesh0.size_global(dim) == mesh1.size_global(dim)

@skip_if_not_HDF5
def test_save_and_read_mesh_3D(tempdir):
    filename = os.path.join(tempdir, "mesh3d.h5")

    # Write to file
    mesh0 = UnitCubeMesh(10, 10, 10)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    mesh_file.close()

    # Read from file
    mesh1 = Mesh()
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh_file.read(mesh1, "/my_mesh", False)
    mesh_file.close()

    assert mesh0.size_global(0) == mesh1.size_global(0)
    dim = mesh0.topology().dim()
    assert mesh0.size_global(dim) == mesh1.size_global(dim)

@skip_if_not_HDF5
def test_mpi_atomicity(tempdir):
    comm_world = mpi_comm_world()
    if MPI.size(comm_world) > 1:
        filename = os.path.join(tempdir, "mpiatomic.h5")
        with HDF5File(mpi_comm_world(), filename, "w") as f:
            assert f.get_mpi_atomicity() is False
            f.set_mpi_atomicity(True)
            assert f.get_mpi_atomicity() is True
