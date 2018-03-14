# Copyright (C) 2012 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import os
from dolfin import *
from dolfin_utils.test import (skip_if_not_HDF5, fixture, tempdir,
                               xfail_with_serial_hdf5_in_parallel)
from dolfin.la import PETScVector
import dolfin.io as io


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_parallel(tempdir):
    filename = os.path.join(tempdir, "y.h5")
    have_parallel = has_hdf5_parallel()
    hdf5 = HDF5File(MPI.comm_world, filename, "w")


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_vector(tempdir):
    filename = os.path.join(tempdir, "x.h5")
    x = PETScVector(MPI.comm_world, 305)
    x[:] = 1.0
    with HDF5File(MPI.comm_world, filename, "w") as vector_file:
        vector_file.write(x, "/my_vector")


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_vector(tempdir):
    filename = os.path.join(tempdir, "vector.h5")

    # Write to file
    x = PETScVector(MPI.comm_world, 305)
    x[:] = 1.2
    with HDF5File(MPI.comm_world, filename, "w") as vector_file:
        vector_file.write(x, "/my_vector")

    # Read from file
    with HDF5File(MPI.comm_world, filename, "r") as vector_file:
        y = vector_file.read_vector(MPI.comm_world, "/my_vector", False)
        assert y.size() == x.size()
        assert (x - y).norm("l1") == 0.0


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_meshfunction_2D(tempdir):
    filename = os.path.join(tempdir, "meshfn-2d.h5")

    # Write to file
    mesh = UnitSquareMesh(MPI.comm_world, 20, 20)
    with HDF5File(mesh.mpi_comm(), filename, "w") as mf_file:

        # save meshfuns to compare when reading back
        meshfunctions = []
        for i in range(0, 3):
            mf = MeshFunction('double', mesh, i, 0.0)
            # NB choose a value to set which will be the same
            # on every process for each entity
            for cell in MeshEntities(mesh, i):
                mf[cell] = cell.midpoint()[0]
            meshfunctions.append(mf)
            mf_file.write(mf, "/meshfunction/meshfun%d" % i)

    # Read back from file
    with HDF5File(mesh.mpi_comm(), filename, "r") as mf_file:
        for i in range(0, 3):
            mf2 = MeshFunction('double', mesh, i, 0.0)
            mf_file.read(mf2, "/meshfunction/meshfun%d" % i)
            for cell in MeshEntities(mesh, i):
                assert meshfunctions[i][cell] == mf2[cell]


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_meshfunction_3D(tempdir):
    filename = os.path.join(tempdir, "meshfn-3d.h5")

    # Write to file
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    mf_file = HDF5File(mesh.mpi_comm(), filename, "w")

    # save meshfuns to compare when reading back
    meshfunctions = []
    for i in range(0, 4):
        mf = MeshFunction('double', mesh, i, 0.0)
        # NB choose a value to set which will be the same
        # on every process for each entity
        for cell in MeshEntities(mesh, i):
            mf[cell] = cell.midpoint()[0]
        meshfunctions.append(mf)
        mf_file.write(mf, "/meshfunction/group/%d/meshfun" % i)
    mf_file.close()

    # Read back from file
    mf_file = HDF5File(mesh.mpi_comm(), filename, "r")
    for i in range(0, 4):
        mf2 = MeshFunction('double', mesh, i, 0.0)
        mf_file.read(mf2, "/meshfunction/group/%d/meshfun" % i)
        for cell in MeshEntities(mesh, i):
            assert meshfunctions[i][cell] == mf2[cell]
    mf_file.close()


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_mesh_value_collection(tempdir):
    ndiv = 2
    filename = os.path.join(tempdir, "mesh_value_collection.h5")
    mesh = UnitCubeMesh(MPI.comm_world, ndiv, ndiv, ndiv)

    def point2list(p): return [p[0], p[1], p[2]]

    # write to file
    with HDF5File(mesh.mpi_comm(), filename, 'w') as f:
        for dim in range(mesh.topology().dim()):
            mvc = MeshValueCollection("size_t", mesh, dim)
            mesh.init(dim)
            for e in MeshEntities(mesh, dim):
                # this can be easily computed to the check the value
                val = int(ndiv*sum(point2list(e.midpoint()))) + 1
                mvc.set_value(e.index(), val)
            f.write(mvc, "/mesh_value_collection_{}".format(dim))

    # read from file
    with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
        for dim in range(mesh.topology().dim()):
            mvc = MeshValueCollection("size_t", mesh, dim)
            f.read(mvc, "/mesh_value_collection_{}".format(dim))
            # check the values
            for (cell, lidx), val in mvc.values().items():
                eidx = Cell(mesh, cell).entities(dim)[lidx]
                mid = point2list(MeshEntity(mesh, dim, eidx).midpoint())
                assert val == int(ndiv*sum(mid)) + 1


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_mesh_value_collection_with_only_one_marked_entity(tempdir):
    ndiv = 2
    filename = os.path.join(tempdir, "mesh_value_collection.h5")
    mesh = UnitCubeMesh(MPI.comm_world, ndiv, ndiv, ndiv)
    mvc = MeshValueCollection("size_t", mesh, 3)
    mesh.init(3)
    if MPI.rank(mesh.mpi_comm()) == 0:
        mvc.set_value(0, 1)

    # write to file
    with HDF5File(mesh.mpi_comm(), filename, 'w') as f:
        f.write(mvc, "/mesh_value_collection")

    # read from file
    with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
        mvc = MeshValueCollection("size_t", mesh, 3)
        f.read(mvc, "/mesh_value_collection")
        assert MPI.sum(mesh.mpi_comm(), mvc.size()) == 1
        if MPI.rank(mesh.mpi_comm()) == 0:
            assert mvc.get_value(0, 0) == 1


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_function(tempdir):
    filename = os.path.join(tempdir, "function.h5")

    mesh = UnitSquareMesh(MPI.comm_world, 10, 10)
    Q = FunctionSpace(mesh, "CG", 3)
    F0 = Function(Q)
    F1 = Function(Q)
    E = Expression("x[0]", degree=1)
    F0.interpolate(E)

    # Save to HDF5 File

    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    hdf5_file.write(F0, "/function")
    hdf5_file.close()

    # Read back from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    F1 = hdf5_file.read_function(Q, "/function")
    result = F0.vector() - F1.vector()
    assert len(result.get_local().nonzero()[0]) == 0
    hdf5_file.close()


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_mesh_2D(tempdir):
    filename = os.path.join(tempdir, "mesh2d.h5")

    # Write to file
    mesh0 = UnitSquareMesh(MPI.comm_world, 20, 20)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    mesh_file.close()

    # Read from file
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh1 = mesh_file.read_mesh(MPI.comm_world, "/my_mesh", False)
    mesh_file.close()

    assert mesh0.num_entities_global(0) == mesh1.num_entities_global(0)
    dim = mesh0.topology().dim()
    assert mesh0.num_entities_global(dim) == mesh1.num_entities_global(dim)


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_save_and_read_mesh_3D(tempdir):
    filename = os.path.join(tempdir, "mesh3d.h5")

    # Write to file
    mesh0 = UnitCubeMesh(MPI.comm_world, 10, 10, 10)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    mesh_file.close()

    # Read from file
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh1 = mesh_file.read_mesh(MPI.comm_world, "/my_mesh", False)
    mesh_file.close()

    assert mesh0.num_entities_global(0) == mesh1.num_entities_global(0)
    dim = mesh0.topology().dim()
    assert mesh0.num_entities_global(dim) == mesh1.num_entities_global(dim)


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_mpi_atomicity(tempdir):
    comm_world = MPI.comm_world
    if MPI.size(comm_world) > 1:
        filename = os.path.join(tempdir, "mpiatomic.h5")
        with HDF5File(MPI.comm_world, filename, "w") as f:
            assert f.get_mpi_atomicity() is False
            f.set_mpi_atomicity(True)
            assert f.get_mpi_atomicity() is True
