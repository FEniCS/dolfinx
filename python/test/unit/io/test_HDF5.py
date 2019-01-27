# Copyright (C) 2012 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

from petsc4py import PETSc

from dolfin import (MPI, Cell, Expression, Function, FunctionSpace,
                    MeshEntities, MeshEntity, MeshFunction,
                    MeshValueCollection, UnitCubeMesh, UnitSquareMesh, cpp,
                    function)
from dolfin.io import HDF5File
from dolfin_utils.test.fixtures import tempdir
from dolfin_utils.test.skips import xfail_if_complex

assert (tempdir)


def test_parallel(tempdir):
    filename = os.path.join(tempdir, "y.h5")
    hdf5 = HDF5File(MPI.comm_world, filename, "w")
    assert (hdf5)


@xfail_if_complex
def test_save_and_read_vector(tempdir):
    filename = os.path.join(tempdir, "vector.h5")

    # Write to file
    local_range = MPI.local_range(MPI.comm_world, 305)
    x = PETSc.Vec()
    x.create(MPI.comm_world)
    x.setSizes((local_range[1] - local_range[0], None))
    x.setFromOptions()
    x.set(1.2)
    with HDF5File(MPI.comm_world, filename, "w") as vector_file:
        vector_file.write(x, "/my_vector")

    # Read from file
    with HDF5File(MPI.comm_world, filename, "r") as vector_file:
        y = vector_file.read_vector(MPI.comm_world, "/my_vector", False)
        assert y.getSize() == x.getSize()
        x.axpy(-1.0, y)
        assert x.norm() == 0.0


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
            mf2 = mf_file.read_mf_double(mesh, "/meshfunction/meshfun%d" % i)
            for cell in MeshEntities(mesh, i):
                assert meshfunctions[i][cell] == mf2[cell]


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
        mf2 = mf_file.read_mf_double(mesh,
                                     "/meshfunction/group/%d/meshfun" % i)
        for cell in MeshEntities(mesh, i):
            assert meshfunctions[i][cell] == mf2[cell]
    mf_file.close()


def test_save_and_read_mesh_value_collection(tempdir):
    ndiv = 2
    filename = os.path.join(tempdir, "mesh_value_collection.h5")
    mesh = UnitCubeMesh(MPI.comm_world, ndiv, ndiv, ndiv)

    def point2list(p):
        return [p[0], p[1], p[2]]

    # write to file
    with HDF5File(mesh.mpi_comm(), filename, 'w') as f:
        for dim in range(mesh.topology.dim):
            mvc = MeshValueCollection("size_t", mesh, dim)
            mesh.init(dim)
            for e in MeshEntities(mesh, dim):
                # this can be easily computed to the check the value
                val = int(ndiv * sum(point2list(e.midpoint()))) + 1
                mvc.set_value(e.index(), val)
            f.write(mvc, "/mesh_value_collection_{}".format(dim))

    # read from file
    with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
        for dim in range(mesh.topology.dim):
            mvc = f.read_mvc_size_t(mesh,
                                    "/mesh_value_collection_{}".format(dim))
            # check the values
            for (cell, lidx), val in mvc.values().items():
                eidx = Cell(mesh, cell).entities(dim)[lidx]
                mid = point2list(MeshEntity(mesh, dim, eidx).midpoint())
                assert val == int(ndiv * sum(mid)) + 1


def test_save_and_read_mesh_value_collection_with_only_one_marked_entity(
        tempdir):
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
        mvc = f.read_mvc_size_t(mesh, "/mesh_value_collection")
        assert MPI.sum(mesh.mpi_comm(), mvc.size()) == 1
        if MPI.rank(mesh.mpi_comm()) == 0:
            assert mvc.get_value(0, 0) == 1


@xfail_if_complex
def test_save_and_read_function(tempdir):
    filename = os.path.join(tempdir, "function.h5")

    mesh = UnitSquareMesh(MPI.comm_world, 10, 10)
    Q = FunctionSpace(mesh, ("CG", 3))
    F0 = Function(Q)
    F1 = Function(Q)

    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = x[:, 0]

    E = Expression(expr_eval)
    F0.interpolate(E)

    # Save to HDF5 File

    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "w")
    hdf5_file.write(F0, "/function")
    hdf5_file.close()

    # Read back from file
    hdf5_file = HDF5File(mesh.mpi_comm(), filename, "r")
    F1 = hdf5_file.read_function(Q, "/function")
    F0.vector().axpy(-1.0, F1.vector())
    assert F0.vector().norm() < 1.0e-12
    hdf5_file.close()


def test_save_and_read_mesh_2D(tempdir):
    filename = os.path.join(tempdir, "mesh2d.h5")

    # Write to file
    mesh0 = UnitSquareMesh(MPI.comm_world, 20, 20)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    mesh_file.close()

    # Read from file
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh1 = mesh_file.read_mesh(MPI.comm_world, "/my_mesh", False,
                                cpp.mesh.GhostMode.none)
    mesh_file.close()

    assert mesh0.num_entities_global(0) == mesh1.num_entities_global(0)
    dim = mesh0.topology.dim
    assert mesh0.num_entities_global(dim) == mesh1.num_entities_global(dim)


def test_save_and_read_mesh_3D(tempdir):
    filename = os.path.join(tempdir, "mesh3d.h5")

    # Write to file
    mesh0 = UnitCubeMesh(MPI.comm_world, 10, 10, 10)
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "w")
    mesh_file.write(mesh0, "/my_mesh")
    mesh_file.close()

    # Read from file
    mesh_file = HDF5File(mesh0.mpi_comm(), filename, "r")
    mesh1 = mesh_file.read_mesh(MPI.comm_world, "/my_mesh", False,
                                cpp.mesh.GhostMode.none)
    mesh_file.close()

    assert mesh0.num_entities_global(0) == mesh1.num_entities_global(0)
    dim = mesh0.topology.dim
    assert mesh0.num_entities_global(dim) == mesh1.num_entities_global(dim)


def test_mpi_atomicity(tempdir):
    comm_world = MPI.comm_world
    if MPI.size(comm_world) > 1:
        filename = os.path.join(tempdir, "mpiatomic.h5")
        with HDF5File(MPI.comm_world, filename, "w") as f:
            assert f.get_mpi_atomicity() is False
            f.set_mpi_atomicity(True)
            assert f.get_mpi_atomicity() is True
