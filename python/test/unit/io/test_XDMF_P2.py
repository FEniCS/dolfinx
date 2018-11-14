# Copyright (C) 2018 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

from dolfin import (MPI, Expression, Function, FunctionSpace,
                    VectorFunctionSpace, cpp, fem, has_petsc_complex)
from dolfin import function
from dolfin.io import XDMFFile
from dolfin_utils.test.fixtures import tempdir

assert (tempdir)


def test_read_write_p2_mesh(tempdir):
    mesh = cpp.generation.UnitDiscMesh.create(MPI.comm_world, 3,
                                              cpp.mesh.GhostMode.none)

    filename = os.path.join(tempdir, "tri6_mesh.xdmf")
    with XDMFFile(
            mesh.mpi_comm(), filename,
            encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write(mesh)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        mesh2 = xdmf.read_mesh(mesh.mpi_comm(), cpp.mesh.GhostMode.none)

    assert mesh2.num_entities_global(
        mesh.topology.dim) == mesh.num_entities_global(mesh.topology.dim)
    assert mesh2.num_entities_global(0) == mesh.num_entities_global(0)


def test_read_write_p2_function(tempdir):
    mesh = cpp.generation.UnitDiscMesh.create(MPI.comm_world, 3,
                                              cpp.mesh.GhostMode.none)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    Q = FunctionSpace(mesh, ("Lagrange", 2))

    F = Function(Q)
    if has_petsc_complex:
        @function.expression.numba_eval
        def expr_eval(values, x, cell_idx):
            values[:, 0] = x[:, 0] + 1.0j * x[:, 0]
        F.interpolate(Expression(expr_eval))
    else:
        @function.expression.numba_eval
        def expr_eval(values, x, cell_idx):
            values[:, 0] = x[:, 0]
        F.interpolate(Expression(expr_eval))

    filename = os.path.join(tempdir, "tri6_function.xdmf")
    with XDMFFile(
            mesh.mpi_comm(), filename,
            encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write(F)

    Q = VectorFunctionSpace(mesh, ("Lagrange", 1))
    F = Function(Q)
    if has_petsc_complex:
        @function.expression.numba_eval
        def expr_eval(values, x, cell_idx):
            values[:, 0] = x[:, 0] + 1.0j * x[:, 0]
            values[:, 1] = x[:, 1] + 1.0j * x[:, 1]
        F.interpolate(Expression(expr_eval, shape=(2,)))
    else:
        @function.expression.numba_eval
        def expr_eval(values, x, cell_idx):
            values[:, 0] = x[:, 0]
            values[:, 1] = x[:, 1]
        F.interpolate(Expression(expr_eval, shape=(2,)))
    filename = os.path.join(tempdir, "tri6_vector_function.xdmf")
    with XDMFFile(
            mesh.mpi_comm(), filename,
            encoding=XDMFFile.Encoding.HDF5) as xdmf:
        xdmf.write(F)
