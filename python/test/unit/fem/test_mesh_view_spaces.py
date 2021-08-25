# Copyright (C) 2021 Joseph P. Dean
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# TODO When cell / facet spaces are working properly, improve these tests.
# They are currently very simplistic.
import dolfinx
from mpi4py import MPI


def check_dofmap(mesh, dim):
    entities = dolfinx.mesh.locate_entities(mesh, dim,
                                            lambda x: x[0] <= 0.5)
    mv_cpp = dolfinx.cpp.mesh.MeshView(mesh, dim, entities)

    V = dolfinx.FunctionSpace(mv_cpp, ("Lagrange", 1))

    assert V.dofmap.list.num_nodes == len(entities)


def test_mesh_view_spaces():
    meshes = [dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 2),
              dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 4, 2, 2)]
    for mesh in meshes:
        for codim in range(0, 2):
            dim = mesh.topology.dim - codim
            check_dofmap(mesh, dim)
