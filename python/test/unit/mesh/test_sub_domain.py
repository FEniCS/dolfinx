# Copyright (C) 2013 Johan Hake
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import pytest
import numpy as np
from dolfin import *
from dolfin_utils.test import skip_in_parallel


def test_creation_and_marking():

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return x[:, 0] < DOLFIN_EPS

    class LeftOnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return np.logical_and(x[:, 0] < DOLFIN_EPS, on_boundary)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return x[:, 0] > 1.0 - DOLFIN_EPS

    class RightOnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return np.logical_and(x[:, 0] > 1.0 - DOLFIN_EPS, on_boundary)

    subdomain_pairs = [(Left(), Right()),
                       (LeftOnBoundary(), RightOnBoundary())]

    for ind, MeshClass in enumerate([UnitIntervalMesh, UnitSquareMesh,
                                     UnitCubeMesh]):
        dim = ind + 1
        args = [10]*dim
        mesh = MeshClass(MPI.comm_world, *args)
        mesh.init()

        for left, right in subdomain_pairs:
            for t_dim, f_dim in [(0, 0),
                                 (mesh.topology.dim-1, dim - 1),
                                 (mesh.topology.dim, dim)]:
                f = MeshFunction("size_t", mesh, t_dim, 0)

                left.mark(f, 1)
                right.mark(f, 2)

                correct = {(1, 0): 1,
                           (1, 0): 1,
                           (1, 1): 0,
                           (2, 0): 11,
                           (2, 1): 10,
                           (2, 2): 0,
                           (3, 0): 121,
                           (3, 2): 200,
                           (3, 3): 0}

                # Check that the number of marked entities are at least the
                # correct number (it can be larger in parallel)
                assert all(value >= correct[dim, f_dim]
                           for value in [
                    MPI.sum(mesh.mpi_comm(), float((f.array() == 2).sum())),
                    MPI.sum(mesh.mpi_comm(), float((f.array() == 1).sum())),
                ])

        for t_dim, f_dim in [(0, 0),
                             (mesh.topology.dim-1, dim-1),
                             (mesh.topology.dim, dim)]:
            f = MeshFunction("size_t", mesh, t_dim, 0)

            class AllTrue(SubDomain):
                def inside(self, x, on_boundary):
                    return np.full(x.shape[0], True)

            class AllFalse(SubDomain):
                def inside(self, x, on_boundary):
                    return np.full(x.shape[0], False)

            empty = AllFalse()
            every = AllTrue()
            empty.mark(f, 1)
            every.mark(f, 2)

            # Check that the number of marked entities is correct
            assert sum(f.array() == 1) == 0
            assert sum(f.array() == 2) == mesh.num_entities(f_dim)
