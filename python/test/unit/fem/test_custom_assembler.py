# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom assemblers"""


from numba import jit

import dolfin


def test_custom_mesh_loop():

    @jit(nopython=True)
    def cell_loop(connections, pos):
        test = 0
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i+1] - pos[i]
            c = connections[cell:cell+num_vertices]
            for v in c:
                test += v*v
        return test

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 256, 256)
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    test = cell_loop(c, pos)
    print("ncells:", mesh.num_cells(), len(pos), test)
