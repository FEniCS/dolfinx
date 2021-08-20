# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from IPython import embed
import dolfinx
from mpi4py import MPI
import numpy as np


mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8)

for i in range(5):
    t0 = dolfinx.common.Timer(f"~Test something {i}")
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    t0.stop()
dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

output = dolfinx.common.timing("~Test something 0")
embed()
exit()


# def test_index_map_compression():

comm = MPI.COMM_WORLD

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8)
vertex_map = mesh.topology.index_map(0)
s_l = vertex_map.size_local
org_ghosts = vertex_map.ghosts
org_ghost_owners = vertex_map.ghost_owner_rank()

# Create an index map where only every third local index is saved
entities = []
for i in range(s_l):
    if i % 4 == 0:
        entities.append(i)

# Add every fourth ghost
owners = []
for i in range(len(org_ghosts)):
    if i % 4 == 0:
        entities.append(s_l + i)
        owners.append(org_ghost_owners[i])

# Create compressed index map
entities = np.array(entities, dtype=np.int32)
new_map, org_glob = dolfinx.cpp.common.compress_index_map(vertex_map, entities)

ghost_owners = new_map.ghost_owner_rank()
ghosts = new_map.ghosts
new_sl = new_map.size_local
for i in range(len(ghosts)):
    print(len(org_glob), new_sl)
    #assert(org_glob[new_sl + i] == ghosts[i])
    # print(len(org_glob), new_sl, i, len(ghosts))
    # index = np.argwhere(org_ghosts == ghosts[i])[0, 0]
    # print(index, org_ghosts)
    # print(org_ghost_owners[index], ghost_owners[i])
    # print(MPI.COMM_WORLD.rank, new_map.size_local, vertex_map.size_local, org_glob)
