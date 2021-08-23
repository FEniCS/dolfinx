# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from IPython import embed
import dolfinx
from mpi4py import MPI
import numpy as np


class MeshView:

    def __init__(self, mesh: dolfinx.cpp.mesh.Mesh, dim: int, entities: np.ndarray):
        mesh.topology.create_connectivity(dim, 0)
        e_to_v = mesh.topology.connectivity(dim, 0)

        # For each entity, find all vertices
        view_vertices = [np.array([], dtype=np.int32)]
        for entity in entities:
            vertices = e_to_v.links(entity)
            view_vertices.append(vertices)
        view_vertices = np.hstack(view_vertices)
        # Create compressed index map
        view_vertices = np.unique(view_vertices)

        org_vertex_map = mesh.topology.index_map(0)
        # out_str = f"---{MPI.COMM_WORLD.rank}--\n"
        # out_str += "Parent vertices on process" + f"{view_vertices}\n"
        # out_str += "Parent num local:" + f"{org_vertex_map.size_local}\n" + "Parent range:" + \
        #     f"[{org_vertex_map.local_range[0]}, {org_vertex_map.local_range[1]})\n"
        # out_str += "Parent ghosts: " + f"{org_vertex_map.ghosts}\n"
        vertex_map, glob_map = dolfinx.cpp.common.compress_index_map(org_vertex_map, view_vertices)

        # Get original local indices of new index map
        org_local = org_vertex_map.global_to_local(glob_map)

        # out_str += f"Child range: [{vertex_map.local_range[0]}, {vertex_map.local_range[1]})\n"
        # out_str += f"Child ghosts {vertex_map.ghosts} \n"
        # out_str += f"Child local to parent global: {glob_map} \n"
        # out_str += f"Child local to parent local {org_local}\n"

        # Create map from original vertex index to compressed vertex index
        old_offset = np.zeros(org_vertex_map.size_local + org_vertex_map.num_ghosts + 1, dtype=np.int32)
        old_data = np.zeros(org_local.size, dtype=np.int32)
        sort_index = np.argsort(org_local)

        for index in org_local:
            old_offset[index + 1:] += 1
        for i, index in enumerate(sort_index):
            old_data[i] = index
        new_adj = dolfinx.cpp.graph.AdjacencyList_int32(old_data, old_offset)

        # out_str += str(new_adj)
        # print(out_str)
        # Create compressed index map for entities on the process
        org_entity_map = mesh.topology.index_map(dim)
        entity_map, c_map = dolfinx.cpp.common.compress_index_map(org_entity_map, entities)
        local_cells = org_entity_map.global_to_local(c_map)

        # Create new entity-vertex connectivity
        data = []
        offsets = [0]
        for i in range(len(local_cells)):
            vertices = e_to_v.links(local_cells[i])
            for vertex in vertices:
                data.append(new_adj.links(vertex)[0])
            offsets.append(len(data))
        e_to_v_new = dolfinx.cpp.graph.AdjacencyList_int32(data, offsets)

        # Create vertex to vertex map (is identity)
        num_vertices = vertex_map.size_local + vertex_map.num_ghosts
        v_map = np.arange(num_vertices, dtype=np.int32)
        offsets = np.arange(num_vertices + 1, dtype=np.int32)
        v_to_v = dolfinx.cpp.graph.AdjacencyList_int32(v_map, offsets)

        cell_type = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, dim)
        self.topology = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, cell_type)
        self.topology.set_index_map(0, vertex_map)
        self.topology.set_index_map(dim, entity_map)
        self.topology.set_connectivity(v_to_v, 0, 0)
        self.topology.set_connectivity(e_to_v_new, dim, 0)

        self.child_vertex_map = org_local
        self.child_entity_map = local_cells


# def test_meshview():
mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 4, 2, 3)
# mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 2)
dim = mesh.topology.dim

cells = dolfinx.mesh.locate_entities(mesh, dim, lambda x: x[0] <= 0.5)

mv_cpp = dolfinx.cpp.mesh.MeshView(mesh, dim, cells)

mv = MeshView(mesh, dim, cells)
e_to_v = mesh.topology.connectivity(dim, 0)
e_to_v_new = mv.topology.connectivity(dim, 0)
e_to_v_cpp = mv_cpp.topology.connectivity(dim, 0)

out_str = f"---{MPI.COMM_WORLD.rank}----\n"
out_str += f"Cells: {cells}\n"
out_str += f"Child->Parent vertex map {mv.child_vertex_map}"
out_str += "---Org---\n"
for cell in range(mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts):
    out_str += f"{cell}, {e_to_v.links(cell)}\n"
out_str += "---new---\n"
for i in range(e_to_v_new.num_nodes):
    out_str += f"{i}, {e_to_v_new.links(i)}\n"
out_str += "---new mapped---\n"
for i in range(e_to_v_new.num_nodes):
    out_str += f"{mv.child_entity_map[i]}, {mv.child_vertex_map[e_to_v_new.links(i)]}\n"
print(out_str)
for i in range(e_to_v_new.num_nodes):
    assert np.allclose(e_to_v.links(mv.child_entity_map[i]), mv.child_vertex_map[e_to_v_new.links(i)])
    assert np.allclose(e_to_v.links(mv.child_entity_map[i]), mv_c.child_vertex_map[e_to_v_new.links(i)])
