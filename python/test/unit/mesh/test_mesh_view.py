# TODO When finished, add to test_mesh.py since "meshview" is a mesh

import dolfinx
from mpi4py import MPI
import numpy as np


def boundary_0(x):
    lr = np.logical_or(np.isclose(x[0], 0.0),
                       np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0),
                       np.isclose(x[1], 1.0))
    return np.logical_or(lr, tb)


def boundary_1(x):
    return np.logical_or(np.isclose(x[0], 1.0),
                         np.isclose(x[1], 1.0))


n = 1
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n, n)
entity_dim = mesh.topology.dim - 1
entities = dolfinx.mesh.locate_entities_boundary(mesh, entity_dim, boundary_1)

submesh = mesh.sub(entity_dim, entities)

# Check topology
print(f"entities = {entities}")
# The vertex map that mesh.sub uses is a sorted list of unique vertices, so
# recreate this here. TODO Could return this from mesh.sub.
mesh.topology.create_connectivity(entity_dim, 0)
mesh_e_to_v = mesh.topology.connectivity(entity_dim, 0)
vertex_map = []
for entity in entities:
    vertices = mesh_e_to_v.links(entity)
    vertex_map.append(vertices)
vertex_map = np.hstack(vertex_map)
vertex_map = np.unique(vertex_map)

submesh.topology.create_connectivity(entity_dim, 0)
submesh_e_to_v = submesh.topology.connectivity(entity_dim, 0)
for submesh_entity in range(len(entities)):
    submesh_entity_vertices = submesh_e_to_v.links(submesh_entity)
    mesh_entity = entities[submesh_entity]
    mesh_entity_vertices = mesh_e_to_v.links(mesh_entity)
    
    for i in range(len(submesh_entity_vertices)):
        assert(vertex_map[submesh_entity_vertices[i]] == mesh_entity_vertices[i])
