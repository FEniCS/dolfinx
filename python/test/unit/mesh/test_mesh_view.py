# TODO When finished, add to test_mesh.py since "meshview" is a mesh

import dolfinx
from dolfinx.generation import UnitSquareMesh
from mpi4py import MPI
import numpy as np
from dolfinx.cpp.mesh import entities_to_geometry
import pytest


def boundary_0(x):
    lr = np.logical_or(np.isclose(x[0], 0.0),
                       np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0),
                       np.isclose(x[1], 1.0))
    return np.logical_or(lr, tb)


def boundary_1(x):
    return np.logical_or(np.isclose(x[0], 1.0),
                         np.isclose(x[1], 1.0))


boundaries = [boundary_0, boundary_1]
ns = [1, 2, 3]


@pytest.mark.parametrize("boundary", boundaries)
@pytest.mark.parametrize("n", ns)
def test_facet_topology(n, boundary):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
    entity_dim = mesh.topology.dim - 1
    entities = dolfinx.mesh.locate_entities_boundary(mesh, entity_dim, boundary)
    topology_test(mesh, entity_dim, entities)


@pytest.mark.parametrize("boundary", boundaries)
@pytest.mark.parametrize("n", ns)
def test_geometry(n, boundary):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
    entity_dim = mesh.topology.dim - 1
    entities = dolfinx.mesh.locate_entities_boundary(mesh, entity_dim, boundary)
    geometry_test(mesh, entity_dim, entities)


def topology_test(mesh, entity_dim, entities):
    submesh = mesh.sub(entity_dim, entities)

    # The vertex map that mesh.sub uses is a sorted list of unique vertices, so
    # recreate this here. TODO Could return this from mesh.sub or save as a property
    # etc.
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


def geometry_test(mesh, entity_dim, entities):
    submesh = mesh.sub(entity_dim, entities)

    assert(mesh.geometry.dim == submesh.geometry.dim)

    e_to_g = entities_to_geometry(mesh, entity_dim, entities, False)

    for submesh_entity in range(len(entities)):
        submesh_x_dofs = submesh.geometry.dofmap.links(submesh_entity)

        # e_to_g[i] gets the mesh x_dofs of entities[i], which should
        # correspond to the submesh x_dofs of submesh cell i
        mesh_x_dofs = e_to_g[submesh_entity]

        for i in range(len(submesh_x_dofs)):
            assert(np.allclose(mesh.geometry.x[mesh_x_dofs[i]],
                               submesh.geometry.x[submesh_x_dofs[i]]))
