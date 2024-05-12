from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx import cpp as _cpp
from dolfinx import default_real_type, geometry
from dolfinx.geometry import bb_tree
from dolfinx.mesh import create_mesh


@pytest.mark.skip_in_parallel
def test_manifold_point_search():
    # Simple two-triangle surface in 3d
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    mesh = create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
    bb = bb_tree(mesh, mesh.topology.dim)

    # Find cell colliding with point
    points = np.array([[0.5, 0.25, 0.75], [0.25, 0.5, 0.75]], dtype=default_real_type)
    cell_candidates = geometry.compute_collisions_points(bb, points)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)

    # Extract vertices of cell
    mesh.topology.create_entity_permutations()
    indices = _cpp.mesh.entities_to_geometry(
        mesh._cpp_object,
        mesh.topology.dim,
        np.array([colliding_cells.links(0)[0], colliding_cells.links(1)[0]]),
    )
    cell_vertices = mesh.geometry.x[indices]

    # Compare vertices with input
    assert np.allclose(cell_vertices, vertices[cells])
