import numpy as np
import pytest

import ufl
from dolfinx import cpp as _cpp
from dolfinx import geometry
from dolfinx.geometry import BoundingBoxTree
from dolfinx.mesh import create_mesh

from mpi4py import MPI


@pytest.mark.skip_in_parallel
def test_manifold_point_search():
    # Simple two-triangle surface in 3d
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
    bb = BoundingBoxTree(mesh, mesh.topology.dim)

    # Find cell colliding with point
    points = np.array([[0.5, 0.25, 0.75], [0.25, 0.5, 0.75]])
    cell_candidates = geometry.compute_collisions(bb, points)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)

    # Extract vertices of cell
    indices = _cpp.mesh.entities_to_geometry(mesh, mesh.topology.dim, [colliding_cells.links(0)[
        0], colliding_cells.links(1)[0]], False)
    cell_vertices = mesh.geometry.x[indices]

    # Compare vertices with input
    assert np.allclose(cell_vertices, vertices[cells])
