import numpy
import ufl
from dolfinx import cpp, geometry
from dolfinx.geometry import BoundingBoxTree
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI


@skip_in_parallel
def test_manifold_point_search():
    # Simple two-triangle surface in 3d
    vertices = numpy.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = numpy.array([[0, 1, 2], [0, 1, 3]], dtype=numpy.int64)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
    x = mesh.geometry.x
    tdim = mesh.topology.dim

    bb = BoundingBoxTree(mesh, tdim)

    # Find cell colliding with point
    p = numpy.array([0.5, 0.25, 0.75])
    cell_candidates = geometry.compute_collisions_point(bb, p)
    cell = cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)

    # Extract vertices of cell
    top_indices = cpp.mesh.entities_to_geometry(mesh, tdim, [cell], False)
    cell_vertices = x[top_indices]

    # Compare vertices with input (should be in cell 0)
    assert numpy.allclose(cell_vertices, vertices[cells[0]])

    # Find cell colliding with point
    p = numpy.array([0.25, 0.5, 0.75])
    cell_candidates = geometry.compute_collisions_point(bb, p)
    cell = cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)

    # Extract vertices of cell
    top_indices = cpp.mesh.entities_to_geometry(mesh, tdim, [cell], False)
    x = mesh.geometry.x
    cell_vertices = x[top_indices]

    # Compare vertices with input (should be in cell 1)
    assert numpy.allclose(cell_vertices, vertices[cells[1]])
