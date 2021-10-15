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
    points = numpy.array([[0.5, 0.25, 0.75], [0.25, 0.5, 0.75]])
    cell_candidates = geometry.compute_collisions(bb, points)
    colliding_cells = cpp.geometry.select_colliding_cells(mesh, cell_candidates, points)

    # Extract vertices of cell
    indices = cpp.mesh.entities_to_geometry(
        mesh, tdim, [colliding_cells.links(0)[0], colliding_cells.links(1)[0]], False)
    cell_vertices = x[indices]
    # Compare vertices with input
    assert numpy.allclose(cell_vertices, vertices[cells])
