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
    vertices = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    cells = [(0, 1, 2), (0, 1, 3)]
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, vertices, domain)

    bb = BoundingBoxTree(mesh, mesh.topology.dim)
    p = numpy.array([0.5, 0.25, 0.75])
    cell_candidates = geometry.compute_collisions_point(bb, p)
    cell = cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)
    assert cell[0] == 0

    p = numpy.array([0.25, 0.5, 0.75])
    cell_candidates = geometry.compute_collisions_point(bb, p)
    cell = cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)
    assert cell[0] == 1
