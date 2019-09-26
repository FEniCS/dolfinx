import numpy

from dolfin import MPI, Mesh, cpp, geometry
from dolfin.cpp.mesh import CellType
from dolfin.geometry import BoundingBoxTree
from dolfin_utils.test.skips import skip_in_parallel


@skip_in_parallel
def test_manifold_point_search():
    # Simple two-triangle surface in 3d
    vertices = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0,
                                                                    0.0)]
    cells = [(0, 1, 2), (0, 1, 3)]
    mesh = Mesh(MPI.comm_world, CellType.triangle,
                numpy.array(vertices, dtype=numpy.float64),
                numpy.array(cells, dtype=numpy.int32), [],
                cpp.mesh.GhostMode.none)

    bb = BoundingBoxTree(mesh, mesh.topology.dim)
    p = numpy.array([0.5, 0.25, 0.75])
    assert geometry.compute_first_entity_collision(bb, mesh, p) == 0

    p = numpy.array([0.25, 0.5, 0.75])
    assert geometry.compute_first_entity_collision(bb, mesh, p) == 1
