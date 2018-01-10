
import numpy
from dolfin import *


def test_manifold_point_search():
    # Simple two-triangle surface in 3d
    vertices = [
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        ]
    cells = [
        (0, 1, 2),
        (0, 1, 3),
    ]
    mesh = Mesh(MPI.comm_world)
    me = MeshEditor()
    me.open(mesh, CellType.Type.triangle, 2, 3)
    me.init_vertices_global(len(vertices), len(vertices))
    for i, v in enumerate(vertices):
        me.add_vertex(i, Point(v))
    me.init_cells_global(len(cells), len(cells))
    for i, c in enumerate(cells):
        me.add_cell(i, numpy.array(c, dtype='uint'))
    me.close()

#    mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0"), degree=0))

    bb = mesh.bounding_box_tree()
    p = Point(0.5, 0.25, 0.75)
    assert bb.compute_first_entity_collision(p) == 0

    p = Point(0.25, 0.5, 0.75)
    assert bb.compute_first_entity_collision(p) == 1
