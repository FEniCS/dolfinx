#!/usr/bin/env py.test
import pytest
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
    mesh = Mesh()
    me = MeshEditor()
    me.open(mesh, "triangle", 2, 3)
    me.init_vertices(len(vertices))
    for i, v in enumerate(vertices):
        me.add_vertex(i, *v)
    me.init_cells(len(cells))
    for i, c in enumerate(cells):
        me.add_cell(i, *c)
    me.close()

    mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0"), degree=0))

    bb = mesh.bounding_box_tree()
    #p = Point(2.0/3.0, 1.0/3.0, 2.0/3.0)
    p = Point(0.5, 0.25, 0.75)
    assert bb.compute_first_entity_collision(p) == 0

    #p = Point(1.0/3.0, 2.0/3.0, 2.0/3.0)
    p = Point(0.25, 0.5, 0.75)
    assert bb.compute_first_entity_collision(p) == 1
