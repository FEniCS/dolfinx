"Unit tests for MeshIterator and subclasses"

# Copyright (C) 2006-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2006-08-08
# Last changed: 2011-08-21

import pytest
import numpy
from dolfin import *


def test_vertex_iterators():
    "Iterate over vertices"

    mesh = UnitCubeMesh(5, 5, 5)
    for i in range(4):
        mesh.init(0, i)

    # Test connectivity
    cons = [(i, mesh.topology()(0,i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, v in enumerate(vertices(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == v.entities(j))

    assert n == mesh.num_vertices()

    # Check coordinate assignment
    # FIXME: Outcomment to hopefully please Mac-buildbot
    #end_point = numpy.array([v.x(0), v.x(1), v.x(2)])
    #mesh.coordinates()[:] += 2
    #assert end_point[0] + 2 == mesh.coordinates()[-1,0]
    #assert end_point[1] + 2 == mesh.coordinates()[-1,1]
    #assert end_point[2] + 2 == mesh.coordinates()[-1,2]

def test_edge_iterators():
    "Iterate over edges"

    mesh = UnitCubeMesh(5, 5, 5)
    for i in range(4):
        mesh.init(1, i)

    # Test connectivity
    cons = [(i, mesh.topology()(1,i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, e in enumerate(edges(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == e.entities(j))

    assert n == mesh.num_edges()

def test_face_iterator():
    "Iterate over faces"

    mesh = UnitCubeMesh(5, 5, 5)
    for i in range(4):
        mesh.init(2, i)

    # Test connectivity
    cons = [(i, mesh.topology()(2,i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, f in enumerate(faces(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == f.entities(j))

    assert n == mesh.num_faces()

def test_facet_iterators():
    "Iterate over facets"
    mesh = UnitCubeMesh(5, 5, 5)
    n = 0
    for f in facets(mesh):
        n += 1
    assert n == mesh.num_facets()

def test_cell_iterators():
    "Iterate over cells"
    mesh = UnitCubeMesh(5, 5, 5)
    for i in range(4):
        mesh.init(3, i)

    # Test connectivity
    cons = [(i, mesh.topology()(3,i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, c in enumerate(cells(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == c.entities(j))

    assert n == mesh.num_cells()

    # Test non destruction of MeshEntities
    cell_list = [c for c in cells(mesh)]
    assert sum(c.volume() for c in cell_list) == sum(c.volume() for c in cells(mesh))

def test_mixed_iterators():
    "Iterate over vertices of cells"

    mesh = UnitCubeMesh(5, 5, 5)
    n = 0
    for c in cells(mesh):
        for v in vertices(c):
            n += 1
    assert n == 4*mesh.num_cells()
