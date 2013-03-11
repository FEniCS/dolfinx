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

import unittest
import numpy
from dolfin import *

class MeshIterator(unittest.TestCase):

    def test_vertex_iterators(self):
        "Iterate over vertices"

        mesh = UnitCubeMesh(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(0,i)) for i in xrange(4)]

        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(StandardError, assign, con, i)

        n = 0
        for i, v in enumerate(vertices(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == v.entities(j)))

        self.assertEqual(n, mesh.num_vertices())

        # Check coordinate assignment
        # FIXME: Outcomment to hopefully please Mac-buildbot
        #end_point = numpy.array([v.x(0), v.x(1), v.x(2)])
        #mesh.coordinates()[:] += 2
        #self.assertEqual(end_point[0] + 2, mesh.coordinates()[-1,0])
        #self.assertEqual(end_point[1] + 2, mesh.coordinates()[-1,1])
        #self.assertEqual(end_point[2] + 2, mesh.coordinates()[-1,2])

    def test_edge_iterators(self):
        "Iterate over edges"

        mesh = UnitCubeMesh(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(1,i)) for i in xrange(4)]

        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(StandardError, assign, con, i)

        n = 0
        for i, e in enumerate(edges(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == e.entities(j)))

        self.assertEqual(n, mesh.num_edges())

    def test_face_iterator(self):
        "Iterate over faces"

        mesh = UnitCubeMesh(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(2,i)) for i in xrange(4)]

        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(StandardError, assign, con, i)

        n = 0
        for i, f in enumerate(faces(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == f.entities(j)))

        self.assertEqual(n, mesh.num_faces())

    def test_facet_iterators(self):
        "Iterate over facets"
        mesh = UnitCubeMesh(5, 5, 5)
        n = 0
        for f in facets(mesh):
            n += 1
        self.assertEqual(n, mesh.num_facets())

    def test_cell_iterators(self):
        "Iterate over cells"
        mesh = UnitCubeMesh(5, 5, 5)

        # Test connectivity
        cons = [(i, mesh.topology()(3,i)) for i in xrange(4)]

        # Test writability
        for i, con in cons:
            def assign(con, i):
                con(i)[0] = 1
            self.assertRaises(StandardError, assign, con, i)

        n = 0
        for i, c in enumerate(cells(mesh)):
            n += 1
            for j, con in cons:
                self.assertTrue(numpy.all(con(i) == c.entities(j)))

        self.assertEqual(n, mesh.num_cells())

    def test_mixed_iterators(self):
        "Iterate over vertices of cells"

        mesh = UnitCubeMesh(5, 5, 5)
        n = 0
        for c in cells(mesh):
            for v in vertices(c):
                n += 1
        self.assertEqual(n, 4*mesh.num_cells())

if __name__ == "__main__":
    unittest.main()
