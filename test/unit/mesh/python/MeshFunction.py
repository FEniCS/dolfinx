"""Unit tests for MeshFunctions"""

# Copyright (C) 2011 Garth N. Wells
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
# First added:  2011-03-10
# Last changed: 2011-03-10

import unittest
import numpy.random
from dolfin import *

class NamedMeshFunctions(unittest.TestCase):

    def setUp(self):
        #self.names = ["Cell", "Vertex", "Edge", "Face", "Facet"]
        #self.tps = ['int', 'uint', 'bool', 'double']
        self.names = ["Cell", "Vertex", "Edge", "Face", "Facet"]
        self.tps = ['int', 'uint', 'bool', 'double']
        self.mesh = UnitCubeMesh(3, 3, 3)
        self.funcs = {}
        for tp in self.tps:
            for name in self.names:
                self.funcs[(tp, name)] = eval("%sFunction('%s', self.mesh)"%\
                                              (name, tp))

    def test_size(self):
        for tp in self.tps:
            for name in self.names:
                if name is "Vertex":
                    a = self.funcs[(tp, name)].size()
                    b = self.mesh.num_vertices()
                    self.assertEqual(a, b)
                else:
                    a = self.funcs[(tp, name)].size()
                    b = getattr(self.mesh, "num_%ss"%name.lower())()
                    self.assertEqual(a, b)

    def test_access_type(self):
        type_dict = dict(int=int, uint=int, double=float, bool=bool)
        for tp in self.tps:
            for name in self.names:
                self.assertTrue(isinstance(self.funcs[(tp, name)][0], \
                                           type_dict[tp]))

    def test_numpy_access(self):
        for tp in self.tps:
            for name in self.names:
                values = self.funcs[(tp, name)].array()
                values[:] = numpy.random.rand(len(values))
                self.assertTrue(all(values[i]==self.funcs[(tp, name)][i]
                                    for i in xrange(len(values))))

class MeshFunctions(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitCubeMesh(8, 8, 8)
        self.f = MeshFunction('int', self.mesh, 0)

    def testCreate(self):
        """Create MeshFunctions."""
        v = MeshFunction("uint", self.mesh)

        v = MeshFunction("uint", self.mesh, 0)
        self.assertEqual(v.size(), self.mesh.num_vertices())

        v = MeshFunction("uint", self.mesh, 1)
        self.assertEqual(v.size(), self.mesh.num_edges())

        v = MeshFunction("uint", self.mesh, 2)
        self.assertEqual(v.size(), self.mesh.num_facets())

        v = MeshFunction("uint", self.mesh, 3)
        self.assertEqual(v.size(), self.mesh.num_cells())

    def testCreateAssign(self):
        """Create MeshFunctions with value."""
        i = 10
        v = MeshFunction("uint", self.mesh, 0, i)
        self.assertEqual(v.size(), self.mesh.num_vertices())
        self.assertEqual(v[0], i)

        v = MeshFunction("uint", self.mesh, 1, i)
        self.assertEqual(v.size(), self.mesh.num_edges())
        self.assertEqual(v[0], i)

        v = MeshFunction("uint", self.mesh, 2, i)
        self.assertEqual(v.size(), self.mesh.num_facets())
        self.assertEqual(v[0], i)

        v = MeshFunction("uint", self.mesh, 3, i)
        self.assertEqual(v.size(), self.mesh.num_cells())
        self.assertEqual(v[0], i)

    def testAssign(self):
        f = self.f
        f[3] = 10
        v = Vertex(self.mesh, 3)
        self.assertEqual(f[v], 10)

if __name__ == "__main__":
    unittest.main()
