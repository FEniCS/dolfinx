"""Unit tests for MeshValueCollection"""

# Copyright (C) 2011 Johan Hake
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

class MeshValueCollections(unittest.TestCase):

    def test_assign_2D_cells(self):
        mesh = UnitSquareMesh(3, 3)
        ncells = mesh.num_cells()
        f = MeshValueCollection("int", mesh, 2)
        all_new = True
        for cell in cells(mesh):
            value = ncells - cell.index()
            all_new = all_new and f.set_value(cell.index(), value)
        g = MeshValueCollection("int", mesh, 2)
        g.assign(f)
        self.assertEqual(ncells, f.size())
        self.assertEqual(ncells, g.size())
        self.assertTrue(all_new)

        for cell in cells(mesh):
            value = ncells - cell.index()
            self.assertEqual(value, g.get_value(cell.index(), 0))

        old_value = g.get_value(0,0)
        g.set_value(0, 0, old_value+1)
        self.assertEqual(old_value+1, g.get_value(0, 0))

    def test_assign_2D_facets(self):
        mesh = UnitSquareMesh(3, 3)
        mesh.init(2,1)
        ncells = mesh.num_cells()
        f = MeshValueCollection("int", mesh, 1)
        all_new = True
        for cell in cells(mesh):
            value = ncells - cell.index()
            for i, facet in enumerate(facets(cell)):
                all_new = all_new and f.set_value(cell.index(), i, value + i)

        g = MeshValueCollection("int", mesh, 1)
        g.assign(f)
        self.assertEqual(ncells*3, f.size())
        self.assertEqual(ncells*3, g.size())
        self.assertTrue(all_new)

        for cell in cells(mesh):
            value = ncells - cell.index()
            for i, facet in enumerate(facets(cell)):
                self.assertEqual(value+i, g.get_value(cell.index(), i))

    def test_assign_2D_vertices(self):
        mesh = UnitSquareMesh(3, 3)
        mesh.init(2,0)
        ncells = mesh.num_cells()
        f = MeshValueCollection("int", mesh, 0)
        all_new = True
        for cell in cells(mesh):
            value = ncells - cell.index()
            for i, vert in enumerate(vertices(cell)):
                all_new = all_new and f.set_value(cell.index(), i, value+i)

        g = MeshValueCollection("int", mesh, 0)
        g.assign(f)
        self.assertEqual(ncells*3, f.size())
        self.assertEqual(ncells*3, g.size())
        self.assertTrue(all_new)

        for cell in cells(mesh):
            value = ncells - cell.index()
            for i, vert in enumerate(vertices(cell)):
                self.assertEqual(value+i, g.get_value(cell.index(), i))

    def test_mesh_function_assign_2D_cells(self):
        mesh = UnitSquareMesh(3, 3)
        ncells = mesh.num_cells()
        f = CellFunction("int", mesh)
        for cell in cells(mesh):
            f[cell] = ncells - cell.index()

        g = MeshValueCollection("int", mesh, 2)
        g.assign(f)
        self.assertEqual(ncells, f.size())
        self.assertEqual(ncells, g.size())

        f2 = MeshFunction("int", mesh, g)

        for cell in cells(mesh):
            value = ncells - cell.index()
            self.assertEqual(value, g.get_value(cell.index(), 0))
            self.assertEqual(f2[cell], g.get_value(cell.index(), 0))

        h = MeshValueCollection("int", mesh, 2)
        global_indices = mesh.topology().global_indices(2)
        ncells_global = mesh.size_global(2)
        for cell in cells(mesh):
            if global_indices[cell.index()] in [5,8,10]:
                continue
            value = ncells_global - global_indices[cell.index()]
            h.set_value(cell.index(), int(value))

        f3 = MeshFunction("int", mesh, h)

        values = f3.array()
        values[values>ncells_global] = 0.

        info(str(values))
        info(str(values.sum()))

        self.assertEqual(MPI.sum(values.sum()*1.0), 140.)

    def test_mesh_function_assign_2D_facets(self):
        mesh = UnitSquareMesh(3, 3)
        mesh.init(1)
        f = FacetFunction("int", mesh, 25)
        for cell in cells(mesh):
            for i, facet in enumerate(facets(cell)):
                self.assertEqual(25, f[facet])

        g = MeshValueCollection("int", mesh, 1)
        g.assign(f)
        self.assertEqual(mesh.num_facets(), f.size())
        self.assertEqual(mesh.num_cells()*3, g.size())
        for cell in cells(mesh):
            for i, facet in enumerate(facets(cell)):
                self.assertEqual(25, g.get_value(cell.index(), i))

        f2 = MeshFunction("int", mesh, g)

        for cell in cells(mesh):
            for i, facet in enumerate(facets(cell)):
                self.assertEqual(f2[facet], g.get_value(cell.index(), i))

    def test_mesh_function_assign_2D_vertices(self):
        mesh = UnitSquareMesh(3, 3)
        mesh.init(0)
        f = VertexFunction("int", mesh, 25)
        g = MeshValueCollection("int", mesh, 0)
        g.assign(f)
        self.assertEqual(mesh.num_vertices(), f.size())
        self.assertEqual(mesh.num_cells()*3, g.size())

        f2 = MeshFunction("int", mesh, g)

        for cell in cells(mesh):
            for i, vert in enumerate(vertices(cell)):
                self.assertEqual(25, g.get_value(cell.index(), i))
                self.assertEqual(f2[vert], g.get_value(cell.index(), i))

if __name__ == "__main__":
    unittest.main()
