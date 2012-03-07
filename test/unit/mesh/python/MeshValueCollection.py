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

  def testAssign2DCells(self):
      mesh = UnitSquare (3, 3)
      ncells = mesh.num_cells()
      f = MeshValueCollection("int", 2)
      all_new = True
      for cell in cells(mesh):
          value = ncells - cell.index()
          all_new = all_new and f.set_value(cell.index(), value, mesh)
      g = MeshValueCollection("int", 2)
      g.assign(f)
      self.assertEqual(ncells, f.size())
      self.assertEqual(ncells, g.size())
      self.assertTrue(all_new)
      
      for cell in cells(mesh):
          value = ncells - cell.index()
          self.assertEqual(value, g.get_value(cell.index(), 0))
          
  def testAssign2DFacets(self):
      mesh = UnitSquare (3, 3)
      mesh.init(2,1)
      ncells = mesh.num_cells()
      f = MeshValueCollection("int", 1)
      all_new = True
      for cell in cells(mesh):
          value = ncells - cell.index()
          for i, facet in enumerate(facets(cell)):
              all_new = all_new and f.set_value(cell.index(), i, value+i)

      g = MeshValueCollection("int", 1)
      g.assign(f)
      self.assertEqual(ncells*3, f.size())
      self.assertEqual(ncells*3, g.size())
      self.assertTrue(all_new)
      
      for cell in cells(mesh):
          value = ncells - cell.index()
          for i, facet in enumerate(facets(cell)):
              self.assertEqual(value+i, g.get_value(cell.index(), i))
          
  def testAssign2DVertices(self):
      mesh = UnitSquare (3, 3)
      mesh.init(2,0)
      ncells = mesh.num_cells()
      f = MeshValueCollection("int", 0)
      all_new = True
      for cell in cells(mesh):
          value = ncells - cell.index()
          for i, vert in enumerate(vertices(cell)):
              all_new = all_new and f.set_value(cell.index(), i, value+i)

      g = MeshValueCollection("int", 0)
      g.assign(f)
      self.assertEqual(ncells*3, f.size())
      self.assertEqual(ncells*3, g.size())
      self.assertTrue(all_new)
      
      for cell in cells(mesh):
          value = ncells - cell.index()
          for i, vert in enumerate(vertices(cell)):
              self.assertEqual(value+i, g.get_value(cell.index(), i))
          
  def testMeshFunctionAssign2DCells(self):
      mesh = UnitSquare (3, 3)
      ncells = mesh.num_cells()
      f = CellFunction("int", mesh)
      for cell in cells(mesh):
          f[cell] = ncells - cell.index()

      g = MeshValueCollection("int", 2)
      g.assign(f)
      self.assertEqual(ncells, f.size())
      self.assertEqual(ncells, g.size())
      
      f2 = MeshFunction("int", mesh, g)
      
      for cell in cells(mesh):
          value = ncells - cell.index()
          self.assertEqual(value, g.get_value(cell.index(), 0))
          self.assertEqual(f2[cell], g.get_value(cell.index(), 0))
          
  def testMeshFunctionAssign2DFacets(self):
      mesh = UnitSquare (3, 3)
      mesh.init(1)
      f = FacetFunction("int", mesh, 25)
      g = MeshValueCollection("int", 1)
      g.assign(f)
      self.assertEqual(mesh.num_facets(), f.size())
      self.assertEqual(mesh.num_cells()*3, g.size())
      
      f2 = MeshFunction("int", mesh, g)
      
      for cell in cells(mesh):
          for i, facet in enumerate(facets(cell)):
              self.assertEqual(25, g.get_value(cell.index(), i))
              self.assertEqual(f2[facet], g.get_value(cell.index(), i))
          
  def testMeshFunctionAssign2DVertices(self):
      mesh = UnitSquare (3, 3)
      mesh.init(0)
      f = VertexFunction("int", mesh, 25)
      g = MeshValueCollection("int", 0)
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
