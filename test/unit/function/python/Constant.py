"""Unit tests for the function library"""

# Copyright (C) 2007 Anders Logg
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
# First added:  2007-05-24
# Last changed: 2011-01-28

import unittest
from numpy import array
from dolfin import *


mesh = UnitCube(8, 8, 8)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

class Constants(unittest.TestCase):

     def testConstantInit(self):
          c0 = Constant(1.)
          c1 = Constant([2,3], interval)
          c2 = Constant([[2,3], [3,4]], triangle)
          c3 = Constant(array([2,3]), tetrahedron)

          self.assertTrue(c0.cell().is_undefined())
          self.assertTrue(c1.cell() == interval)
          self.assertTrue(c2.cell() == triangle)
          self.assertTrue(c3.cell() == tetrahedron)

          self.assertTrue(c0.shape() == ())
          self.assertTrue(c1.shape() == (2,))
          self.assertTrue(c2.shape() == (2,2))
          self.assertTrue(c3.shape() == (2,))

     def testGrad(self):
          import ufl
          zero = ufl.constantvalue.Zero((2,3))
          c0 = Constant(1.)
          c3 = Constant(array([2,3]), tetrahedron)
          def gradient(c):
               return grad(c)
          self.assertRaises(UFLException, gradient, c0)
          self.assertEqual(zero, gradient(c3))

     def test_compute_vertex_values(self):
          from numpy import zeros, all, array
          
          e0 = Constant(1)
          e1 = Constant((1, 2, 3))
          
          e0_values = zeros(mesh.num_vertices(),dtype='d')
          e1_values = zeros(mesh.num_vertices()*3,dtype='d')
          
          e0.compute_vertex_values(e0_values, mesh)
          e1.compute_vertex_values(e1_values, mesh)
        
          self.assertTrue(all(e0_values==1))
          self.assertTrue(all(e1_values[:mesh.num_vertices()]==1))
          self.assertTrue(all(e1_values[mesh.num_vertices():mesh.num_vertices()*2]==2))
          self.assertTrue(all(e1_values[mesh.num_vertices()*2:mesh.num_vertices()*3]==3))

if __name__ == "__main__":
    unittest.main()
