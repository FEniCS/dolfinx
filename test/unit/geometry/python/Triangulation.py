"""Unit tests for triangulation algorithms. Note that essential parts
of these algorithms are implemented as part of the mesh library but
tested here."""

# Copyright (C) 2013 Anders Logg
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
# First added:  2014-01-03
# Last changed: 2014-01-06

import unittest
from dolfin import UnitSquareMesh, Cell, Point, vertices

# For debugging and testing triangulation
plot=True
if plot: import pylab

def plot_triangulation_2d(triangulation):
    num_triangles = len(triangulation) / 6
    for i in range(len(triangulation) / 6):
        x0, y0, x1, y1, x2, y2 = triangulation[6*i:6*(i+1)]
        pylab.plot([x0, x1, x2, x0], [y0, y1, y2, y0], 'r')

def plot_cell_2d(cell):
    x = [v.point().x() for v in vertices(cell)]
    y = [v.point().y() for v in vertices(cell)]
    pylab.plot(x + [x[0]], y + [y[0]])

class Triangulation(unittest.TestCase):

    def test_triangulate_intersection_2d(self):

        # Create two meshes of the unit square
        mesh_0 = UnitSquareMesh(1, 1)
        mesh_1 = UnitSquareMesh(1, 1)

        # Translate second mesh
        dx = Point(-0.75, 0.75)
        mesh_1.translate(dx)

        # Extract cells
        c00 = Cell(mesh_0, 0)
        c01 = Cell(mesh_0, 1)
        c10 = Cell(mesh_1, 0)
        c11 = Cell(mesh_1, 1)

        # Compute triangulations
        #print c00.triangulate_intersection(c00)
        #print c00.triangulate_intersection(c01)
        T = c01.triangulate_intersection(c10)

        print len(T)
        print len(T) / 6
        print T

        plot_cell_2d(c01)
        plot_cell_2d(c10)
        plot_triangulation_2d(T)

        pylab.show()

if __name__ == "__main__":
    print ""
    print "Testing triangulation algorithms"
    print "------------------------------------------------"
    unittest.main()
