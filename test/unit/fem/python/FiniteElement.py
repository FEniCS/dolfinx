"""Unit tests for the fem interface"""

# Copyright (C) 2009 Garth N. Wells
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
# First added:  2009-07-28
# Last changed: 2009-07-28

import unittest
import numpy
from dolfin import *

class FiniteElementTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquareMesh(4, 4)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.Q = VectorFunctionSpace(self.mesh, "CG", 1)
        self.W = self.V * self.Q

    def test_evaluate_dofs(self):
        e = Expression("x[0]+x[1]")
        e2 = Expression(("x[0]+x[1]", "x[0]+x[1]"))

        coords = numpy.zeros((3, 2), dtype="d")
        coord = numpy.zeros(2, dtype="d")
        values0 = numpy.zeros(3, dtype="d")
        values1 = numpy.zeros(3, dtype="d")
        values2 = numpy.zeros(3, dtype="d")
        values3 = numpy.zeros(3, dtype="d")
        values4 = numpy.zeros(6, dtype="d")
        for cell in cells(self.mesh):
            self.V.dofmap().tabulate_coordinates(cell, coords)
            for i in xrange(coords.shape[0]):
                coord[:] = coords[i,:]
                values0[i] = e(*coord)
            self.W.sub(0).element().evaluate_dofs(values1, e, cell)
            L = self.W.sub(1)
            L.sub(0).element().evaluate_dofs(values2, e, cell)
            L.sub(1).element().evaluate_dofs(values3, e, cell)
            L.element().evaluate_dofs(values4, e2, cell)

            for i in range(3):
                self.assertAlmostEqual(values0[i], values1[i])
                self.assertAlmostEqual(values0[i], values2[i])
                self.assertAlmostEqual(values0[i], values3[i])
                self.assertAlmostEqual(values4[:3][i], values0[i])
                self.assertAlmostEqual(values4[3:][i], values0[i])

    def test_evaluate_dofs_manifolds_affine(self):
        "Testing evaluate_dofs vs tabulated coordinates."
        n = 4
        mesh = BoundaryMesh(UnitSquareMesh(n, n))
        mesh2 = BoundaryMesh(UnitCubeMesh(n, n, n))
        DG0 = FunctionSpace(mesh, "DG", 0)
        DG1 = FunctionSpace(mesh, "DG", 1)
        CG1 = FunctionSpace(mesh, "CG", 1)
        CG2 = FunctionSpace(mesh, "CG", 2)
        DG20 = FunctionSpace(mesh2, "DG", 0)
        DG21 = FunctionSpace(mesh2, "DG", 1)
        CG21 = FunctionSpace(mesh2, "CG", 1)
        CG22 = FunctionSpace(mesh2, "CG", 2)
        elements = [DG0, DG1, CG1, CG2, DG20, DG21, CG21, CG22]

        f = Expression("x[0]+x[1]")
        for V in elements:
            sdim = V.element().space_dimension()
            gdim = V.mesh().geometry().dim()
            coords = numpy.zeros((sdim, gdim), dtype="d")
            coord = numpy.zeros(gdim, dtype="d")
            values0 = numpy.zeros(sdim, dtype="d")
            values1 = numpy.zeros(sdim, dtype="d")
            for cell in cells(V.mesh()):
                V.dofmap().tabulate_coordinates(cell, coords)
                for i in xrange(coords.shape[0]):
                    coord[:] = coords[i,:]
                    values0[i] = f(*coord)
                V.element().evaluate_dofs(values1, f, cell)
                for i in range(sdim):
                    self.assertAlmostEqual(values0[i], values1[i])

class DofMapTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquareMesh(4, 4)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.Q = VectorFunctionSpace(self.mesh, "CG", 1)
        self.W = self.V*self.Q

    def test_tabulate_coord(self):

        coord0 = numpy.zeros((3,2), dtype="d")
        coord1 = numpy.zeros((3,2), dtype="d")
        coord2 = numpy.zeros((3,2), dtype="d")
        coord3 = numpy.zeros((3,2), dtype="d")
        coord4 = numpy.zeros((6,2), dtype="d")

        for cell in cells(self.mesh):
            self.V.dofmap().tabulate_coordinates(cell, coord0)
            self.W.sub(0).dofmap().tabulate_coordinates(cell, coord1)
            L = self.W.sub(1)
            L.sub(0).dofmap().tabulate_coordinates(cell, coord2)
            L.sub(1).dofmap().tabulate_coordinates(cell, coord3)
            L.dofmap().tabulate_coordinates(cell, coord4)

            self.assertTrue((coord0 == coord1).all())
            self.assertTrue((coord0 == coord2).all())
            self.assertTrue((coord0 == coord3).all())
            self.assertTrue((coord4[:3] == coord0).all())
            self.assertTrue((coord4[3:] == coord0).all())


if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN FiniteElement operations"
    print "------------------------------------------------"
    unittest.main()
