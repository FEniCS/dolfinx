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
import numpy as np
from dolfin import *

class DofMapTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquareMesh(4, 4)
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)
        self.Q = VectorFunctionSpace(self.mesh, "Lagrange", 1)
        self.W = self.V*self.Q

    def test_tabulate_coord(self):

        L0  = self.W.sub(0)
        L1  = self.W.sub(1)
        L01 = L1.sub(0)
        L11 = L1.sub(1)

        coord0 = np.zeros((3,2), dtype="d")
        coord1 = np.zeros((3,2), dtype="d")
        coord2 = np.zeros((3,2), dtype="d")
        coord3 = np.zeros((3,2), dtype="d")

        for cell in cells(self.mesh):
            self.V.dofmap().tabulate_coordinates(cell, coord0)
            L0.dofmap().tabulate_coordinates(cell, coord1)
            L01.dofmap().tabulate_coordinates(cell, coord2)
            L11.dofmap().tabulate_coordinates(cell, coord3)
            coord4 = L1.dofmap().tabulate_coordinates(cell)

            self.assertTrue((coord0 == coord1).all())
            self.assertTrue((coord0 == coord2).all())
            self.assertTrue((coord0 == coord3).all())
            self.assertTrue((coord4[:3] == coord0).all())
            self.assertTrue((coord4[3:] == coord0).all())

    def test_tabulate_dofs(self):

        L0   = self.W.sub(0)
        L1   = self.W.sub(1)
        L01  = L1.sub(0)
        L11  = L1.sub(1)

        for i, cell in enumerate(cells(self.mesh)):

            dofs0 = L0.dofmap().cell_dofs(cell.index())

            dofs1 = L01.dofmap().cell_dofs(cell.index())
            dofs2 = L11.dofmap().cell_dofs(cell.index())
            dofs3 = L1.dofmap().cell_dofs(cell.index())

            self.assertTrue(np.array_equal(dofs0, \
                                L0.dofmap().cell_dofs(i)))
            self.assertTrue(np.array_equal(dofs1,
                                L01.dofmap().cell_dofs(i)))
            self.assertTrue(np.array_equal(dofs2,
                                L11.dofmap().cell_dofs(i)))
            self.assertTrue(np.array_equal(dofs3,
                                L1.dofmap().cell_dofs(i)))

            self.assertEqual(len(np.intersect1d(dofs0, dofs1)), 0)
            self.assertEqual(len(np.intersect1d(dofs0, dofs2)), 0)
            self.assertEqual(len(np.intersect1d(dofs1, dofs2)), 0)
            self.assertTrue(np.array_equal(np.append(dofs1, dofs2), dofs3))

    def test_tabulate_coord_periodic(self):

        class PeriodicBoundary2(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < DOLFIN_EPS
            def map(self, x, y):
                y[0] = x[0] - 1.0
                y[1] = x[1]

        # Create periodic boundary condition
        periodic_boundary = PeriodicBoundary2()

        mesh = UnitSquareMesh(4, 4)

        V = FunctionSpace(mesh, "Lagrange", 1,  constrained_domain=periodic_boundary)
        Q = VectorFunctionSpace(mesh, "Lagrange", 1,  constrained_domain=periodic_boundary)
        W = V*Q

        L0  = W.sub(0)
        L1  = W.sub(1)
        L01 = L1.sub(0)
        L11 = L1.sub(1)

        coord0 = np.zeros((3,2), dtype="d")
        coord1 = np.zeros((3,2), dtype="d")
        coord2 = np.zeros((3,2), dtype="d")
        coord3 = np.zeros((3,2), dtype="d")

        for cell in cells(mesh):
            V.dofmap().tabulate_coordinates(cell, coord0)
            L0.dofmap().tabulate_coordinates(cell, coord1)
            L01.dofmap().tabulate_coordinates(cell, coord2)
            L11.dofmap().tabulate_coordinates(cell, coord3)
            coord4 = L1.dofmap().tabulate_coordinates(cell)

            self.assertTrue((coord0 == coord1).all())
            self.assertTrue((coord0 == coord2).all())
            self.assertTrue((coord0 == coord3).all())
            self.assertTrue((coord4[:3] == coord0).all())
            self.assertTrue((coord4[3:] == coord0).all())

    def test_tabulate_dofs_periodic(self):

        class PeriodicBoundary2(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < DOLFIN_EPS
            def map(self, x, y):
                y[0] = x[0] - 1.0
                y[1] = x[1]

        mesh = UnitSquareMesh(5, 5)

        # Create periodic boundary
        periodic_boundary = PeriodicBoundary2()

        V = FunctionSpace(mesh, "Lagrange", 2, constrained_domain=periodic_boundary)
        Q = VectorFunctionSpace(mesh, "Lagrange", 2, constrained_domain=periodic_boundary)
        W = V*Q

        L0   = W.sub(0)
        L1   = W.sub(1)
        L01  = L1.sub(0)
        L11  = L1.sub(1)

        # Check dimensions
        self.assertEqual(V.dim(), 110)
        self.assertEqual(Q.dim(), 220)
        self.assertEqual(L0.dim(), V.dim())
        self.assertEqual(L1.dim(), Q.dim())
        self.assertEqual(L01.dim(), V.dim())
        self.assertEqual(L11.dim(), V.dim())

        for i, cell in enumerate(cells(mesh)):

            dofs0 = L0.dofmap().cell_dofs(cell.index())

            dofs1 = L01.dofmap().cell_dofs(cell.index())
            dofs2 = L11.dofmap().cell_dofs(cell.index())
            dofs3 = L1.dofmap().cell_dofs(cell.index())

            self.assertTrue(np.array_equal(dofs0, \
                                L0.dofmap().cell_dofs(i)))
            self.assertTrue(np.array_equal(dofs1,
                                L01.dofmap().cell_dofs(i)))
            self.assertTrue(np.array_equal(dofs2,
                                L11.dofmap().cell_dofs(i)))
            self.assertTrue(np.array_equal(dofs3,
                                L1.dofmap().cell_dofs(i)))

            self.assertEqual(len(np.intersect1d(dofs0, dofs1)), 0)
            self.assertEqual(len(np.intersect1d(dofs0, dofs2)), 0)
            self.assertEqual(len(np.intersect1d(dofs1, dofs2)), 0)
            self.assertTrue(np.array_equal(np.append(dofs1, dofs2), dofs3))

    def test_global_dof_builder(self):

        mesh = UnitSquareMesh(3, 3)

        V = VectorFunctionSpace(mesh, "CG", 1)
        Q = FunctionSpace(mesh, "CG", 1)
        R = FunctionSpace(mesh, "R", 0)

        W = MixedFunctionSpace([Q, Q, Q, R])
        W = MixedFunctionSpace([Q, Q, R, Q])
        W = MixedFunctionSpace([V, R])
        W = MixedFunctionSpace([R, V])

    def test_vertex_to_dof_map(self):

        # Check for both reordered and UFC ordered dofs
        for reorder_dofs in [True, False]:
            parameters.reorder_dofs_serial = reorder_dofs

            V = FunctionSpace(self.mesh, "Lagrange", 1)
            Q = VectorFunctionSpace(self.mesh, "Lagrange", 1)
            W = V*Q

            u = Function(V)
            e = Expression("x[0]+x[1]")
            u.interpolate(e)

            vert_values = self.mesh.coordinates().sum(1)
            func_values = -1*np.ones(len(vert_values))
            func_values[V.dofmap().vertex_to_dof_map(self.mesh)] = u.vector().array()

            for v_val, f_val in zip(vert_values, func_values):
                # Do not compare dofs owned by other process
                if f_val != -1:
                    self.assertAlmostEqual(f_val, v_val)

            c0 = Constant((1,2))
            u0 = Function(Q)
            u0.interpolate(c0)

            vert_values = np.zeros(self.mesh.num_vertices()*2)
            u1 = Function(Q)
            vert_values[::2] = 1
            vert_values[1::2] = 2

            u1.vector().set_local(vert_values[Q.dofmap().vertex_to_dof_map(self.mesh)].copy())
            self.assertAlmostEqual((u0.vector()-u1.vector()).sum(), 0.0)

            W = FunctionSpace(self.mesh, "DG", 0)
            self.assertRaises(RuntimeError, lambda : W.dofmap().vertex_to_dof_map(self.mesh))

            W = Q*FunctionSpace(self.mesh, "R", 0)
            self.assertRaises(RuntimeError, lambda : W.dofmap().vertex_to_dof_map(self.mesh))
            W = FunctionSpace(self.mesh, "CG", 2)
            self.assertRaises(RuntimeError, lambda : W.dofmap().vertex_to_dof_map(self.mesh))

if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN DofMap operations"
    print "------------------------------------------------"
    unittest.main()
