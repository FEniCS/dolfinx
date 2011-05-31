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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-07-28
# Last changed: 2009-07-28

import unittest
import numpy
from dolfin import *

class Assembly(unittest.TestCase):

    def test_facet_assembly(self):

        if MPI.num_processes() > 1:
            print "FIXME: This unit test does not work in parallel, skipping"
            return

        mesh = UnitSquare(24, 24)
        V = FunctionSpace(mesh, "DG", 1)

        # Define test and trial functions
        v = TestFunction(V)
        u = TrialFunction(V)

        # Define normal component, mesh size and right-hand side
        n = V.cell().n
        h = CellSize(mesh)
        h_avg = (h('+') + h('-'))/2
        f = Expression("500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)

        # Define bilinear form
        a = dot(grad(v), grad(u))*dx \
           - dot(avg(grad(v)), jump(u, n))*dS \
           - dot(jump(v, n), avg(grad(u)))*dS \
           + 4.0/h_avg*dot(jump(v, n), jump(u, n))*dS \
           - dot(grad(v), u*n)*ds \
           - dot(v*n, grad(u))*ds \
           + 8.0/h*v*u*ds

        # Define linear form
        L = v*f*dx

        # Reference values
        A_frobenius_norm = 157.867392938645
        b_l2_norm = 1.48087142738768

        # Assemble A and b separately
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

        # Assemble system
        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # Assemble A and b separately (multi-threaded)
        if MPI.num_processes() == 1:
            parameters["num_threads"] = 4
            self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
            self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

    def test_cell_assembly(self):

        mesh = UnitCube(4, 4, 4)
        V = VectorFunctionSpace(mesh, "DG", 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        f = Constant((10, 20, 30))

        def epsilon(v):
            return 0.5*(grad(v) + grad(v).T)

        a = inner(epsilon(v), epsilon(u))*dx
        L = inner(v, f)*dx

        A_frobenius_norm =  4.3969686527582512
        b_l2_norm = 0.95470326978246278

        # Assemble A and b separately
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

        # Assemble system
        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # Assemble A and b separately (multi-threaded)
        if MPI.num_processes() == 1:
            parameters["num_threads"] = 4
            self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
            self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

    def test_nonsquare_assembly(self):
        """Test assembly of a rectangular matrix"""

        mesh = UnitSquare(16, 16)

        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V*Q

        (v, q) = TestFunctions(W)
        (u, p) = TrialFunctions(W)

        a = div(v)*p*dx
        A_frobenius_norm = 9.6420303878382718e-01

        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)

        if MPI.num_processes() == 1:
            parameters["num_threads"] = 4
            self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)

    def test_subdomain_assembly(self):
        """Test assembly over subdomains"""

        # Define mesh
        mesh = UnitSquare(8, 8)

        # This is a hack to get around a DOLFIN bug
        if MPI.num_processes() > 1:
            cpp.MeshPartitioning.number_entities(mesh, mesh.topology().dim() - 1);

        # Define domain for lower left corner
        class MyDomain(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and x[1] < 0.5 + DOLFIN_EPS
        my_domain = MyDomain()

        # Mark mesh functions
        D = mesh.topology().dim()
        cell_domains = MeshFunction("uint", mesh, D)
        exterior_facet_domains = MeshFunction("uint", mesh, D - 1)
        cell_domains.set_all(1)
        exterior_facet_domains.set_all(1)
        my_domain.mark(cell_domains, 0)
        my_domain.mark(exterior_facet_domains, 0)

        # Define forms
        c = Constant(1.0)
        a0 = c*dx
        a1 = c*ds

        # Attach subdomains
        a0.cell_domains = cell_domains
        a1.exterior_facet_domains = exterior_facet_domains

        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(a0, mesh=mesh), 0.25)
        self.assertAlmostEqual(assemble(a1, mesh=mesh), 1.0)

"""
class FiniteElementTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquare(4, 4)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.Q = VectorFunctionSpace(self.mesh, "CG", 1)
        self.W = self.V * self.Q

    def test_evaluate_dofs(self):
        e = Expression("x[0]+x[1]+x[2]")
        e2 = Expression(("x[0]+x[1]+x[2]", "x[0]+x[1]+x[2]"))

        coords = numpy.zeros((3, 3), dtype="d")
        coord = numpy.zeros(3, dtype="d")
        values0 = numpy.zeros(3, dtype="d")
        values1 = numpy.zeros(3, dtype="d")
        values2 = numpy.zeros(3, dtype="d")
        values3 = numpy.zeros(3, dtype="d")
        values4 = numpy.zeros(6, dtype="d")
        for cell in cells(self.mesh):
            self.V.dofmap().tabulate_coordinates(coords, cell)
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
                # FIXME: Not working
                #self.assertAlmostEqual(values4[:3][i], values0[i])
                #self.assertAlmostEqual(values4[3:][i], values0[i])
"""

class DofMapTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquare(4, 4)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.Q = VectorFunctionSpace(self.mesh, "CG", 1)
        self.W = self.V*self.Q

    def test_tabulate_coord(self):

        coord0 = numpy.zeros((3,3), dtype="d")
        coord1 = numpy.zeros((3,3), dtype="d")
        coord2 = numpy.zeros((3,3), dtype="d")
        coord3 = numpy.zeros((3,3), dtype="d")
        coord4 = numpy.zeros((6,3), dtype="d")

        for cell in cells(self.mesh):
            self.V.dofmap().tabulate_coordinates(coord0, cell)
            self.W.sub(0).dofmap().tabulate_coordinates(coord1, cell)
            L = self.W.sub(1)
            L.sub(0).dofmap().tabulate_coordinates(coord2, cell)
            L.sub(1).dofmap().tabulate_coordinates(coord3, cell)
            L.dofmap().tabulate_coordinates(coord4, cell)

            self.assertTrue((coord0 == coord1).all())
            self.assertTrue((coord0 == coord2).all())
            self.assertTrue((coord0 == coord3).all())
            self.assertTrue((coord4[:3] == coord0).all())
            self.assertTrue((coord4[3:] == coord0).all())

    def test_tabulate_dofs(self):

        dofs0 = numpy.zeros(3, dtype="I")
        dofs1 = numpy.zeros(3, dtype="I")
        dofs2 = numpy.zeros(3, dtype="I")
        dofs3 = numpy.zeros(6, dtype="I")

        for i, cell in enumerate(cells(self.mesh)):
            
            self.W.sub(0).dofmap().tabulate_dofs(dofs0, cell)
            
            L = self.W.sub(1)
            L.sub(0).dofmap().tabulate_dofs(dofs1, cell)
            L.sub(1).dofmap().tabulate_dofs(dofs2, cell)
            L.dofmap().tabulate_dofs(dofs3, cell)
            
            self.assertTrue(numpy.array_equal(dofs0, \
                                self.W.sub(0).dofmap().cell_dofs(i)))
            self.assertTrue(numpy.array_equal(dofs1,
                                L.sub(0).dofmap().cell_dofs(i)))
            self.assertTrue(numpy.array_equal(dofs2,
                                L.sub(1).dofmap().cell_dofs(i)))
            self.assertTrue(numpy.array_equal(dofs3,
                                L.dofmap().cell_dofs(i)))
            
            self.assertEqual(len(numpy.intersect1d(dofs0, dofs1)), 0)
            self.assertEqual(len(numpy.intersect1d(dofs0, dofs2)), 0)
            self.assertEqual(len(numpy.intersect1d(dofs1, dofs2)), 0)
            self.assertTrue(numpy.array_equal(numpy.append(dofs1, dofs2), dofs3))

if __name__ == "__main__":
    print ""
    print "Testing basic PyDOLFIN fem operations"
    print "------------------------------------------------"
    unittest.main()
