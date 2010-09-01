"""Unit tests for the fem interface"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2009-07-28"
__copyright__ = "Copyright (C) 2009 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

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
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

        # Assemble system
        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

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

        A_frobenius_norm =  4.3969686527582512;
        b_l2_norm = 0.95470326978246278;
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        if MPI.num_processes() > 1:
            print "FIXME: This unit test does not work in parallel, skipping"
            return

class DofMap(unittest.TestCase):

    def test_sub_dofmap(self):
        """ Test extraction of sub- and sub-sub-dofmaps."""
        mesh = UnitSquare(1, 1)
        V = FunctionSpace(mesh, "CG", 1)
        Q = VectorFunctionSpace(mesh, "CG", 1)
        W = V * Q

        for cell in cells(mesh):
            dofs0 = numpy.array((0,)*3, dtype="I")
            dofs1 = numpy.array((0,)*3, dtype="I")
            dofs2 = numpy.array((0,)*3, dtype="I")
            dofs3 = numpy.array((0,)*6, dtype="I")

            W.sub(0).dofmap().tabulate_dofs(dofs0, cell)

            L = W.sub(1)
            L.sub(0).dofmap().tabulate_dofs(dofs1, cell)
            L.sub(1).dofmap().tabulate_dofs(dofs2, cell)
            L.dofmap().tabulate_dofs(dofs3, cell)

            self.assertEqual(len(numpy.intersect1d(dofs0, dofs1)), 0)
            self.assertEqual(len(numpy.intersect1d(dofs0, dofs2)), 0)
            self.assertEqual(len(numpy.intersect1d(dofs1, dofs2)), 0)
            self.assertTrue(numpy.array_equal(numpy.append(dofs1, dofs2), dofs3))

        if MPI.num_processes() > 1:
            print "FIXME: This unit test does not work in parallel, skipping"
            return

if __name__ == "__main__":
    print ""
    print "Testing basic PyDOLFIN fem operations"
    print "------------------------------------------------"
    unittest.main()
