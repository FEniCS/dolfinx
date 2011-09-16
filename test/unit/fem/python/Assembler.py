"""Unit tests for assembly"""

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
# Modified by Marie E. Rognes 2011
# Modified by Anders Logg 2011
#
# First added:  2011-03-12
# Last changed: 2011-03-12

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

        dxs = dx[cell_domains]
        a0 = c*dxs(0)
        dss = ds[exterior_facet_domains]
        a1 = c*dss(0)

        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(a0, mesh=mesh), 0.25)
        self.assertAlmostEqual(assemble(a1, mesh=mesh), 1.0)

    def test_functional_assembly(self):

        mesh = UnitSquare(24, 24)

        # This is a hack to get around a DOLFIN bug
        if MPI.num_processes() > 1:
            cpp.MeshPartitioning.number_entities(mesh, mesh.topology().dim() - 1);

        f = Constant(1.0)
        M = f*dx
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 1.0)

        M = f*ds
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 4.0)

    def test_colored_cell_assembly(self):

        # Coloring and renumbering not supported in parallel
        if MPI.num_processes() != 1:
            return

        # Create mesh, then color and renumber
        old_mesh = UnitCube(4, 4, 4)
        old_mesh.color("vertex")
        mesh = old_mesh.renumber_by_color()

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

        # Assemble A and b separately (multi-threaded)
        parameters["num_threads"] = 4
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

    def test_subdomains_assembly(self):
        """
        Test assembly with subdomains specified in a form directly
        and of derived forms.
        """

        # Skip in parallel
        if MPI.num_processes() > 1:
            return

        # Define some haphazardly chosen cell/facet function
        mesh = UnitSquare(4, 4)
        domains = CellFunction("uint", mesh)
        domains.set_all(0)
        domains[0] = 1
        domains[1] = 1

        boundaries = FacetFunction("uint", mesh)
        boundaries.set_all(0)
        boundaries[0] = 1
        boundaries[1] = 1
        boundaries[2] = 1
        boundaries[3] = 1

        V = FunctionSpace(mesh, "CG", 2)
        f = Expression("x[0] + 2")
        g = Expression("x[1] + 1")

        f = interpolate(f, V)
        g = interpolate(g, V)

        dxs = dx[domains]
        dss = ds[boundaries]
        M = f*f*dxs(0) + g*f*dxs(1) + f*f*dss(1)

        # Check that domains are respected
        reference = 7.33040364583
        self.assertAlmostEqual(assemble(M), reference, 10)

        # Check that given exterior_facet_domains override
        new_boundaries = FacetFunction("uint", mesh)
        new_boundaries.set_all(0)
        reference2 = 6.2001953125
        value2 = assemble(M, exterior_facet_domains=new_boundaries)
        self.assertAlmostEqual(value2, reference2, 10)

        # Check that the form itself assembles as before
        self.assertAlmostEqual(assemble(M), reference, 10)

        # Take action of derivative of M on f
        df = TestFunction(V)
        L = derivative(M, f, df)
        dg = TrialFunction(V)
        F = derivative(L, g, dg)
        b = action(F, f)

        # Check that domain data carries across transformations:
        reference = 0.0626219513355
        self.assertAlmostEqual(assemble(b).norm("l2"), reference, 8)

    def test_meshdomains_assembly(self):
        "Test assembly with subdomains marked directly as part of the mesh"

        # Create a mesh of the unit cube
        mesh = UnitCube(4, 4, 4)

        # Define subdomains for 3 faces of the unit cube
        class F0(SubDomain):
            def inside(self, x, inside):
                return near(x[0], 0.0)
        class F1(SubDomain):
            def inside(self, x, inside):
                return near(x[1], 0.0)
        class F2(SubDomain):
            def inside(self, x, inside):
                return near(x[2], 0.0)

        # Define subdomain for left of x = 0.5
        class S0(SubDomain):
            def inside(self, x, inside):
                return x[0] < 0.5 + DOLFIN_EPS

        # Mark mesh
        f0 = F0()
        f1 = F1()
        f2 = F2()
        s0 = S0()
        f0.mark_facets(mesh, 0)
        f1.mark_facets(mesh, 1)
        f2.mark_facets(mesh, 2)
        s0.mark_cells(mesh, 0)

        # Assemble a form on marked subdomains
        M0 = Constant(1.0)*dx(0) + \
             Constant(2.0)*ds(0) + Constant(3.0)*ds(1) + Constant(4.0)*ds(2)
        m0 = assemble(M0, mesh=mesh)

        # Assemble a form on unmarked subdomains (marked automatically)
        M1 = Constant(1.0)*dx(1) + Constant(2.0)*ds(3)
        m1 = assemble(M1, mesh=mesh)

        # Check values
        self.assertAlmostEqual(m0, 9.5)
        self.assertAlmostEqual(m1, 6.5)

if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN assembly operations"
    print "------------------------------------------------"
    unittest.main()
