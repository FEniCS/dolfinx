"""Unit tests for the fem interface"""

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
# First added:  2011-12-02
# Last changed: 2012-12-11
#
# Modified by Marie E. Rognes (meg@simula.no), 2012

import unittest
import numpy
import ufl
from dolfin import *

class FormTest(unittest.TestCase):

    def setUp(self):
        self.mesh = UnitSquareMesh(10, 10)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.f = Expression("sin(pi*x[0]*x[1])")
        self.v = TestFunction(self.V)
        self.u = TrialFunction(self.V)

    def test_assemble(self):
        ufl_form = self.f*self.u*self.v*dx

        dolfin_form = Form(ufl_form)
        ufc_form = dolfin_form._compiled_form

        A_ufl_norm = assemble(ufl_form).norm("frobenius")
        A_dolfin_norm = assemble(dolfin_form).norm("frobenius")
        A_ufc_norm = assemble(ufc_form, coefficients=[self.f],
                              function_spaces=[self.V, self.V]).norm("frobenius")

        self.assertAlmostEqual(A_ufl_norm, A_dolfin_norm)
        self.assertAlmostEqual(A_ufl_norm, A_ufc_norm)

class FormTestsOverManifolds(unittest.TestCase):

    def setUp(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        # 1D in 2D spaces
        self.square = UnitSquareMesh(2, 2)
        self.mesh1 = BoundaryMesh(self.square)
        self.V1 = FunctionSpace(self.mesh1, "CG", 1)
        self.VV1 = VectorFunctionSpace(self.mesh1, "CG", 1)
        self.Q1 = FunctionSpace(self.mesh1, "DG", 0)

        # 2D in 3D spaces
        self.cube = UnitCubeMesh(2, 2, 2)
        self.mesh2 = BoundaryMesh(self.cube)
        self.V2 = FunctionSpace(self.mesh2, "CG", 1)
        self.VV2 = VectorFunctionSpace(self.mesh2, "CG", 1)
        self.Q2 = FunctionSpace(self.mesh2, "DG", 0)

    def test_assemble_functional(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        u = Function(self.V1)
        u.vector()[:] = 1.0
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 4.0)

        u = Function(self.V2)
        u.vector()[:] = 1.0
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 6.0)

        f = Expression("1.0")
        u = interpolate(f, self.V1)
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 4.0)

        f = Expression("1.0")
        u = interpolate(f, self.V2)
        surfacearea = assemble(u*dx)
        self.assertAlmostEqual(surfacearea, 6.0)

    def test_assemble_linear(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        u = Function(self.V1)
        w = TestFunction(self.Q1)
        u.vector()[:] = 0.5
        facetareas = assemble(u*w*dx).array().sum()
        self.assertAlmostEqual(facetareas, 2.0)

        u = Function(self.V2)
        w = TestFunction(self.Q2)
        u.vector()[:] = 0.5
        a = u*w*dx
        b = assemble(a)
        facetareas = assemble(u*w*dx).array().sum()
        self.assertAlmostEqual(facetareas, 3.0)

        mesh = UnitSquareMesh(8, 8)
        bdry = BoundaryMesh(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        BV = FunctionSpace(bdry, "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)

        a = assemble(inner(u, v)*ds).array().sum()
        b = assemble(inner(bu, bv)*dx).array().sum()
        self.assertAlmostEqual(a, b)

    def test_assemble_bilinear_1D_2D(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        V = FunctionSpace(self.square, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        bu = TrialFunction(self.V1)
        bv = TestFunction(self.V1)

        a = assemble(inner(u, v)*ds).array().sum()
        b = assemble(inner(bu, bv)*dx).array().sum()
        self.assertAlmostEqual(a, b)

        bottom = compile_subdomains("near(x[1], 0.0)")
        foo = abs(assemble(inner(grad(u)[0], grad(v)[0])*ds,
                           exterior_facet_domains=bottom).array()).sum()
        BV = FunctionSpace(SubMesh(self.mesh1, bottom), "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)
        bar = abs(assemble(inner(grad(bu), grad(bv))*dx).array()).sum()
        self.assertAlmostEqual(bar, foo)

    def test_assemble_bilinear_2D_3D(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        V = FunctionSpace(self.cube, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        bu = TrialFunction(self.V2)
        bv = TestFunction(self.V2)

        a = assemble(inner(u, v)*ds).array().sum()
        b = assemble(inner(bu, bv)*dx).array().sum()
        self.assertAlmostEqual(a, b)

        bottom = compile_subdomains("near(x[1], 0.0)")
        foo = abs(assemble(inner(grad(u)[0], grad(v)[0])*ds,
                           exterior_facet_domains=bottom).array()).sum()
        BV = FunctionSpace(SubMesh(self.mesh1, bottom), "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)
        bar = abs(assemble(inner(grad(bu), grad(bv))*dx).array()).sum()
        self.assertAlmostEqual(bar, foo)

class FormTestsOverFunnySpaces(unittest.TestCase):

    def setUp(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        # Set-up meshes
        n = 16
        plane = compile_subdomains("near(x[1], 1.0)")
        self.square = UnitSquareMesh(n, n)
        self.square3d = SubMesh(BoundaryMesh(UnitCubeMesh(n, n, n)), plane)

        # Define global normal and create orientation map
        global_normal = Expression(("0.0", "1.0", "0.0"))
        self.square3d.init_cell_orientations(global_normal)

        self.CG2 = VectorFunctionSpace(self.square, "CG", 1)
        self.CG3 = VectorFunctionSpace(self.square3d, "CG", 1)
        self.RT2 = FunctionSpace(self.square, "RT", 1)
        self.RT3 = FunctionSpace(self.square3d, "RT", 1)
        self.DG2 = FunctionSpace(self.square, "DG", 0)
        self.DG3 = FunctionSpace(self.square3d, "DG", 0)
        self.W2 = self.RT2*self.DG2
        self.W3 = self.RT3*self.DG3

    def test_basic_rt(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        f2 = Expression(("2.0", "1.0"))
        f3 = Expression(("1.0", "0.0", "2.0"))

        u2 = TrialFunction(self.RT2)
        u3 = TrialFunction(self.RT3)
        v2 = TestFunction(self.RT2)
        v3 = TestFunction(self.RT3)

        # Project
        pw2 = project(f2, self.RT2)
        pw3 = project(f3, self.RT3)
        pa2 = assemble(inner(pw2, pw2)*dx)
        pa3 = assemble(inner(pw3, pw3)*dx)

        # Project explicitly
        a2 = inner(u2, v2)*dx
        a3 = inner(u3, v3)*dx
        L2 = inner(f2, v2)*dx
        L3 = inner(f3, v3)*dx
        w2 = Function(self.RT2)
        w3 = Function(self.RT3)
        A2 = assemble(a2)
        b2 = assemble(L2)
        A3 = assemble(a3)
        b3 = assemble(L3)
        solve(A2, w2.vector(), b2)
        solve(A3, w3.vector(), b3)
        a2 = assemble(inner(w2, w2)*dx)
        a3 = assemble(inner(w3, w3)*dx)

        # Compare various results
        self.assertAlmostEqual((w2.vector() - pw2.vector()).norm("l2"), 0.0,
                               places=6)
        self.assertAlmostEqual(a3, 5.0)
        self.assertAlmostEqual(a2, a3)
        self.assertAlmostEqual(pa2, a2)
        self.assertAlmostEqual(pa2, pa3)

    def test_mixed_poisson_solve(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        f = Constant(1.0)

        # Solve mixed Poisson on standard unit square
        (sigma2, u2) = TrialFunctions(self.W2)
        (tau2, v2) = TestFunctions(self.W2)
        a = (inner(sigma2, tau2) + div(tau2)*u2 + div(sigma2)*v2)*dx
        L = f*v2*dx
        w2 = Function(self.W2)
        solve(a == L, w2)

        # Solve mixed Poisson on unit square in 3D
        (sigma3, u3) = TrialFunctions(self.W3)
        (tau3, v3) = TestFunctions(self.W3)
        a = (inner(sigma3, tau3) + div(tau3)*u3 + div(sigma3)*v3)*dx
        L = f*v3*dx
        w3 = Function(self.W3)
        solve(a == L, w3)

        # Check that results are about the same
        self.assertAlmostEqual(assemble(inner(w2, w2)*dx),
                               assemble(inner(w3, w3)*dx))

class TestGeometricQuantitiesOverManifolds(unittest.TestCase):

    def setUp(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        m = 3
        self.m = m
        plane = compile_subdomains("near(x[1], 0.0)")
        self.mesh1 = BoundaryMesh(UnitSquareMesh(m, m))
        self.bottom1 = SubMesh(self.mesh1, plane)

        self.mesh2 = BoundaryMesh(UnitCubeMesh(m, m, m))
        self.bottom2 = SubMesh(self.mesh2, plane)

        line = compile_subdomains("near(x[0], 0.0)")
        self.mesh3 = BoundaryMesh(SubMesh(self.mesh2, plane))
        self.bottom3 = SubMesh(self.mesh3, line)

    def test_normals_2D_1D(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        "Testing assembly of normals for 1D meshes embedded in 2D"
        n = ufl.Cell("interval", Space(2)).n
        a = inner(n, n)*ds
        value_bottom1 = assemble(a, mesh=self.bottom1)
        self.assertAlmostEqual(value_bottom1, 2.0)
        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b, mesh=self.bottom1)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c, mesh=self.bottom1)
        self.assertAlmostEqual(b1, self.m-1)
        self.assertAlmostEqual(c1, - b1)

    def test_normals_3D_1D(self):
        "Testing assembly of normals for 1D meshes embedded in 3D"

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        n = ufl.Cell("interval", Space(3)).n
        a = inner(n, n)*ds
        v1 = assemble(a, mesh=self.bottom3)
        self.assertAlmostEqual(v1, 2.0)
        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b, mesh=self.bottom3)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c, mesh=self.bottom3)
        self.assertAlmostEqual(b1, self.m-1)
        self.assertAlmostEqual(c1, - b1)

    def test_normals_3D_2D(self):
        "Testing assembly of normals for 2D meshes embedded in 3D"

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        n = ufl.Cell("triangle", Space(3)).n
        a = inner(n, n)*ds
        v1 = assemble(a, mesh=self.bottom2)
        self.assertAlmostEqual(v1, 4.0)

        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b, mesh=self.bottom2)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c, mesh=self.bottom2)
        self.assertAlmostEqual(c1, - b1)

    def test_cell_volume(self):
        "Testing assembly of volume for embedded meshes"

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        volume = ufl.Cell("interval", Space(2)).volume
        a = volume*dx
        b = assemble(a, mesh=self.bottom1)
        self.assertAlmostEqual(b, 1.0/self.m)

        volume = ufl.Cell("interval", Space(3)).volume
        a = volume*dx
        b = assemble(a, mesh=self.bottom3)
        self.assertAlmostEqual(b, 1.0/self.m)

        volume = ufl.Cell("triangle", Space(3)).volume
        a = volume*dx
        b = assemble(a, mesh=self.bottom2)
        self.assertAlmostEqual(b, 1.0/(2*self.m*self.m))

    def test_circumradius(self):
        "Testing assembly of circumradius for embedded meshes"

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        r = ufl.Cell("interval", Space(2)).circumradius
        a = r*dx
        b = assemble(a, mesh=self.bottom1)
        self.assertAlmostEqual(b, 1.0/self.m)

        r = ufl.Cell("interval", Space(3)).circumradius
        a = r*dx
        b = assemble(a, mesh=self.bottom3)
        self.assertAlmostEqual(b, 1.0/self.m)

        r = ufl.Cell("triangle", Space(2)).circumradius
        a = r*dx
        b0 = assemble(a, mesh=UnitSquareMesh(self.m, self.m))
        r = ufl.Cell("triangle", Space(3)).circumradius
        a = r*dx
        b1 = assemble(a, mesh=self.bottom2)
        self.assertAlmostEqual(b0, b1)

    def test_facetarea(self):
        "Testing assembly of facet area for embedded meshes"

        # Boundary mesh not running in parallel
        if MPI.num_processes() > 1:
            return

        area = ufl.Cell("interval", Space(2)).facet_area
        a = area*ds
        b = assemble(a, mesh=self.bottom1)
        self.assertAlmostEqual(b, 2.0)

        area = ufl.Cell("interval", Space(3)).facet_area
        a = area*ds
        b = assemble(a, mesh=self.bottom3)
        self.assertAlmostEqual(b, 2.0)

        area = ufl.Cell("triangle", Space(2)).facet_area
        a = area*ds
        b0 = assemble(a, mesh=UnitSquareMesh(self.m, self.m))

        area = ufl.Cell("triangle", Space(3)).facet_area
        a = area*ds
        b1 = assemble(a, mesh=self.bottom2)
        self.assertAlmostEqual(b0, b1)

if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN Form operations"
    print "------------------------------------------------"
    unittest.main()
