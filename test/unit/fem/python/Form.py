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

        self.cube = UnitCubeMesh(2, 2, 2)

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube.mpi_comm()) > 1:
            return

        # 1D in 2D spaces
        self.square = UnitSquareMesh(2, 2)
        self.square_bnd = BoundaryMesh(self.square, "exterior")
        self.V1 = FunctionSpace(self.square_bnd, "CG", 1)
        self.VV1 = VectorFunctionSpace(self.square_bnd, "CG", 1)
        self.Q1 = FunctionSpace(self.square_bnd, "DG", 0)

        # 2D in 3D spaces
        self.cube_bnd = BoundaryMesh(self.cube, "exterior")
        self.V2 = FunctionSpace(self.cube_bnd, "CG", 1)
        self.VV2 = VectorFunctionSpace(self.cube_bnd, "CG", 1)
        self.Q2 = FunctionSpace(self.cube_bnd, "DG", 0)

    def test_assemble_functional(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube.mpi_comm()) > 1:
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
        if MPI.num_processes(self.cube.mpi_comm()) > 1:
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
        bdry = BoundaryMesh(mesh, "exterior")
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
        if MPI.num_processes(self.cube.mpi_comm()) > 1:
            return

        V = FunctionSpace(self.square, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        bu = TrialFunction(self.V1)
        bv = TestFunction(self.V1)

        a = assemble(inner(u, v)*ds).array().sum()
        b = assemble(inner(bu, bv)*dx).array().sum()
        self.assertAlmostEqual(a, b)

        bottom = CompiledSubDomain("near(x[1], 0.0)")

        form = inner(grad(u)[0], grad(v)[0])*ds(0)
        vec = assemble(form, exterior_facet_domains=bottom)
        foo = abs(vec.array()).sum()

        BV = FunctionSpace(SubMesh(self.square_bnd, bottom), "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)

        form = inner(grad(bu), grad(bv))*dx
        vec = assemble(form)
        bar = abs(vec.array()).sum()

        self.assertAlmostEqual(bar, foo)

    def test_assemble_bilinear_2D_3D(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube.mpi_comm()) > 1:
            return

        V = FunctionSpace(self.cube, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        bu = TrialFunction(self.V2)
        bv = TestFunction(self.V2)

        a = assemble(inner(u, v)*ds).array().sum()
        b = assemble(inner(bu, bv)*dx).array().sum()
        self.assertAlmostEqual(a, b)

        bottom = CompiledSubDomain("near(x[1], 0.0)")

        vec = assemble(inner(grad(u)[0], grad(v)[0])*ds(0),
                           exterior_facet_domains=bottom)
        foo = abs(vec.array()).sum()

        BV = FunctionSpace(SubMesh(self.square_bnd, bottom), "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)

        vec = assemble(inner(grad(bu), grad(bv))*dx)
        bar = abs(vec.array()).sum()

        self.assertAlmostEqual(bar, foo)

class FormTestsOverFunnySpaces(unittest.TestCase):

    def setUp(self):

        # Set-up meshes
        n = 16
        plane = CompiledSubDomain("near(x[1], 1.0)")
        self.square = UnitSquareMesh(n, n)

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.square.mpi_comm()) > 1:
            return

        self.square3d = SubMesh(BoundaryMesh(UnitCubeMesh(n, n, n), "exterior"), plane)

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
        if MPI.num_processes(self.square.mpi_comm()) > 1:
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
        self.assertAlmostEqual((w2.vector() - pw2.vector()).norm("l2"), 0.0, \
                               places=6)
        self.assertAlmostEqual(a3, 5.0)
        self.assertAlmostEqual(a2, a3)
        self.assertAlmostEqual(pa2, a2)
        self.assertAlmostEqual(pa2, pa3)

    def test_mixed_poisson_solve(self):

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.square.mpi_comm()) > 1:
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

        m = 3
        self.m = m
        self.cube_bnd = BoundaryMesh(UnitCubeMesh(m, m, m), "exterior")

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        plane = CompiledSubDomain("near(x[1], 0.0)")
        self.square_bnd = BoundaryMesh(UnitSquareMesh(m, m), "exterior")
        self.square_bottom = SubMesh(self.square_bnd, plane)

        self.cube_bottom = SubMesh(self.cube_bnd, plane)

        line = CompiledSubDomain("near(x[0], 0.0)")
        self.cube_side_border = BoundaryMesh(SubMesh(self.cube_bnd, plane), "exterior")
        self.cube_edge = SubMesh(self.cube_side_border, line)

    def test_normals_2D_1D(self):
        "Testing assembly of normals for 1D meshes embedded in 2D"

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        cell = ufl.Cell("interval", geometric_dimension=2)
        n = ufl.FacetNormal(cell)
        a = inner(n, n)*ds
        value_bottom1 = assemble(a, mesh=self.square_bottom)
        self.assertAlmostEqual(value_bottom1, 2.0)
        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b, mesh=self.square_bottom)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c, mesh=self.square_bottom)
        self.assertAlmostEqual(b1, self.m-1)
        self.assertAlmostEqual(c1, - b1)

    def test_normals_3D_1D(self):
        "Testing assembly of normals for 1D meshes embedded in 3D"

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        cell = ufl.Cell("interval", geometric_dimension=3)
        n = ufl.FacetNormal(cell)
        a = inner(n, n)*ds
        v1 = assemble(a, mesh=self.cube_edge)
        self.assertAlmostEqual(v1, 2.0)
        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b, mesh=self.cube_edge)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c, mesh=self.cube_edge)
        self.assertAlmostEqual(b1, self.m-1)
        self.assertAlmostEqual(c1, - b1)

    def test_normals_3D_2D(self):
        "Testing assembly of normals for 2D meshes embedded in 3D"

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        cell = ufl.Cell("triangle", geometric_dimension=3)
        n = ufl.FacetNormal(cell)
        a = inner(n, n)*ds
        v1 = assemble(a, mesh=self.cube_bottom)
        self.assertAlmostEqual(v1, 4.0)

        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b, mesh=self.cube_bottom)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c, mesh=self.cube_bottom)
        self.assertAlmostEqual(c1, - b1)

    def test_cell_volume(self):
        "Testing assembly of volume for embedded meshes"

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        cell = ufl.Cell("interval", geometric_dimension=2)
        volume = ufl.CellVolume(cell)
        a = volume*dx
        b = assemble(a, mesh=self.square_bottom)
        self.assertAlmostEqual(b, 1.0/self.m)

        cell = ufl.Cell("interval", geometric_dimension=3)
        volume = ufl.CellVolume(cell)
        a = volume*dx
        b = assemble(a, mesh=self.cube_edge)
        self.assertAlmostEqual(b, 1.0/self.m)

        cell = ufl.Cell("triangle", geometric_dimension=3)
        volume = ufl.CellVolume(cell)
        a = volume*dx
        b = assemble(a, mesh=self.cube_bottom)
        self.assertAlmostEqual(b, 1.0/(2*self.m*self.m))

    def test_circumradius(self):
        "Testing assembly of circumradius for embedded meshes"

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        cell = ufl.Cell("interval", geometric_dimension=2)
        r = ufl.Circumradius(cell)
        a = r*dx
        b = assemble(a, mesh=self.square_bottom)
        self.assertAlmostEqual(b, 0.5*(1.0/self.m))

        cell = ufl.Cell("interval", geometric_dimension=3)
        r = ufl.Circumradius(cell)
        a = r*dx
        b = assemble(a, mesh=self.cube_edge)
        self.assertAlmostEqual(b, 0.5*(1.0/self.m))

        cell = ufl.Cell("triangle", geometric_dimension=2)
        r = ufl.Circumradius(cell)
        a = r*dx
        b0 = assemble(a, mesh=UnitSquareMesh(self.m, self.m))
        cell = ufl.Cell("triangle", geometric_dimension=3)
        r = ufl.Circumradius(cell)
        a = r*dx
        b1 = assemble(a, mesh=self.cube_bottom)
        self.assertAlmostEqual(b0, b1)

    def test_facetarea(self):
        "Testing assembly of facet area for embedded meshes"

        # Boundary mesh not running in parallel
        if MPI.num_processes(self.cube_bnd.mpi_comm()) > 1:
            return

        area = FacetArea(self.square_bottom)
        a = area*ds
        b = assemble(a)
        self.assertAlmostEqual(b, 2.0)

        cell = ufl.Cell("interval", geometric_dimension=3)

        domain_numbering = {}
        #print ufl.as_domain(cell).signature_data(domain_numbering=domain_numbering)
        #print self.cube_edge.ufl_domain().signature_data(domain_numbering=domain_numbering)

        #area = ufl.FacetArea(cell) # Does not work!
        area = ufl.FacetArea(self.cube_edge)
        a = area*ds(self.cube_edge)
        b = assemble(a)
        self.assertAlmostEqual(b, 2.0)

        cell = ufl.Cell("triangle", geometric_dimension=2)
        area = ufl.FacetArea(cell)
        a = area*ds
        b0 = assemble(a, mesh=UnitSquareMesh(self.m, self.m))

        cell = ufl.Cell("triangle", geometric_dimension=3)
        area = ufl.FacetArea(cell)
        a = area*ds
        b1 = assemble(a, mesh=self.cube_bottom)
        self.assertAlmostEqual(b0, b1)

if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN Form operations"
    print "------------------------------------------------"
    unittest.main()
