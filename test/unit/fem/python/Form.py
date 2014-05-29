"""Unit tests for the fem interface"""

# Copyright (C) 2011-2014 Johan Hake
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
# Modified by Marie E. Rognes (meg@simula.no), 2012
# Modified by Martin S. Alnaes (martinal@simula.no), 2014

import unittest
import numpy
import ufl
from dolfin import *

class FormTestsOverManifolds(unittest.TestCase):

    def setUp(self):

        # 1D in 2D spaces
        self.square = UnitSquareMesh(2, 2)
        self.square_boundary = BoundaryMesh(self.square, "exterior")
        self.V1 = FunctionSpace(self.square_boundary, "CG", 1)
        self.VV1 = VectorFunctionSpace(self.square_boundary, "CG", 1)
        self.Q1 = FunctionSpace(self.square_boundary, "DG", 0)

        # 2D in 3D spaces
        self.cube = UnitCubeMesh(2, 2, 2)
        self.cube_boundary = BoundaryMesh(self.cube, "exterior")
        self.V2 = FunctionSpace(self.cube_boundary, "CG", 1)
        self.VV2 = VectorFunctionSpace(self.cube_boundary, "CG", 1)
        self.Q2 = FunctionSpace(self.cube_boundary, "DG", 0)

    def test_assemble_functional(self):

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

        u = Function(self.V1)
        w = TestFunction(self.Q1)
        u.vector()[:] = 0.5
        facetareas = MPI.sum(self.square_boundary.mpi_comm(),
                             assemble(u*w*dx).array().sum())
        self.assertAlmostEqual(facetareas, 2.0)

        u = Function(self.V2)
        w = TestFunction(self.Q2)
        u.vector()[:] = 0.5
        a = u*w*dx
        b = assemble(a)
        facetareas = MPI.sum(self.cube_boundary.mpi_comm(),
                             assemble(u*w*dx).array().sum())
        self.assertAlmostEqual(facetareas, 3.0)

        mesh = UnitSquareMesh(8, 8)
        bdry = BoundaryMesh(mesh, "exterior")
        V = FunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        BV = FunctionSpace(bdry, "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)

        a = MPI.sum(mesh.mpi_comm(),
                    assemble(inner(u, v)*ds).array().sum())
        b = MPI.sum(bdry.mpi_comm(),
                    assemble(inner(bu, bv)*dx).array().sum())
        self.assertAlmostEqual(a, b)

    def test_assemble_bilinear_1D_2D(self):

        # SubMesh not running in parallel
        if MPI.size(self.square.mpi_comm()) > 1:
            return

        V = FunctionSpace(self.square, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        bu = TrialFunction(self.V1)
        bv = TestFunction(self.V1)

        a = MPI.sum(self.square.mpi_comm(),
                    assemble(inner(u, v)*ds).array().sum())
        b = MPI.sum(self.square_boundary.mpi_comm(),
                    assemble(inner(bu, bv)*dx).array().sum())
        self.assertAlmostEqual(a, b)

        # Assemble over subset of mesh facets
        subdomain = CompiledSubDomain("near(x[1], 0.0)")
        bottom = FacetFunctionSizet(self.square)
        bottom.set_all(0)
        subdomain.mark(bottom, 1)
        dss = ds[bottom]
        foo = MPI.sum(self.square.mpi_comm(),
                      abs(assemble(inner(grad(u)[0], grad(v)[0])*dss(1)).array()).sum())
        # Assemble over all cells of submesh created from subset of boundary mesh
        bottom2 = CellFunctionSizet(self.square_boundary)
        bottom2.set_all(0)
        subdomain.mark(bottom2, 1)
        BV = FunctionSpace(SubMesh(self.square_boundary, bottom2, 1), "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)
        bar = MPI.sum(self.square_boundary.mpi_comm(),
                      abs(assemble(inner(grad(bu)[0], grad(bv)[0])*dx).array()).sum())
        # Should give same result
        self.assertAlmostEqual(bar, foo)

    def test_assemble_bilinear_2D_3D(self):

        # SubMesh not running in parallel
        if MPI.size(self.cube.mpi_comm()) > 1:
            return

        V = FunctionSpace(self.cube, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        # V2 is a FunctionSpace over self.cube_boundary
        bu = TrialFunction(self.V2)
        bv = TestFunction(self.V2)

        a = MPI.sum(self.cube.mpi_comm(),
                    assemble(inner(u, v)*ds).array().sum())
        b = MPI.sum(self.cube_boundary.mpi_comm(),
                    assemble(inner(bu, bv)*dx).array().sum())
        self.assertAlmostEqual(a, b)

        # Assemble over subset of mesh facets
        subdomain = CompiledSubDomain("near(x[1], 0.0)")
        bottom = FacetFunctionSizet(self.cube)
        bottom.set_all(0)
        subdomain.mark(bottom, 1)
        dss = ds[bottom]
        foo = MPI.sum(self.cube.mpi_comm(),
                      abs(assemble(inner(grad(u)[0], grad(v)[0])*dss(1)).array()).sum())
        #foo = MPI.sum(self.cube.mpi_comm(),
        #              abs(assemble(inner(grad(u)[0], grad(v)[0])*ds(1),
        #                           exterior_facet_domains=bottom).array()).sum())
        # Assemble over all cells of submesh created from subset of boundary mesh
        bottom2 = CellFunctionSizet(self.cube_boundary)
        bottom2.set_all(0)
        subdomain.mark(bottom2, 1)
        BV = FunctionSpace(SubMesh(self.cube_boundary, bottom2, 1), "CG", 1)
        bu = TrialFunction(BV)
        bv = TestFunction(BV)
        bar = MPI.sum(self.cube_boundary.mpi_comm(),
                      abs(assemble(inner(grad(bu)[0], grad(bv)[0])*dx).array()).sum())
        # Should give same result
        self.assertAlmostEqual(bar, foo)

class FormTestsOverFunnySpaces(unittest.TestCase):

    def setUp(self):

        # Set-up meshes
        n = 16
        plane = CompiledSubDomain("near(x[1], 1.0)")
        self.square = UnitSquareMesh(n, n)

        # SubMesh not running in parallel
        if MPI.size(self.square.mpi_comm()) > 1:
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

        # SubMesh not running in parallel
        if MPI.size(self.square.mpi_comm()) > 1:
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
                               places=5)
        self.assertAlmostEqual(a3, 5.0)
        self.assertAlmostEqual(a2, a3)
        self.assertAlmostEqual(pa2, a2)
        self.assertAlmostEqual(pa2, pa3)

    def test_mixed_poisson_solve(self):

        # SubMesh not running in parallel
        if MPI.size(self.square.mpi_comm()) > 1:
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
        self.cube = UnitCubeMesh(m, m, m)
        self.cube_boundary = BoundaryMesh(self.cube, "exterior")

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        plane = CompiledSubDomain("near(x[1], 0.0)")
        self.square = UnitSquareMesh(m, m)
        self.square_boundary = BoundaryMesh(self.square, "exterior")
        self.bottom1 = SubMesh(self.square_boundary, plane)

        self.bottom2 = SubMesh(self.cube_boundary, plane)
        line = CompiledSubDomain("near(x[0], 0.0)")
        self.mesh3 = BoundaryMesh(SubMesh(self.cube_boundary, plane), "exterior")
        self.bottom3 = SubMesh(self.mesh3, line)

    def test_normals_2D_1D(self):
        "Testing assembly of normals for 1D meshes embedded in 2D"

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        n = FacetNormal(self.bottom1)
        a = inner(n, n)*ds
        value_bottom1 = assemble(a)
        self.assertAlmostEqual(value_bottom1, 2.0)
        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c)
        self.assertAlmostEqual(b1, self.m-1)
        self.assertAlmostEqual(c1, - b1)

    def test_normals_3D_1D(self):
        "Testing assembly of normals for 1D meshes embedded in 3D"

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        n = FacetNormal(self.bottom3)
        a = inner(n, n)*ds
        v1 = assemble(a)
        self.assertAlmostEqual(v1, 2.0)
        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c)
        self.assertAlmostEqual(b1, self.m-1)
        self.assertAlmostEqual(c1, - b1)

    def test_normals_3D_2D(self):
        "Testing assembly of normals for 2D meshes embedded in 3D"

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        n = FacetNormal(self.bottom2)
        a = inner(n, n)*ds
        v1 = assemble(a)
        self.assertAlmostEqual(v1, 4.0)

        b = inner(n('+'), n('+'))*dS
        b1 = assemble(b)
        c = inner(n('+'), n('-'))*dS
        c1 = assemble(c)
        self.assertAlmostEqual(c1, - b1)

    def test_cell_volume(self):
        "Testing assembly of volume for embedded meshes"

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        volume = CellVolume(self.bottom1)
        a = volume*dx
        b = assemble(a)
        self.assertAlmostEqual(b, 1.0/self.m)

        volume = CellVolume(self.bottom3)
        a = volume*dx
        b = assemble(a)
        self.assertAlmostEqual(b, 1.0/self.m)

        volume = CellVolume(self.bottom2)
        a = volume*dx
        b = assemble(a)
        self.assertAlmostEqual(b, 1.0/(2*self.m*self.m))

    def test_circumradius(self):
        "Testing assembly of circumradius for embedded meshes"

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        r = Circumradius(self.bottom1)
        a = r*dx
        b = assemble(a)
        self.assertAlmostEqual(b, 0.5*(1.0/self.m))

        r = Circumradius(self.bottom3)
        a = r*dx
        b = assemble(a)
        self.assertAlmostEqual(b, 0.5*(1.0/self.m))

        square = UnitSquareMesh(self.m, self.m)
        r = Circumradius(square)
        a = r*dx
        b0 = assemble(a)

        r = Circumradius(self.bottom2)
        a = r*dx
        b1 = assemble(a)
        self.assertAlmostEqual(b0, b1)

    def test_facetarea(self):
        "Testing assembly of facet area for embedded meshes"

        # SubMesh not running in parallel
        if MPI.size(self.cube_boundary.mpi_comm()) > 1:
            return

        area = FacetArea(self.bottom1)
        a = area*ds
        b = assemble(a)
        self.assertAlmostEqual(b, 2.0)

        area = FacetArea(self.bottom3)
        a = area*ds
        b = assemble(a)
        self.assertAlmostEqual(b, 2.0)

        square = UnitSquareMesh(self.m, self.m)
        area = FacetArea(square)
        a = area*ds
        b0 = assemble(a)

        area = FacetArea(self.bottom2)
        a = area*ds
        b1 = assemble(a)
        self.assertAlmostEqual(b0, b1)

if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN Form operations"
    print "------------------------------------------------"
    unittest.main()
