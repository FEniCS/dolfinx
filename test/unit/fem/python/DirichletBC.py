"""Unit tests for Dirichlet boundary conditions"""

# Copyright (C) 2011-2012 Garth N. Wells
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
# Modified by Kent-Andre Mardal 2011
# Modified by Anders Logg 2011
# Modified by Martin Alnaes 2012
#
# First added:  2011-09-19
# Last changed: 2012-10-16

import unittest
import numpy
from dolfin import *

class DirichletBCTest(unittest.TestCase):

    def test_instantiation(self):
        """ A rudimentary test for instantiation"""
        # FIXME: Needs to be expanded
        mesh = UnitCubeMesh(8, 8, 8)
        V = FunctionSpace(mesh, "CG", 1)

        bc0 = DirichletBC(V, 1, "x[0]<0")
        bc1 = DirichletBC(bc0)
        self.assertTrue(bc0.function_space() == bc1.function_space())

    def test_director_lifetime(self):
        """Test for any problems with objects with directors going out
        of scope"""

        class Boundary(SubDomain):
            def inside(self, x, on_boundary): return on_boundary

        class BoundaryFunction(Expression):
            def eval(self, values, x): values[0] = 1.0

        mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        v, u = TestFunction(V), TrialFunction(V)
        A = assemble(v*u*dx)
        bc = DirichletBC(V, BoundaryFunction(), Boundary())

        bc.apply(A)

    def test_get_values(self):
        mesh = UnitSquareMesh(8, 8)
        dofs = numpy.zeros(3, dtype="I")

        def upper(x, on_boundary):
            return x[1] > 0.5 + DOLFIN_EPS

        V = FunctionSpace(mesh, "CG", 1)
        bc = DirichletBC(V, 0.0, upper)
        bc_values = bc.get_boundary_values()

    def test_meshdomain_bcs(self):
        """Test application of Dirichlet boundary conditions stored as
        part of the mesh. This test is also a compatibility test for
        VMTK."""

        mesh = Mesh("../../../../data/meshes/aneurysm.xml.gz")
        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)

        f = Constant(0)
        u1 = Constant(1)
        u2 = Constant(2)
        u3 = Constant(3)

        bc1 = DirichletBC(V, u1, 1)
        bc2 = DirichletBC(V, u2, 2)
        bc3 = DirichletBC(V, u3, 3)

        bcs = [bc1, bc2, bc3]

        L = f*v*dx

        b = assemble(L)
        [bc.apply(b) for bc in bcs]

        self.assertAlmostEqual(norm(b), 16.55294535724685)

    def test_bc_for_piola_on_manifolds(self):
        "Testing DirichletBC for piolas over standard domains vs manifolds."

        if MPI.num_processes() > 1:
            # SubMesh not working in parallel (the rest should)
            return

        n = 4
        side = compile_subdomains("near(x[2], 0.0)")

        mesh = SubMesh(BoundaryMesh(UnitCubeMesh(n, n, n), "exterior"), side)
        square = UnitSquareMesh(n, n)
        mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0")))

        RT1 = lambda mesh: FunctionSpace(mesh, "RT", 1)
        BDM1 = lambda mesh: FunctionSpace(mesh, "BDM", 1)
        BDM2 = lambda mesh: FunctionSpace(mesh, "BDM", 2)
        N1curl1 = lambda mesh: FunctionSpace(mesh, "N1curl", 1)
        N2curl1 = lambda mesh: FunctionSpace(mesh, "N2curl", 1)
        N1curl2 = lambda mesh:FunctionSpace(mesh, "N1curl", 2)
        N2curl2 = lambda mesh: FunctionSpace(mesh, "N2curl", 2)
        elements = [N1curl1, N2curl1,  N1curl2, N2curl2, RT1, BDM1, BDM2]

        for element in elements:
            V = element(mesh)
            bc = DirichletBC(V, (1.0, 0.0, 0.0), "on_boundary")
            u = Function(V)
            bc.apply(u.vector())
            b0 = assemble(inner(u, u)*dx)

            V = element(square)
            bc = DirichletBC(V, (1.0, 0.0), "on_boundary")
            u = Function(V)
            bc.apply(u.vector())
            b1 = assemble(inner(u, u)*dx)
            self.assertAlmostEqual(b0, b1)

if __name__ == "__main__":
    print ""
    print "Testing Dirichlet boundary conditions"
    print "------------------------------------------------"
    unittest.main()
