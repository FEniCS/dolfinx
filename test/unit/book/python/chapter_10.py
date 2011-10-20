"""
Unit tests for Chapter 10 (DOLFIN: A C++/Python finite element library).
Page numbering starts at 1 and is relative to the chapter (not the book).
"""

# Copyright (C) 2011 Anders Logg
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
# First added:  2011-10-20
# Last changed: 2011-10-20

import unittest
from dolfin import *

def create_data(A=None):
    "This function creates data used in the tests below"
    mesh = UnitSquare(2, 2)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    if A is None:
        A = assemble(u*v*dx)
    else:
        assemble(u*v*dx, tensor=A)
    b = assemble(v*dx)
    x = Vector()
    return A, x, b

class TestPage4(unittest.TestCase):

    def test_box_1(self):
        x = Vector()

    def test_box_2(self):
        A = Matrix()

class TestPage5(unittest.TestCase):

    def test_box_1(self):
        x = Vector(100)

    def test_box_2(self):
        A, x, b = create_data()
        solve(A, x, b)

    def test_box_3(self):
        A, x, b = create_data()
        solve(A, x, b, "lu");
        solve(A, x, b, "gmres", "ilu")

    def test_box_4(self):
        list_lu_solver_methods()
        list_krylov_solver_methods()
        list_krylov_solver_preconditioners()

class TestPage6(unittest.TestCase):

    def test_box_1(self):
        A, x, b = create_data()
        solver = LUSolver(A)
        solver.solve(x, b)

    def test_box_2(self):
        A, x, b = create_data()
        solver = LUSolver()
        solver.set_operator(A)
        solver.solve(x, b)

    def test_box_3(self):
        solver = LUSolver()
        solver.parameters["same_nonzero_pattern"] = True

    def test_box_4(self):
        A, x, b = create_data()
        solver = KrylovSolver(A)
        solver.solve(x, b)

class TestPage7(unittest.TestCase):

    def test_box_1(self):
        A, x, b = create_data()
        solver = KrylovSolver()
        solver.set_operator(A)
        solver.solve(x, b)

    def test_box_2(self):
        A, x, b = create_data()
        P = A
        solver = KrylovSolver()
        solver.set_operators(A, P)
        solver.solve(x, b)

    def test_box_3(self):
        solver = KrylovSolver()
        solver.parameters["relative_tolerance"] = 1.0e-6
        solver.parameters["absolute_tolerance"] = 1.0e-15
        solver.parameters["divergence_limit"] = 1.0e4
        solver.parameters["maximum_iterations"] = 10000
        solver.parameters["error_on_nonconvergence"] = True
        solver.parameters["nonzero_initial_guess"] = False

    def test_box_4(self):
        solver = KrylovSolver()
        solver.parameters["report"] = True
        solver.parameters["monitor_convergence"] = True

class TestPage8(unittest.TestCase):

    def test_box_1(self):
        solver = KrylovSolver("gmres", "ilu")

    def test_box_2(self):
        A = PETScMatrix()
        A, x, b = create_data(A)
        eigensolver = SLEPcEigenSolver(A)
        eigensolver.solve()
        lambda_r, lambda_c, x_real, x_complex = eigensolver.get_eigenpair(0)

class TestPage9(unittest.TestCase):

    def test_box_1(self):
        A = PETScMatrix()
        M = PETScMatrix()
        A, x, b = create_data(A)
        M, x, b = create_data(M)
        eigensolver = SLEPcEigenSolver(A, M)

    def test_box_2(self):
        parameters["linear_algebra_backend"] = "PETSc"

    def test_box_3(self):
        x = PETScVector()
        solver = PETScLUSolver()

class TestPage10(unittest.TestCase):

    def test_box_1(self):

        class MyNonlinearProblem(NonlinearProblem):
            def __init__(self, L, a, bc):
                NonlinearProblem.__init__(self)
                self.L = L
                self.a = a
                self.bc = bc

            def F(self, b, x):
                assemble(self.L, tensor=b)
                self.bc.apply(b, x)

            def J(self, A, x):
                assemble(self.a, tensor=A)
                self.bc.apply(A)

class TestPage11(unittest.TestCase):

    def test_box_1(self):

        class MyNonlinearProblem(NonlinearProblem):
            def __init__(self, L, a, bc):
                NonlinearProblem.__init__(self)
                self.L = L
                self.a = a
                self.bc = bc

            def F(self, b, x):
                assemble(self.L, tensor=b)
                self.bc.apply(b, x)

            def J(self, A, x):
                assemble(self.a, tensor=A)
                self.bc.apply(A)

        mesh = UnitSquare(2, 2)
        V  = FunctionSpace(mesh, "Lagrange", 1)
        u  = Function(V)
        du = TrialFunction(V)
        v  = TestFunction(V)
        a  = du*v*dx
        L  = u*v*dx - v*dx
        bc = DirichletBC(V, 0.0, DomainBoundary())

        problem = MyNonlinearProblem(L, a, bc)
        newton_solver = NewtonSolver()
        newton_solver.solve(problem, u.vector())

    def test_box_2(self):
        newton_solver = NewtonSolver()
        newton_solver.parameters["maximum_iterations"] = 20
        newton_solver.parameters["relative_tolerance"] = 1.0e-6
        newton_solver.parameters["absolute_tolerance"] = 1.0e-10
        newton_solver.parameters["error_on_nonconvergence"] = False

    def test_box_3(self):
        unit_square = UnitSquare(16, 16)
        unit_cube = UnitCube(16, 16, 16)

class TestPage12(unittest.TestCase):

    def test_box_1(self):
        mesh = Mesh();
        editor = MeshEditor();
        editor.open(mesh, 2, 2)
        editor.init_vertices(4)
        editor.init_cells(2)
        editor.add_vertex(0, 0.0, 0.0)
        editor.add_vertex(1, 1.0, 0.0)
        editor.add_vertex(2, 1.0, 1.0)
        editor.add_vertex(3, 0.0, 1.0)
        editor.add_cell(0, 0, 1, 2)
        editor.add_cell(1, 0, 2, 3)
        editor.close()

    def test_box_2(self):
        mesh = Mesh("mesh.xml")

class TestPage14(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(8, 8)
        entity = MeshEntity(mesh, 0, 33)
        vertex = Vertex(mesh, 33)
        cell = Cell(mesh, 25)


if __name__ == "__main__":
    print ""
    print "Testing the FEniCS Book, Chapter 10"
    print "-----------------------------------"
    unittest.main()
