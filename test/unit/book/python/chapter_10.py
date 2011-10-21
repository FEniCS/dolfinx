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

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        gdim = mesh.topology().dim()
        tdim = mesh.geometry().dim()

class TestPage15(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        mesh.init(2)
        mesh.init(0, 0)
        mesh.init(1, 1)

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        for c in cells(mesh):
            for v0 in vertices(c):
                for v1 in vertices(v0):
                    print v1

class TestPage16(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        D = mesh.topology().dim()
        for c in entities(mesh, D):
            for v0 in entities(c, 0):
                for v1 in entities(v0, 0):
                    print v1

    def test_box_2(self):
        mesh = UnitSquare(2, 2)

        sub_domains = CellFunction("uint", mesh)
        sub_domains.set_all(0)
        for cell in cells(mesh):
            p = cell.midpoint()
            if p.x() > 0.5:
                sub_domains[cell] = 1

        boundary_markers = FacetFunction("uint", mesh)
        boundary_markers.set_all(0)
        for facet in facets(mesh):
            p = facet.midpoint()
            if near(p.y(), 0.0) or near(p.y(), 1.0):
                boundary_markers[facet] = 1

class TestPage17(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        sub_domains = mesh.data().create_mesh_function("sub_domains")
        sub_domains = mesh.data().mesh_function("sub_domains")

    def test_box_2(self):
        mesh = UnitSquare(8, 8)

        mesh = refine(mesh)

        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False)
        origin = Point(0.0, 0.0, 0.0)
        for cell in cells(mesh):
            p = cell.midpoint()
            if p.distance(origin) < 0.1:
                cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)

class TestPage19(unittest.TestCase):

    def test_box_1(self):
        element = FiniteElement("Lagrange", tetrahedron, 5)
        element = FiniteElement("CG", tetrahedron, 5)
        element = FiniteElement("Brezzi-Douglas-Marini", triangle, 3)
        element = FiniteElement("BDM", triangle, 3)
        element = FiniteElement("Nedelec 1st kind H(curl)", tetrahedron, 2)
        element = FiniteElement("N1curl", tetrahedron, 2)

class TestPage20(unittest.TestCase):

    def test_box_1(self):
        element = FiniteElement("Lagrange", triangle, 1)

    def test_box_2(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)

    def test_box_3(self):
        V = VectorElement("Lagrange", triangle, 2)
        Q = FiniteElement("Lagrange", triangle, 1)
        W = V*Q

class TestPage21(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = VectorFunctionSpace(mesh, "Lagrange", 2)
        Q = FunctionSpace(mesh, "Lagrange", 1)
        W = V*Q

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = Function(V)

class TestPage22(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        A = assemble(u*v*dx)
        b = assemble(v*dx)
        u = Function(V)
        solve(A, u.vector(), b)

    def test_box_2(self):
        mesh = UnitCube(2, 2, 2)

        V = FunctionSpace(mesh, "Lagrange", 1)
        u = Function(V)
        scalar = u(0.1, 0.2, 0.3)

        V = VectorFunctionSpace(mesh, "Lagrange", 1)
        u = Function(V)
        vector = u(0.1, 0.2, 0.3)

class TestPage23(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = VectorFunctionSpace(mesh, "Lagrange", 2)
        Q = FunctionSpace(mesh, "Lagrange", 1)
        W = V*Q

        w = Function(W)
        u, p = w.split()

        uu, pp = w.split(deepcopy=True)

class TestPage24(unittest.TestCase):

    def test_box_1(self):
        class MyExpression(Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])*cos(x[1])
        f = MyExpression()
        print f((0.5, 0.5))

    def test_box_2(self):
        class MyExpression(Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])
                values[1] = cos(x[1])
            def value_shape(self):
                return (2,)
        g = MyExpression()
        print g((0.5, 0.5))

    def test_box_3(self):
        f = Expression("sin(x[0])*cos(x[1])")
        g = Expression(("sin(x[0])", "cos(x[1])"))

class TestPage25(unittest.TestCase):

    def test_box_1(self):
        t = 0.0
        T = 1.0
        dt = 0.5

        h = Expression("t*sin(x[0])*cos(x[1])", t=0.0)
        while t < T:
            h.t = t
            t += dt

class TestPage26(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(8, 8)
        V = VectorFunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Expression(("0.0", "0.0"))

        E = 10.0
        nu = 0.3

        mu = E/(2.0*(1.0 + nu))
        lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

        def sigma(v):
            return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(v.cell().d)

        a = inner(sigma(u), sym(grad(v)))*dx
        L = dot(f, v)*dx

class TestPage27(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Function(V)
        M = f*dx
        L = f*v*dx
        a = u*v*dx

        m = assemble(M)
        b = assemble(L)
        A = assemble(a)

class TestPage29(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        class NeumannBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        neumann_boundary = NeumannBoundary()
        exterior_facet_domains = FacetFunction("uint", mesh)
        exterior_facet_domains.set_all(1)
        neumann_boundary.mark(exterior_facet_domains, 0)

class TestPage30(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        v = TestFunction(V)
        g = Function(V)
        h = Function(V)
        class NeumannBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + DOLFIN_EPS and on_boundary
        neumann_boundary = NeumannBoundary()

        dss = ds[neumann_boundary]
        a = g*v*dss(0) + h*v*dss(1)

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        class DirichletValue(Expression):
            def eval(self, value, x):
                values[0] = sin(x[0])
        class DirichletBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] > 0.5 - DOLFIN_EPS and on_boundary
        u_0 = DirichletValue()
        Gamma_D = DirichletBoundary()
        bc = DirichletBC(V, u_0, Gamma_D)

class TestPage31(unittest.TestCase):

     def test_box_1(self):
         mesh = UnitSquare(2, 2)
         V = FunctionSpace(mesh, "CG", 1)
         u_0 = Expression("sin(x[0])")
         bc = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")

     def test_box_2(self):
         mesh = UnitSquare(2, 2)
         V = FunctionSpace(mesh, "CG", 1)
         A, x, b = create_data()
         u_0 = Expression("sin(x[0])")
         bc = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         u = Function(V)

         bc.apply(A, b)
         bc.apply(u.vector())

class TestPage32(unittest.TestCase):

     def test_box_1(self):
         mesh = UnitSquare(2, 2)
         V = FunctionSpace(mesh, "CG", 1)
         u_0 = Expression("sin(x[0])")
         bc0 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bc1 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bc2 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         u = TrialFunction(V)
         v = TestFunction(V)
         a = u*v*dx
         L = v*dx

         bcs = [bc0, bc1, bc2]
         u = Function(V)
         solve(a == L, u, bcs=bcs)

     def test_box_2(self):
         mesh = UnitSquare(2, 2)
         V = FunctionSpace(mesh, "CG", 1)
         f = Expression("0.0")
         u_0 = Expression("sin(x[0])")
         bc0 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bc1 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bc2 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bcs = [bc0, bc1, bc2]
         du = TrialFunction(V)

         u = Function(V)
         v = TestFunction(V)
         F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx

         solve(F == 0, u, bcs=bcs)

         J = derivative(F, u)
         solve(F == 0, u, bcs=bcs, J=J)

     def test_box_3(self):
         mesh = UnitSquare(2, 2)
         V = FunctionSpace(mesh, "CG", 1)
         f = Expression("0.0")
         u_0 = Expression("sin(x[0])")
         bc0 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bc1 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bc2 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
         bcs = [bc0, bc1, bc2]
         du = TrialFunction(V)

         u = Function(V)
         v = TestFunction(V)
         F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx
         J = derivative(F, u)

         u = Function(V)
         problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
         solver = NonlinearVariationalSolver(problem)
         solver.parameters["linear_solver"] = "gmres"
         solver.parameters["preconditioner"] = "ilu"
         solver.solve()

class TestPage33(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        f = Expression("0.0")
        u_0 = Expression("sin(x[0])")
        bc0 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
        bc1 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
        bc2 = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
        bcs = [bc0, bc1, bc2]
        du = TrialFunction(V)
        u = Function(V)
        v = TestFunction(V)
        F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx
        J = derivative(F, u)
        u = Function(V)
        problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
        solver = NonlinearVariationalSolver(problem)

        info(solver.parameters, True)

    def test_box_2(self):
        mesh = Mesh("mesh.xml")

class TestPage34(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        mesh_function = MeshFunction("uint", mesh, 0)

        plot(u)
        plot(mesh)
        plot(mesh_function)

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        plot(grad(u))
        plot(u*u)
        element = FiniteElement("BDM", tetrahedron, 3)

        # Disabled since it claims the terminal
        #plot(element)

class TestPage35(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        file = File("solution.pvd")
        file << u

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        t = 1.0

        file = File("solution.pvd", "compressed");
        file << (u, t)

class TestPage36(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        matrix, vector, b = create_data()

        vector_file = File("vector.xml")
        vector_file << vector
        vector_file >> vector
        mesh_file = File("mesh.xml")
        mesh_file << mesh
        mesh_file >> mesh
        # FIXME: Not working
        #parameters_file = File("parameters.xml")
        #parameters_file << parameters
        #parameters_file >> parameters

class TestPage37(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        dt = 0.5
        t = 0.0
        T = 1.0

        time_series = TimeSeries("simulation_data")
        while t < T:
            time_series.store(u.vector(), t)
            time_series.store(mesh, t)
            t += dt

    def test_box_2(self):
        time_series = TimeSeries("simulation_data")
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        t = 0.5

        time_series.retrieve(u.vector(), t)
        time_series.retrieve(mesh, t)

    def test_box_3(self):
        M = 100
        N = 100
        info("Assembling system of size %d x %d." % (M, N))

class TestPage38(unittest.TestCase):

    def test_box_1(self):
        info("Test message")
        log(DEBUG, "Test message")
        log(15, "Test message")
        set_log_level(DEBUG)
        info("Test message")
        log(DEBUG, "Test message")
        log(15, "Test message")
        set_log_level(WARNING)
        info("Test message")
        warning("Test message")
        print "Test message"

class TestPage39(unittest.TestCase):

     def test_box_1(self):
         matrix, vector, b = create_data()
         solver = KrylovSolver()
         mesh = UnitSquare(2, 2)
         mesh_function = MeshFunction("uint", mesh, 0)
         function_space = FunctionSpace(mesh, "CG", 1)
         function = Function(function_space)

         info(vector)
         info(matrix)
         info(solver)
         info(mesh)
         info(mesh_function)
         info(function)
         info(function_space)
         info(parameters)

     def test_box_2(self):
         mesh = UnitSquare(2, 2)
         info(mesh, True)

class TestPage40(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        t = 0.0
        T = 1.0
        dt = 0.5

        begin("Starting nonlinear iteration.")
        info("Updating velocity.")
        info("Updating pressure.")
        info("Computing residual.")
        end()

        p = Progress("Iterating over all cells.", mesh.num_cells())
        for cell in cells(mesh):
            p += 1

        q = Progress("Time-stepping")
        while t < T:
            t += dt
            q.update(t / T)

    def test_box_2(self):
        def solve(A, b):
            timer = Timer("Linear solve")
            x = 1
            return x
        solve(None, None)

class TestPage41(unittest.TestCase):

    def test_box_1(self):
        info(parameters, True)
        num_threads = parameters["num_threads"]
        allow_extrapolation = parameters["allow_extrapolation"]
        parameters["num_threads"] = 8
        parameters["allow_extrapolation"] = True

    def test_box_2(self):
        solver = KrylovSolver()
        solver.parameters["absolute_tolerance"] = 1e-6
        solver.parameters["report"] = True
        solver.parameters["gmres"]["restart"] = 50
        solver.parameters["preconditioner"]["reuse"] = True

class TestPage42(unittest.TestCase):

    def test_box_1(self):
        my_parameters = Parameters("my_parameters")
        my_parameters.add("foo", 3)
        my_parameters.add("bar", 0.1)

    def test_box_2(self):
        d = dict(num_threads=4, krylov_solver=dict(absolute_tolerance=1e-6))
        parameters.update(d)

    def test_box_3(self):
        my_parameters = Parameters("my_parameters", foo=3, bar=0.1,
                                   nested=Parameters("nested", baz=True))

    def test_box_4(self):
        parameters.parse()

class TestPage43(unittest.TestCase):

    def test_box_1(self):
        pass
        # FIXME: Not working
        #file = File("parameters.xml")
        #file << parameters
        #file >> parameters

class TestPage45(unittest.TestCase):

    def test_box_1(self):
        parameters["mesh_partitioner"] = "ParMETIS"

    def test_box_2(self):
         from dolfin import cpp
         Function = cpp.Function
         assemble = cpp.assemble

class TestPage46(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)

class TestPage47(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

    def test_box_2(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "Lagrange", 1)
        class Source(Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])
        v = TestFunction(V)
        f = Source()
        L = f*v*dx
        assemble(L)

class TestPage48(unittest.TestCase):

     def test_box_1(self):
         e = Expression("sin(x[0])")

class TestPage49(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        c = Expression("sin(x[0])")
        a = c*dot(grad(u), grad(v))*dx
        A = assemble(a)

class TestPage50(unittest.TestCase):

    def test_box_1(self):
        parameters["form_compiler"]["name"] = "sfc"

    def test_box_2(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = u*v*dx

        A = assemble(a)
        AA = A.array()

    def test_box_3(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        v = TestFunction(V)
        L = v*dx

        parameters["linear_algebra_backend"] = "uBLAS"
        b = assemble(L)
        bb = b.data()

        # Reset linear algebra backend so that other tests work
        parameters["linear_algebra_backend"] = "PETSc"

    def test_box_4(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = u*v*dx

        # FIXME: Enable, don't have MTL4
        #parameters["linear_algebra_backend"] = "MTL4"
        #A = assemble(a)
        #rows, columns, values = A.data()

        # Reset linear algebra backend so that other tests work
        parameters["linear_algebra_backend"] = "PETSc"

class TestPage51(unittest.TestCase):

    def test_box_1(self):
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = u*v*dx

        # FIXME: Enable
        #parameters["linear_algebra_backend"] = "uBLAS"
        #A = assemble(a)

        #from scipy.sparse import csr_matrix
        #rows, columns, values = A.data()
        #csr = csr_matrix(values, rows, columns)

        # Reset linear algebra backend so that other tests work
        parameters["linear_algebra_backend"] = "PETSc"

    def test_box_2(self):
        b = Vector(10)
        c = Vector(10)
        b_copy = b[:]
        b[:] = c
        b[b < 0] = 0
        b2 = b[::2]

    def test_box_3(self):
        from numpy import array
        b = Vector(20)
        b1 = b[[0, 4, 7, 10]]
        b2 = b[array((0, 4, 7, 10))]

if __name__ == "__main__":
    print ""
    print "Testing the FEniCS Book, Chapter 10"
    print "-----------------------------------"
    unittest.main()
