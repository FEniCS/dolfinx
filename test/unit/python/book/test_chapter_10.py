#!/usr/bin/env py.test

"""
Unit tests for Chapter 10 (DOLFIN: A C++/Python finite element library).
Page numbering starts at 1 and is relative to the chapter (not the book).
"""

# Copyright (C) 2011-2014 Anders Logg
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

from __future__ import print_function
import pytest
from dolfin import *
import os

from dolfin_utils.test import * #cd_tempdir, pushpop_parameters, skip_in_parallel, use_gc_barrier

def create_data(A=None):
    "This function creates data used in the tests below"
    mesh = UnitSquareMesh(4, 4)
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


@use_gc_barrier
def test_p4_box_1():
    x = Vector()


@use_gc_barrier
def test_p4_box_2():
    A = Matrix()


@use_gc_barrier
def test_p5_box_1():
    x = Vector(mpi_comm_world(), 100)


@use_gc_barrier
def test_p5_box_2():
    A, x, b = create_data()
    solve(A, x, b)


@skip_in_parallel
def test_p5_box_3():
    A, x, b = create_data()
    solve(A, x, b, "lu")
    solve(A, x, b, "gmres", "ilu")


@skip_in_parallel
def test_p5_box_4():
    list_lu_solver_methods()
    list_krylov_solver_methods()
    list_krylov_solver_preconditioners()


@use_gc_barrier
def test_p6_box_1():
    A, x, b = create_data()
    solver = LUSolver(A)
    solver.solve(x, b)


@use_gc_barrier
def test_p6_box_2():
    A, x, b = create_data()
    solver = LUSolver()
    solver.set_operator(A)
    solver.solve(x, b)


@skip_in_parallel
def test_p6_box_3():
    solver = LUSolver()
    solver.parameters["same_nonzero_pattern"] = True


@use_gc_barrier
def test_p6_box_4():
    A, x, b = create_data()
    solver = KrylovSolver(A)
    solver.solve(x, b)


@use_gc_barrier
def test_p7_box_1():
    A, x, b = create_data()
    solver = KrylovSolver()
    solver.set_operator(A)
    solver.solve(x, b)


@use_gc_barrier
def test_p7_box_2():
    A, x, b = create_data()
    P = A
    solver = KrylovSolver()
    solver.set_operators(A, P)
    solver.solve(x, b)


@use_gc_barrier
def test_p7_box_3():
    solver = KrylovSolver()
    solver.parameters["relative_tolerance"] = 1.0e-6
    solver.parameters["absolute_tolerance"] = 1.0e-15
    solver.parameters["divergence_limit"] = 1.0e4
    solver.parameters["maximum_iterations"] = 10000
    solver.parameters["error_on_nonconvergence"] = True
    solver.parameters["nonzero_initial_guess"] = False


@use_gc_barrier
def test_p7_box_4():
    solver = KrylovSolver()
    solver.parameters["report"] = True
    solver.parameters["monitor_convergence"] = True


@use_gc_barrier
def test_p8_box_1():
    solver = KrylovSolver("gmres", "ilu")


@skip_if_not_PETsc_or_not_slepc
@use_gc_barrier
def test_p8_box_2():
    A = PETScMatrix()
    A, x, b = create_data(A)
    eigensolver = SLEPcEigenSolver(A)
    eigensolver.solve()
    lambda_r, lambda_c, x_real, x_complex = eigensolver.get_eigenpair(0)


@skip_if_not_PETsc_or_not_slepc
@use_gc_barrier
def test_p9_box_1():
    A = PETScMatrix()
    M = PETScMatrix()
    A, x, b = create_data(A)
    M, x, b = create_data(M)
    eigensolver = SLEPcEigenSolver(A, M)


@skip_if_not_PETSc
@use_gc_barrier
def test_p9_box_2(pushpop_parameters):
    parameters["linear_algebra_backend"] = "PETSc"


@skip_if_not_PETSc
@use_gc_barrier
def test_p9_box_3():
    x = PETScVector()
    solver = PETScLUSolver()


@use_gc_barrier
def test_p10_box_1():

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


@use_gc_barrier
def test_p11_box_1():

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

    mesh = UnitSquareMesh(3, 3)
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


@use_gc_barrier
def test_p11_box_2():
    newton_solver = NewtonSolver()
    newton_solver.parameters["maximum_iterations"] = 20
    newton_solver.parameters["relative_tolerance"] = 1.0e-6
    newton_solver.parameters["absolute_tolerance"] = 1.0e-10
    newton_solver.parameters["error_on_nonconvergence"] = False


@use_gc_barrier
def test_p11_box_3():
    unit_square = UnitSquareMesh(16, 16)
    unit_cube = UnitCubeMesh(16, 16, 16)


@skip_in_parallel
def test_p12_box_1():
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

@use_gc_barrier
def test_p13_box_2():
    mesh = Mesh(os.path.join(os.path.dirname(__file__), "mesh.xml"))

@skip_in_parallel
def test_p14_box_1():
    mesh = UnitSquareMesh(8, 8)
    entity = MeshEntity(mesh, 0, 33)
    vertex = Vertex(mesh, 33)
    cell = Cell(mesh, 25)


@use_gc_barrier
def test_p14_box_2():
    mesh = UnitSquareMesh(3, 3)
    gdim = mesh.topology().dim()
    tdim = mesh.geometry().dim()


@skip_in_parallel
def test_p15_box_1():
    mesh = UnitSquareMesh(3, 3)
    mesh.init(2)
    mesh.init(0, 0)
    mesh.init(1, 1)


@use_gc_barrier
def test_p15_box_2():
    mesh = UnitSquareMesh(3, 3)
    for c in cells(mesh):
        for v0 in vertices(c):
            for v1 in vertices(v0):
                print(v1)


@use_gc_barrier
def test_p16_box_1():
    mesh = UnitSquareMesh(3, 3)
    D = mesh.topology().dim()
    for c in entities(mesh, D):
        for v0 in entities(c, 0):
            for v1 in entities(v0, 0):
                print(v1)


@use_gc_barrier
def test_p16_box_2():
    mesh = UnitSquareMesh(3, 3)

    sub_domains = CellFunction("size_t", mesh)
    sub_domains.set_all(0)
    for cell in cells(mesh):
        p = cell.midpoint()
        if p.x() > 0.5:
            sub_domains[cell] = 1

    boundary_markers = FacetFunction("size_t", mesh)
    boundary_markers.set_all(0)
    for facet in facets(mesh):
        p = facet.midpoint()
        if near(p.y(), 0.0) or near(p.y(), 1.0):
            boundary_markers[facet] = 1


@use_gc_barrier
def test_p17_box_1():
    mesh = UnitSquareMesh(3, 3)
    # Note: MeshData no longer returns MeshFunctions. This was
    #       necessary to remove a circular code dependency.
    #       Accessing a MeshFunction will now throw an error with
    #       the suggestion to use arrays.
    #sub_domains = mesh.data().create_mesh_function("sub_domains")
    #sub_domains = mesh.data().mesh_function("sub_domains")
    sub_domains = mesh.data().create_array("sub_domains", 2)
    sub_domains = mesh.data().array("sub_domains", 2)


@use_gc_barrier
def test_p17_box_2():
    mesh = UnitSquareMesh(8, 8)

    mesh = refine(mesh)

    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(0.0, 0.0, 0.0)
    for cell in cells(mesh):
        p = cell.midpoint()
        if p.distance(origin) < 0.1:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)


@use_gc_barrier
def test_p19_box_1():
    element = FiniteElement("Lagrange", tetrahedron, 5)
    element = FiniteElement("CG", tetrahedron, 5)
    element = FiniteElement("Brezzi-Douglas-Marini", triangle, 3)
    element = FiniteElement("BDM", triangle, 3)
    element = FiniteElement("Nedelec 1st kind H(curl)", tetrahedron, 2)
    element = FiniteElement("N1curl", tetrahedron, 2)


@use_gc_barrier
def test_p20_box_1():
    element = FiniteElement("Lagrange", triangle, 1)


@use_gc_barrier
def test_p20_box_2():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)


@use_gc_barrier
def test_p20_box_3():
    V = VectorElement("Lagrange", triangle, 2)
    Q = FiniteElement("Lagrange", triangle, 1)
    W = V*Q


@use_gc_barrier
def test_p21_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    #W = V*Q # deprecated!


@use_gc_barrier
def test_p21_box_2():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)


@use_gc_barrier
def test_p22_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = assemble(u*v*dx)
    b = assemble(v*dx)
    u = Function(V)
    solve(A, u.vector(), b)


@skip_in_parallel
def test_p22_box_2():
    mesh = UnitCubeMesh(2, 2, 2)

    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    scalar = u(0.1, 0.2, 0.3)

    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    vector = u(0.1, 0.2, 0.3)


@use_gc_barrier
def test_p23_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V*Q)

    w = Function(W)
    u, p = w.split()

    uu, pp = w.split(deepcopy=True)


@skip_in_parallel
def test_p24_box_1():
    class MyExpression(Expression):
        def eval(self, values, x):
            values[0] = sin(x[0])*cos(x[1])
    f = MyExpression()
    print(f((0.5, 0.5)))


@skip_in_parallel
def test_p24_box_2():
    class MyExpression(Expression):
        def eval(self, values, x):
            values[0] = sin(x[0])
            values[1] = cos(x[1])
        def value_shape(self):
            return (2,)
    g = MyExpression()
    print(g((0.5, 0.5)))


@use_gc_barrier
def test_p24_box_3():
    f = Expression("sin(x[0])*cos(x[1])")
    g = Expression(("sin(x[0])", "cos(x[1])"))


@use_gc_barrier
def test_p25_box_1():
    t = 0.0
    T = 1.0
    dt = 0.5

    h = Expression("t*sin(x[0])*cos(x[1])", t=0.0)
    while t < T:
        h.t = t
        t += dt


@use_gc_barrier
def test_p26_box_1():
    mesh = UnitSquareMesh(8, 8)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression(("0.0", "0.0"))

    E = 10.0
    nu = 0.3

    mu = E/(2.0*(1.0 + nu))
    lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

    def sigma(v):
        # Note: Changed from v.cell().d to len(v), cell.d is being removed.
        return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

    a = inner(sigma(u), sym(grad(v)))*dx
    L = dot(f, v)*dx


@use_gc_barrier
def test_p27_box_1():
    mesh = UnitSquareMesh(3, 3)
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


@use_gc_barrier
def test_p29_box_1():
    mesh = UnitSquareMesh(3, 3)
    class NeumannBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.5 + DOLFIN_EPS and on_boundary
    neumann_boundary = NeumannBoundary()
    exterior_facet_domains = FacetFunction("size_t", mesh)
    exterior_facet_domains.set_all(1)
    neumann_boundary.mark(exterior_facet_domains, 0)


@use_gc_barrier
def test_p30_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    g = Function(V)
    h = Function(V)
    class NeumannBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.5 + DOLFIN_EPS and on_boundary
    neumann_boundary = NeumannBoundary()

    # Changed from notation dss = ds[neumann_boundary]
    dss = ds(subdomain_data=neumann_boundary)
    a = g*v*dss(0) + h*v*dss(1)


@use_gc_barrier
def test_p30_box_2():
    mesh = UnitSquareMesh(3, 3)
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


@use_gc_barrier
def test_p31_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u_0 = Expression("sin(x[0])")
    bc = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")


@use_gc_barrier
def test_p31_box_2():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    A, x, b = create_data()
    u_0 = Expression("sin(x[0])")
    bc = DirichletBC(V, u_0, "x[0] > 0.5 && on_boundary")
    u = Function(V)

    bc.apply(A, b)
    bc.apply(u.vector())


@use_gc_barrier
def test_p32_box_1():
    mesh = UnitSquareMesh(3, 3)
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


@use_gc_barrier
def test_p32_box_2():
    mesh = UnitSquareMesh(3, 3)
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


@skip_in_parallel
def test_p32_box_3():
    mesh = UnitSquareMesh(3, 3)
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
    solver.parameters["newton_solver"]["linear_solver"] = "gmres"
    solver.parameters["newton_solver"]["preconditioner"] = "ilu"
    solver.solve()


@use_gc_barrier
def test_p33_box_1():
    mesh = UnitSquareMesh(3, 3)
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


@use_gc_barrier
def test_p33_box_2():
    mesh = Mesh(os.path.join(os.path.dirname(__file__), "mesh.xml"))


@use_gc_barrier
def test_p34_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    mesh_function = MeshFunction("size_t", mesh, 0)

    #plot(u)
    #plot(mesh)
    #plot(mesh_function)


@use_gc_barrier
def test_p34_box_2():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    #plot(grad(u))
    #plot(u*u)
    element = FiniteElement("BDM", tetrahedron, 3)

    # Disabled since it claims the terminal
    #plot(element)


@use_gc_barrier
def test_p35_box_1(cd_tempdir):
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    file = File("solution.pvd")
    file << u


@use_gc_barrier
def test_p35_box_2(cd_tempdir):
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    t = 1.0

    file = File("solution.pvd", "compressed");
    file << (u, t)


@skip_in_parallel
def test_p36_box_1(cd_tempdir, pushpop_parameters):
    mesh = UnitSquareMesh(3, 3)
    matrix, x, vector = create_data()

    vector_file = File("vector.xml")
    vector_file << vector
    vector_file >> vector
    mesh_file = File("mesh.xml")
    mesh_file << mesh
    mesh_file >> mesh
    parameters_file = File("parameters.xml")
    parameters_file << parameters
    parameters_file >> parameters


@skip_in_parallel
def test_p37_box_1(cd_tempdir):
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    dt = 0.5
    t = 0.0
    T = 1.0

    time_series = TimeSeries(mpi_comm_world(), "simulation_data")
    while t < T:
        time_series.store(u.vector(), t)
        time_series.store(mesh, t)
        t += dt


@skip_in_parallel
def test_p37_box_2():
    time_series = TimeSeries(mpi_comm_world(), "simulation_data")
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    t = 0.5

    #time_series.retrieve(u.vector(), t)
    #time_series.retrieve(mesh, t)


@use_gc_barrier
def test_p37_box_3():
    M = 100
    N = 100
    info("Assembling system of size %d x %d." % (M, N))


@use_gc_barrier
def test_p38_box_1():
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
    print("Test message")


@use_gc_barrier
def test_p39_box_1():
    matrix, vector, b = create_data()
    solver = KrylovSolver()
    mesh = UnitSquareMesh(3, 3)
    mesh_function = MeshFunction("size_t", mesh, 0)
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


@use_gc_barrier
def test_p39_box_2():
    mesh = UnitSquareMesh(3, 3)
    info(mesh, True)


@use_gc_barrier
def test_p40_box_1():
    mesh = UnitSquareMesh(3, 3)
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


@use_gc_barrier
def test_p40_box_2():
    def solve(A, b):
        timer = Timer("Linear solve")
        x = 1
        return x
    solve(None, None)


@use_gc_barrier
def test_p41_box_1(pushpop_parameters):
    info(parameters, True)
    num_threads = parameters["num_threads"]
    allow_extrapolation = parameters["allow_extrapolation"]
    parameters["num_threads"] = 8
    parameters["allow_extrapolation"] = True


@use_gc_barrier
def test_p41_box_2():
    solver = KrylovSolver()
    solver.parameters["absolute_tolerance"] = 1e-6
    solver.parameters["report"] = True
    solver.parameters["gmres"]["restart"] = 50
    #solver.parameters["preconditioner"]["reuse"] = True
    solver.parameters["preconditioner"]["structure"] = "same"


@use_gc_barrier
def test_p42_box_1():
    my_parameters = Parameters("my_parameters")
    my_parameters.add("foo", 3)
    my_parameters.add("bar", 0.1)


@use_gc_barrier
def test_p42_box_2(pushpop_parameters):
    d = dict(num_threads=4, krylov_solver=dict(absolute_tolerance=1e-6))
    parameters.update(d)


@use_gc_barrier
def test_p42_box_3():
    my_parameters = Parameters("my_parameters", foo=3, bar=0.1,
                                nested=Parameters("nested", baz=True))


@use_gc_barrier
def test_p42_box_4(pushpop_parameters):
    argv = ["dummy.py"]
    parameters.parse(argv)

    # Original test was just:
    #parameters.parse()
    # but this is not testable without external sys.argv.
    # Feel free to improve by adding something to argv above.


@skip_in_parallel
def test_p43_box_1(cd_tempdir, pushpop_parameters):
    file = File("parameters.xml")
    file << parameters
    file >> parameters


@use_gc_barrier
def test_p45_box_1(pushpop_parameters):
    parameters["mesh_partitioner"] = "ParMETIS"


@use_gc_barrier
def test_p45_box_2():
    from dolfin import cpp
    Function = cpp.Function
    assemble = cpp.assemble


@use_gc_barrier
def test_p46_box_1():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)


@use_gc_barrier
def test_p47_box_1():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)


@use_gc_barrier
def test_p47_box_2():
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "Lagrange", 1)
    class Source(Expression):
        def eval(self, values, x):
            values[0] = sin(x[0])
    v = TestFunction(V)
    f = Source()
    L = f*v*dx
    assemble(L)


@use_gc_barrier
def test_p48_box_1():
    e = Expression("sin(x[0])")


@use_gc_barrier
def test_p49_box_1():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    c = Expression("sin(x[0])")
    a = c*dot(grad(u), grad(v))*dx
    A = assemble(a)


@use_gc_barrier
def test_p50_box_1(pushpop_parameters):
    parameters["form_compiler"]["name"] = "sfc"


@use_gc_barrier
def test_p50_box_2():
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx

    A = assemble(a)
    AA = A.array()


@skip_in_parallel
def test_p50_box_3(pushpop_parameters):
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(V)
    L = v*dx

    parameters["linear_algebra_backend"] = "Eigen"
    b = assemble(L)
    b = as_backend_type(b)
    bb = b.data()


@skip_in_parallel
def test_p51_box_1(pushpop_parameters):
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx
    parameters["linear_algebra_backend"] = "Eigen"
    A = assemble(a)

    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        pass
    else:
        A = as_backend_type(A)
        rows, columns, values = A.data()
        csr = csr_matrix((values, columns, rows))


@use_gc_barrier
def test_p51_box_2():
    from numpy import arange
    b = Vector(mpi_comm_world(), 10)
    c = Vector(mpi_comm_world(), 10)
    b_copy = b[:]
    b[:] = c
    b[b < 0] = 0

    # Since 1.5 we do not support slicing access as it does not make
    # sense in parallel
    #b2 = b[::2]

    # You can use an alternative syntax though
    b2 = b[arange(0, b.local_size(), 2)]


@skip_in_parallel
def test_p51_box_3():
    from numpy import array
    b = Vector(mpi_comm_world(), 20)
    b1 = b[[0, 4, 7, 10]]
    b2 = b[array((0, 4, 7, 10))]
