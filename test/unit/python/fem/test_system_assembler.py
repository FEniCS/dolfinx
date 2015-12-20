#!/usr/bin/env py.test

"""Unit tests for class SystemAssembler"""

# Copyright (C) 2011-2013 Garth N. Wells, 2013 Jan Blechta
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

import pytest
import numpy
import os
from dolfin import *
from dolfin_utils.test import *


def test_cell_assembly():

    mesh = UnitCubeMesh(4, 4, 4)
    V = VectorFunctionSpace(mesh, "DG", 1)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant((10, 20, 30))

    def epsilon(v):
        return 0.5*(grad(v) + grad(v).T)

    a = inner(epsilon(v), epsilon(u))*dx
    L = inner(v, f)*dx

    A_frobenius_norm = 4.3969686527582512
    b_l2_norm = 0.95470326978246278

    # Assemble system
    A, b = assemble_system(a, L)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0

    # SystemAssembler construction
    assembler = SystemAssembler(a, L)

    # Test SystemAssembler for LHS and RHS
    A = Matrix()
    b = Vector()
    assembler.assemble(A, b)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0

    A = Matrix()
    b = Vector()

    assembler.assemble(A)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0

    assembler.assemble(b)
    assert round(b.norm("l2") - b_l2_norm, 10) == 0


def test_cell_assembly_bc():

    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    bc = DirichletBC(V, 1.0, "on_boundary")

    u, v = TrialFunction(V), TestFunction(V)
    f = Constant(10)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A_frobenius_norm = 96.847818767384
    b_l2_norm = 96.564760289080

    # Assemble system
    A, b = assemble_system(a, L, bc)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0

    # Create assembler
    assembler = SystemAssembler(a, L, bc)

    # Test for assembling A and b via assembler object
    A, b = Matrix(), Vector()
    assembler.assemble(A, b)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(b.norm("l2") - b_l2_norm, 10) == 0

    # Assemble LHS only (first time)
    A = Matrix()
    assembler.assemble(A)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0

    # Assemble LHS only (second time)
    assembler.assemble(A)
    assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0

    # Assemble RHS only (first time)
    b = Vector()
    assembler.assemble(b)
    assert round(b.norm("l2") - b_l2_norm, 10) == 0

    # Assemble RHS only (second time time)
    assembler.assemble(b)
    assert round(b.norm("l2") - b_l2_norm, 10) == 0


def test_facet_assembly():

    def test(mesh):
        V = FunctionSpace(mesh, "DG", 1)

        # Define test and trial functions
        v = TestFunction(V)
        u = TrialFunction(V)

        # Define normal component, mesh size and right-hand side
        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg = (h('+') + h('-'))/2
        f = Expression("500.0*exp(-(pow(x[0] - 0.5, 2) \
+ pow(x[1] - 0.5, 2)) / 0.02)", degree=1)

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

        # Assemble system
        A, b = assemble_system(a, L)
        assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
        assert round(b.norm("l2") - b_l2_norm, 10) == 0

        # Test SystemAssembler
        assembler = SystemAssembler(a, L)
        A = Matrix()
        b = Vector()

        assembler.assemble(A, b)
        assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
        assert round(b.norm("l2") - b_l2_norm, 10) == 0

        A = Matrix()
        b = Vector()
        assembler.assemble(A)
        assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
        assembler.assemble(b)
        assert round(b.norm("l2") - b_l2_norm, 10) == 0

    parameters["ghost_mode"] = "shared_facet"
    mesh = UnitSquareMesh(24, 24)
    test(mesh)

    parameters["ghost_mode"] = "shared_vertex"
    mesh = UnitSquareMesh(24, 24)
    test(mesh)

    parameters["ghost_mode"] = "none"


def test_vertex_assembly():

    # Create mesh and define function space
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    def center_func(x):
        return 0.25 <= x[0] and x[0] <= 0.75 and near(x[1], 0.5)

    # Define domain for point integral
    center_domain = VertexFunction("size_t", mesh, 0)
    center = AutoSubDomain(center_func)
    center.mark(center_domain, 1)
    dPP = dP(subdomain_data=center_domain)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(.4)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dPP(1)

    with pytest.raises(RuntimeError):
        A, b = assemble_system(a, L)


def test_incremental_assembly():

    for f in [Constant(0.0), Constant(1e4)]:

        # Laplace/Poisson problem
        mesh = UnitSquareMesh(20, 20)
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        a, L = inner(grad(u), grad(v))*dx, f*v*dx
        uD = Expression("42.0*(2.0*x[0]-1.0)")
        bc = DirichletBC(V, uD, "on_boundary")

        # Initialize initial guess by some number
        u = Function(V)
        x = u.vector()
        x[:] = 30.0

        # Assemble incremental system
        assembler = SystemAssembler(a, -L, bc)
        A, b = Matrix(), Vector()
        assembler.assemble(A, b, x)

        # Solve for (negative) increment
        Dx = Vector(x)
        Dx.zero()
        solve(A, Dx, b)

        # Update solution
        x[:] -= Dx[:]

        # Check solution
        u_true = Function(V)
        solve(a == L, u_true, bc)
        u.vector()[:] -= u_true.vector()[:]
        error = norm(u.vector(), 'linf')
        assert round(error - 0.0, 7) == 0


@skip_in_parallel
def test_domains():

    class RightSubDomain(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.5

    mesh = UnitSquareMesh(24, 24)

    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    sub_domains.set_all(1)
    right = RightSubDomain()
    right.mark(sub_domains, 2)

    V = FunctionSpace(mesh, "DG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    # the numerical answer (initialized to some number)
    x = Function(V)
    x.vector()[:] = 30.0

    dx = Measure("dx", subdomain_data=sub_domains)
    # the forms
    a = v*u*dx(1) + 2*v*u*dx(2)
    L = v*Constant(1.0)*dx(1) + v*Constant(2.0)*dx(2)
    # test cell-wise assembly
    assembler = SystemAssembler(a, L)

    A0 = Matrix()
    b = Vector()
    assembler.assemble(A0, b)

    solve(A0, x.vector(), b)

    # check solution
    x.vector()[:] -= 1.0
    error = norm(x.vector(), 'linf')
    assert round(error - 0.0, 7) == 0

    # now give the form some internal facet integrals
    a += v('+')*u('+')*Constant(0.0)('+')*dS
    assembler = SystemAssembler(a, L)
    A1 = Matrix()
    # test facet-wise assembly
    assembler.assemble(A1, b)

    # reset solution vector to some number
    x.vector()[:] = 30.0
    solve(A1, x.vector(), b)

    # check solution
    x.vector()[:] -= 1.0
    error = norm(x.vector(), 'linf')
    assert round(error - 0.0, 7) == 0


@skip_in_parallel
def test_facet_assembly_cellwise_insertion(filedir):

    def run_test(mesh):
        c_f = FunctionSpace(mesh, "DG", 0)
        v = Constant((-1.0,))
        dt = Constant(1.0)

        c_t = TestFunction(c_f)
        c_a = TrialFunction(c_f)

        n = FacetNormal(mesh)
        vn = dot(v, n)
        vout = 0.5*(vn + abs(vn))

        # forms:
        # a has no facet integrals
        a = c_t*c_a*dx
        # L has facet integrals so we end up in facet wise assembly
        L = c_t('+')*vout('+')*dt('+')*dS + c_t('-')*vout('-')*dt('-')*dS  \
            + c_t*vout*dt*ds
        # but have to use cell wise insertion because the sparsity
        # pattern doesn't support the macro element

        A = Matrix()
        b = Vector()

        assembler = SystemAssembler(a, L)
        assembler.assemble(A, b)

        A_frobenius_norm = ((0.1**2)*10)**0.5
        A_linf_norm = 0.1
        b_l2_norm = 10.0**0.5
        b_linf_norm = 1.0

        assert round(A.norm("frobenius") - A_frobenius_norm, 10) == 0
        assert round(b.norm("l2") - b_l2_norm, 10) == 0
        assert round(A.norm("linf") - A_linf_norm, 10) == 0
        assert round(b.norm("linf") - b_linf_norm, 10) == 0

        x = Function(c_f)
        x.vector()[:] = 30.0

        solver_worked = True
        try:
            solve(A, x.vector(), b)
        except:
            solver_worked = False
        assert solver_worked

        x.vector()[:] -= 10.0
        error = norm(x.vector(), 'linf')
        assert round(error - 0.0, 7) == 0

    # Run tests
    run_test(UnitIntervalMesh(10))
    run_test(Mesh(os.path.join(filedir, "gmsh_unit_interval.xml")))


def test_non_square_assembly():
    mesh = UnitSquareMesh(14, 14)

    def bound(x):
        return (x[0] == 0)

    # Assemble four blocks in VxV, VxQ, QxV and VxV
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Q = FunctionSpace(mesh, P1)
    V = FunctionSpace(mesh, P2)
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    a00 = inner(grad(u), grad(v))*dx
    a01 = dot(grad(p), v)*dx
    a10 = q*div(u)*dx
    a11 = p*q*dx
    L0 = dot(Constant((0.0, 0.0)), v)*dx
    L1 = Constant(0.0)*q*dx
    bc = DirichletBC(V.sub(0), Constant(1.0), bound)

    assembler = SystemAssembler(a00, L0, bc)
    A = Matrix()
    b = Vector()
    assembler.assemble(A, b)
    Anorm1 = A.norm("frobenius")**2

    assembler = SystemAssembler(a01, L0, bc)
    A = Matrix()
    assembler.add_values = True
    assembler.assemble(A, b)
    Anorm1 += A.norm("frobenius")**2
    bnorm1 = b.norm("l2")**2

    assembler = SystemAssembler(a10, L1, bc)
    A = Matrix()
    b = Vector()
    assembler.assemble(A, b)
    Anorm1 += A.norm("frobenius")**2

    assembler = SystemAssembler(a11, L1, bc)
    A = Matrix()
    assembler.add_values = True
    assembler.assemble(A, b)
    Anorm1 += A.norm("frobenius")**2
    bnorm1 += b.norm("l2")**2

    # Same problem as a MixedFunctionSpace
    W = FunctionSpace(mesh, P2*P1)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx + dot(grad(p), v)*dx + q*div(u)*dx + p*q*dx
    L = dot(Constant((0.0, 0.0)), v)*dx + Constant(0.0)*q*dx
    bc = DirichletBC(W.sub(0).sub(0), Constant(1.0), bound)
    assembler = SystemAssembler(a, L, bc)
    A = Matrix()
    b = Vector()
    assembler.assemble(A, b)

    bnorm2 = b.norm("l2")**2
    Anorm2 = A.norm("frobenius")**2
    assert round(1.0 - bnorm1/bnorm2, 10) == 0
    assert round(1.0 - Anorm1/Anorm2, 10) == 0
