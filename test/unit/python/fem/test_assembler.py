#!/usr/bin/env py.test

"""Unit tests for assembly"""

# Copyright (C) 2011-2014 Garth N. Wells
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
# Modified by Martin Alnes 2014

import pytest
import os
import numpy
from dolfin import *

from dolfin_utils.test import skip_in_parallel, filedir, pushpop_parameters


def test_cell_size_assembly_1D():
    mesh = UnitIntervalMesh(10)
    assert round(assemble(CellSize(mesh)*dx) - 0.1, 12) == 0
    assert round(assemble(CellVolume(mesh)*dx) - 0.1, 12) == 0


def test_cell_assembly_1D():
    mesh = UnitIntervalMesh(48)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant(10.0)

    a = inner(grad(v), grad(u))*dx
    L = inner(v, f)*dx

    A_frobenius_norm = 811.75365721381274397572
    b_l2_norm = 1.43583841167606474087

    # Assemble A and b
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0


@skip_in_parallel
def test_cell_assembly_1D_multithreaded(pushpop_parameters):

    mesh = UnitIntervalMesh(48)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant(10.0)

    a = inner(grad(v), grad(u))*dx
    L = inner(v, f)*dx

    A_frobenius_norm = 811.75365721381274397572
    b_l2_norm = 1.43583841167606474087

    # Assemble A and b
    parameters["num_threads"] = 4
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0


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

    # Assemble A and b
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0


@skip_in_parallel
def test_cell_assembly_multithreaded():
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

    # Assemble A and b
    parameters["num_threads"] = 4
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0
    parameters["num_threads"] = 0


def test_facet_assembly():
    parameters["ghost_mode"] = "shared_facet"
    mesh = UnitSquareMesh(24, 24)
    parameters["ghost_mode"] = "none"
    V = FunctionSpace(mesh, "DG", 1)

    # Define test and trial functions
    v = TestFunction(V)
    u = TrialFunction(V)

    # Define normal component, mesh size and right-hand side
    n = FacetNormal(mesh)
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

    # Assemble A and b
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0


@skip_in_parallel
def test_facet_assembly_multithreaded(pushpop_parameters):
    mesh = UnitSquareMesh(24, 24)
    V = FunctionSpace(mesh, "DG", 1)

    # Define test and trial functions
    v = TestFunction(V)
    u = TrialFunction(V)

    # Define normal component, mesh size and right-hand side
    n = FacetNormal(mesh)
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

    # Assemble A and b
    parameters["num_threads"] = 4
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0


def test_functional_assembly():
    mesh = UnitSquareMesh(24, 24)

    f = Constant(1.0)
    M0 = f*dx(mesh)
    assert round(assemble(M0) - 1.0, 7) == 0

    M1 = f*ds(mesh)
    assert round(assemble(M1) - 4.0, 7) == 0


@skip_in_parallel
def test_functional_assembly_multithreaded(pushpop_parameters):

    mesh = UnitSquareMesh(24, 24)

    f = Constant(1.0)
    M0 = f*dx(mesh)
    assert round(assemble(M0) - 1.0, 7) == 0

    M1 = f*ds(mesh)
    assert round(assemble(M1) - 4.0, 7) == 0

    parameters["num_threads"] = 4
    assert round(assemble(M0) - 1.0, 7) == 0


def test_subdomain_and_fulldomain_assembly_meshdomains():
    """Test assembly over subdomains AND the full domain with markers
    stored as part of the mesh.
    """

    # Create a mesh of the unit cube
    mesh = UnitCubeMesh(4, 4, 4)

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

    # Define subdomains for 3 parts of the unit cube
    class S0(SubDomain):
        def inside(self, x, inside):
            return x[0] > 0.25

    class S1(SubDomain):
        def inside(self, x, inside):
            return x[0] > 0.5

    class S2(SubDomain):
        def inside(self, x, inside):
            return x[0] > 0.75

    # Mark mesh
    f0 = F0()
    f1 = F1()
    f2 = F2()
    f0.mark_facets(mesh, 0)
    f1.mark_facets(mesh, 1)
    f2.mark_facets(mesh, 3)  # NB! 3, to leave a gap

    s0 = S0()
    s1 = S1()
    s2 = S2()
    s0.mark_cells(mesh, 0)
    s1.mark_cells(mesh, 1)
    s2.mark_cells(mesh, 3)  # NB! 3, to leave a gap

    # Assemble forms on subdomains and full domain and compare
    krange = list(range(5))
    for dmu in (dx, ds):
        full = assemble(Constant(3.0)*dmu(mesh))
        subplusfull = [assemble(Constant(3.0)*dmu(mesh) +
                                Constant(1.0)*dmu(k, domain=mesh))
                       for k in krange]
        sub = [assemble(Constant(1.0)*dmu(k, domain=mesh)) for k in krange]
        for k in krange:
            # print sub[k] + full, subplusfull[k]
            assert round(sub[k] + full - subplusfull[k], 7) == 0


@skip_in_parallel
def test_subdomain_assembly_form_1():
    "Test assembly over subdomains with markers stored as part of form"

    mesh = UnitSquareMesh(4, 4)

    # Define cell/facet function
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.49
    subdomains = CellFunction("size_t", mesh)
    subdomains.set_all(0)
    left = Left()
    left.mark(subdomains, 1)

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.95
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    right = RightBoundary()
    right.mark(boundaries, 1)

    V = FunctionSpace(mesh, "CG", 2)
    f = Expression("x[0] + 2", degree=1)
    g = Expression("x[1] + 1", degree=1)

    f = interpolate(f, V)
    g = interpolate(g, V)

    mesh1 = subdomains.mesh()
    mesh2 = boundaries.mesh()
    assert mesh1.id() == mesh2.id()
    assert mesh1.ufl_domain().ufl_id() == mesh2.ufl_domain().ufl_id()

    dxs = dx(subdomain_data=subdomains)
    dss = ds(subdomain_data=boundaries)
    assert dxs.ufl_domain() == None
    assert dss.ufl_domain() == None
    assert dxs.subdomain_data() == subdomains
    assert dss.subdomain_data() == boundaries

    M = f*f*dxs(0) + g*f*dxs(1) + f*f*dss(1)
    assert M.ufl_domains() == (mesh.ufl_domain(),)
    sd = M.subdomain_data()[mesh.ufl_domain()]
    assert sd["cell"] == subdomains
    assert sd["exterior_facet"] == boundaries

    # Check that subdomains are respected
    reference = 15.0
    assert round(assemble(M) - reference, 10) == 0

    # Check that the form itself assembles as before
    assert round(assemble(M) - reference, 10) == 0

    # Take action of derivative of M on f
    df = TestFunction(V)
    L = derivative(M, f, df)
    dg = TrialFunction(V)
    F = derivative(L, g, dg)
    b = action(F, f)

    # Check that domain data carries across transformations:
    reference = 0.136477465659
    assert round(assemble(b).norm("l2") - reference, 8) == 0


@skip_in_parallel
def test_subdomain_assembly_form_1_multithreaded(pushpop_parameters):
    "Test assembly over subdomains with markers stored as part of form"

    mesh = UnitSquareMesh(4, 4)

    # Define cell/facet function
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.49
    subdomains = CellFunction("size_t", mesh)
    subdomains.set_all(0)
    left = Left()
    left.mark(subdomains, 1)

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.95
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    right = RightBoundary()
    right.mark(boundaries, 1)

    V = FunctionSpace(mesh, "CG", 2)
    f = Expression("x[0] + 2", degree=1)
    g = Expression("x[1] + 1", degree=1)

    f = interpolate(f, V)
    g = interpolate(g, V)

    mesh1 = subdomains.mesh()
    mesh2 = boundaries.mesh()
    assert mesh1.id() == mesh2.id()
    assert mesh1.ufl_domain().ufl_id() == mesh2.ufl_domain().ufl_id()

    dxs = dx(subdomain_data=subdomains)
    dss = ds(subdomain_data=boundaries)
    assert dxs.ufl_domain() == None
    assert dss.ufl_domain() == None
    assert dxs.subdomain_data() == subdomains
    assert dss.subdomain_data() == boundaries

    M = f*f*dxs(0) + g*f*dxs(1) + f*f*dss(1)
    assert M.ufl_domains() == (mesh.ufl_domain(),)
    sd = M.subdomain_data()[mesh.ufl_domain()]
    assert sd["cell"] == subdomains
    assert sd["exterior_facet"] == boundaries

    # Check that subdomains are respected
    reference = 15.0
    assert round(assemble(M) - reference, 10) == 0

    # Assemble form (multi-threaded)
    parameters["num_threads"] = 4
    assert round(assemble(M) - reference, 10) == 0

    # Check that the form itself assembles as before
    assert round(assemble(M) - reference, 10) == 0

    # Assemble form  (multi-threaded)
    assert round(assemble(M) - reference, 10) == 0

    # Take action of derivative of M on f
    df = TestFunction(V)
    L = derivative(M, f, df)
    dg = TrialFunction(V)
    F = derivative(L, g, dg)
    b = action(F, f)

    # Check that domain data carries across transformations:
    reference = 0.136477465659
    assert round(assemble(b).norm("l2") - reference, 8) == 0

    # Assemble form  (multi-threaded)
    assert round(assemble(b).norm("l2") - reference, 8) == 0


def test_subdomain_assembly_form_2():
    "Test assembly over subdomains with markers stored as part of form"

    # Define mesh
    mesh = UnitSquareMesh(8, 8)

    # Define domain for lower left corner
    class MyDomain(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.5 + DOLFIN_EPS and x[1] < 0.5 + DOLFIN_EPS
    my_domain = MyDomain()

    # Define boundary for lower left corner
    class MyBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] < 0.5 + DOLFIN_EPS and x[1] < DOLFIN_EPS) or \
                   (x[1] < 0.5 + DOLFIN_EPS and x[0] < DOLFIN_EPS)
    my_boundary = MyBoundary()

    # Mark mesh functions
    D = mesh.topology().dim()
    cell_domains = MeshFunction("size_t", mesh, D)
    exterior_facet_domains = MeshFunction("size_t", mesh, D - 1)
    cell_domains.set_all(1)
    exterior_facet_domains.set_all(1)
    my_domain.mark(cell_domains, 0)
    my_boundary.mark(exterior_facet_domains, 0)

    # Define forms
    c = Constant(1.0)

    a0 = c*dx(0, domain=mesh, subdomain_data=cell_domains)
    a1 = c*ds(0, domain=mesh, subdomain_data=exterior_facet_domains)

    assert round(assemble(a0) - 0.25, 7) == 0
    assert round(assemble(a1) - 1.0, 7) == 0


@skip_in_parallel
def test_colored_cell_assembly():

    old_mesh = UnitCubeMesh(4, 4, 4)

    # Create mesh, then color and renumber
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

    A_frobenius_norm = 4.3969686527582512
    b_l2_norm = 0.95470326978246278

    # Assemble A and b
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0

    # Assemble A and b multi-threaded
    parameters["num_threads"] = 4
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    assert round(assemble(L).norm("l2") - b_l2_norm, 10) == 0
    parameters["num_threads"] = 0


def test_nonsquare_assembly():
    """Test assembly of a rectangular matrix"""

    mesh = UnitSquareMesh(16, 16)

    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = V*Q
    V = FunctionSpace(mesh, V)
    Q = FunctionSpace(mesh, Q)
    W = FunctionSpace(mesh, W)

    (v, q) = TestFunctions(W)
    (u, p) = TrialFunctions(W)
    a = div(v)*p*dx
    A_frobenius_norm = 9.6420303878382718e-01
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0

    v = TestFunction(V)
    p = TrialFunction(Q)
    a = inner(grad(p), v)*dx
    A_frobenius_norm = 0.935414346693
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0


@skip_in_parallel
def test_nonsquare_assembly_multithreaded():
    """Test assembly of a rectangular matrix"""

    mesh = UnitSquareMesh(16, 16)

    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V*Q)

    (v, q) = TestFunctions(W)
    (u, p) = TrialFunctions(W)

    a = div(v)*p*dx
    A_frobenius_norm = 9.6420303878382718e-01

    parameters["num_threads"] = 4
    assert round(assemble(a).norm("frobenius") - A_frobenius_norm, 10) == 0
    parameters["num_threads"] = 0


@skip_in_parallel
def test_reference_assembly(filedir, pushpop_parameters):
    "Test assembly against a reference solution"

    # NOTE: This test is not robust as it relies on specific
    #       DOF order, which cannot be guaranteed
    reorder_dofs = parameters["reorder_dofs_serial"]
    parameters["reorder_dofs_serial"] = False

    # Load reference mesh (just a simple tetrahedron)
    mesh = Mesh(os.path.join(filedir, "tetrahedron.xml"))

    # Assemble stiffness and mass matrices
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A, M = EigenMatrix(), EigenMatrix()
    assemble(dot(grad(v), grad(u))*dx, tensor=A)
    assemble(v*u*dx, tensor=M)

    # Run test (requires SciPy)
    try:
        import scipy
        A = A.sparray().todense()
        M = M.sparray().todense()

        # Create reference matrices and set entries
        A0 = numpy.array([[1.0/2.0, -1.0/6.0, -1.0/6.0, -1.0/6.0],
                          [-1.0/6.0, 1.0/6.0, 0.0, 0.0],
                          [-1.0/6.0, 0.0, 1.0/6.0, 0.0],
                          [-1.0/6.0, 0.0, 0.0, 1.0/6.0]])
        M0 = numpy.array([[1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0],
                          [1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0],
                          [1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0],
                          [1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0]])

        C = A - A0
        assert round(numpy.linalg.norm(C, 'fro') - 0.0, 7) == 0
        D = M - M0
        assert round(numpy.linalg.norm(D, 'fro') - 0.0, 7) == 0

    except:
        print("Cannot run this test without SciPy")


def test_ways_to_pass_mesh_to_assembler():
    mesh = UnitSquareMesh(16, 16)

    # Geometry with mesh (ufl.Domain with mesh in domain data)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Geometry with just cell (no reference to mesh, for backwards
    # compatibility)
    x2 = SpatialCoordinate(mesh)
    n2 = FacetNormal(mesh)

    # A function equal to x[0] for comparison
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(Expression("x[0]", degree=1))

    # An expression equal to x[0], with different geometry info:
    e = Expression("x[0]", degree=1)  # nothing
    e2 = Expression("x[0]", cell=mesh.ufl_cell(), degree=1)  # cell
    e3 = Expression("x[0]", element=V.ufl_element())  # ufl element
    e4 = Expression("x[0]", domain=mesh, degree=1)  # mesh

    # Provide mesh in measure:
    dx2 = Measure("dx", domain=mesh)
    assert round(1.0 - assemble(1*dx(mesh)), 7) == 0
    assert round(1.0 - assemble(Constant(1.0)*dx(mesh)), 7) == 0
    assert round(1.0 - assemble(Constant(1.0)*dx2), 7) == 0

    # Try with cell argument to Constant as well:
    assert round(1.0 - assemble(Constant(1.0,
                                         cell=mesh.ufl_cell())*dx(mesh))) == 0
    assert round(1.0 - assemble(Constant(1.0, cell=mesh.ufl_cell())*dx2)) == 0
    assert round(1.0 - assemble(Constant(1.0,
                                         cell=mesh.ufl_cell())*dx(mesh))) == 0
    assert round(1.0 - assemble(Constant(1.0, cell=mesh.ufl_cell())*dx2)) == 0

    # Geometric quantities with mesh in domain:
    assert round(0.5 - assemble(x[0]*dx), 7) == 0
    assert round(0.5 - assemble(x[0]*dx(mesh)), 7) == 0

    # Geometric quantities without mesh in domain:
    assert round(0.5 - assemble(x2[0]*dx(mesh)), 7) == 0

    # Functions with mesh in domain:
    assert round(0.5 - assemble(f*dx), 7) == 0
    assert round(0.5 - assemble(f*dx(mesh)), 7) == 0

    # Expressions with and without mesh in domain:
    assert round(0.5 - assemble(e*dx(mesh)), 7) == 0
    assert round(0.5 - assemble(e2*dx(mesh)), 7) == 0
    assert round(0.5 - assemble(e3*dx(mesh)), 7) == 0
    assert round(0.5 - assemble(e4*dx), 7) == 0  # e4 has a domain with mesh reference
    assert round(0.5 - assemble(e4*dx(mesh)), 7) == 0

    # Geometric quantities with mesh in domain:
    assert round(0.0 - assemble(n[0]*ds), 7) == 0
    assert round(0.0 - assemble(n[0]*ds(mesh)), 7) == 0

    # Geometric quantities without mesh in domain:
    assert round(0.0 - assemble(n2[0]*ds(mesh)), 7) == 0
