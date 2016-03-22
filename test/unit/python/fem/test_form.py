#!/usr/bin/env py.test

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

import pytest
import numpy
import ufl
from dolfin import *
from dolfin_utils.test import skip_in_parallel, fixture


@fixture
def square():
    return UnitSquareMesh(8, 8)


@fixture
def square_boundary(square):
    return BoundaryMesh(square, "exterior")


@fixture
def cube_boundary(cube):
    return BoundaryMesh(cube, "exterior")


@fixture
def cube():
    return UnitCubeMesh(2, 2, 2)


@fixture
def V1(square_boundary):
    return FunctionSpace(square_boundary, "CG", 1)


@fixture
def VV1(square_boundary):
    return VectorFunctionSpace(square_boundary, "CG", 1)


@fixture
def Q1(square_boundary):
    return FunctionSpace(square_boundary, "DG", 0)


@fixture
def V2(cube_boundary):
    return FunctionSpace(cube_boundary, "CG", 1)


@fixture
def VV2(cube_boundary):
    return VectorFunctionSpace(cube_boundary, "CG", 1)


@fixture
def Q2(cube_boundary):
    return FunctionSpace(cube_boundary, "DG", 0)


def test_assemble_functional(V1, V2):

    mesh = V1.mesh()
    surfacearea = assemble(1*dx(mesh))
    assert round(surfacearea - 4.0, 7) == 0

    u = Function(V1)
    u.vector()[:] = 1.0
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 4.0, 7) == 0

    u = Function(V2)
    u.vector()[:] = 1.0
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 6.0, 7) == 0

    f = Expression("1.0", degree=0)
    u = interpolate(f, V1)
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 4.0, 7) == 0

    f = Expression("1.0", degree=0)
    u = interpolate(f, V2)
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 6.0, 7) == 0


def test_assemble_linear(V1, Q1, square_boundary, V2, Q2, cube_boundary):

    u = Function(V1)
    w = TestFunction(Q1)
    u.vector()[:] = 0.5
    facetareas = MPI.sum(square_boundary.mpi_comm(),
                         assemble(u*w*dx).array().sum())
    assert round(facetareas - 2.0, 7) == 0

    u = Function(V2)
    w = TestFunction(Q2)
    u.vector()[:] = 0.5
    a = u*w*dx
    b = assemble(a)
    facetareas = MPI.sum(cube_boundary.mpi_comm(),
                         assemble(u*w*dx).array().sum())
    assert round(facetareas - 3.0, 7) == 0

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
    assert round(a - b, 7) == 0


@skip_in_parallel
def test_assemble_bilinear_1D_2D(square, V1, square_boundary):

    V = FunctionSpace(square, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    bu = TrialFunction(V1)
    bv = TestFunction(V1)

    a = MPI.sum(square.mpi_comm(),
                assemble(inner(u, v)*ds).array().sum())
    b = MPI.sum(square_boundary.mpi_comm(),
                assemble(inner(bu, bv)*dx).array().sum())
    assert round(a - b, 7) == 0

    # Assemble over subset of mesh facets
    subdomain = CompiledSubDomain("near(x[1], 0.0)")
    bottom = FacetFunctionSizet(square)
    bottom.set_all(0)
    subdomain.mark(bottom, 1)
    dss = ds(subdomain_data=bottom)
    foo = MPI.sum(square.mpi_comm(),
                  abs(assemble(inner(grad(u)[0],
                                     grad(v)[0])*dss(1)).array()).sum())

    # Assemble over all cells of submesh created from subset of
    # boundary mesh
    bottom2 = CellFunctionSizet(square_boundary)
    bottom2.set_all(0)
    subdomain.mark(bottom2, 1)
    BV = FunctionSpace(SubMesh(square_boundary, bottom2, 1), "CG", 1)
    bu = TrialFunction(BV)
    bv = TestFunction(BV)
    bar = MPI.sum(square_boundary.mpi_comm(),
                  abs(assemble(inner(grad(bu)[0],
                                     grad(bv)[0])*dx).array()).sum())
    # Should give same result
    assert round(bar - foo, 7) == 0


@skip_in_parallel
def test_assemble_bilinear_2D_3D(cube, V2, cube_boundary):

    V = FunctionSpace(cube, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # V2 is a FunctionSpace over cube_boundary
    bu = TrialFunction(V2)
    bv = TestFunction(V2)

    a = MPI.sum(cube.mpi_comm(),
                assemble(inner(u, v)*ds).array().sum())
    b = MPI.sum(cube_boundary.mpi_comm(),
                assemble(inner(bu, bv)*dx).array().sum())
    assert round(a - b, 7) == 0

    # Assemble over subset of mesh facets
    subdomain = CompiledSubDomain("near(x[1], 0.0)")
    bottom = FacetFunctionSizet(cube)
    bottom.set_all(0)
    subdomain.mark(bottom, 1)
    dss = ds(subdomain_data=bottom)
    foo = MPI.sum(cube.mpi_comm(),
                  abs(assemble(inner(grad(u)[0],
                                     grad(v)[0])*dss(1)).array()).sum())

    # Assemble over all cells of submesh created from subset of
    # boundary mesh
    bottom2 = CellFunctionSizet(cube_boundary)
    bottom2.set_all(0)
    subdomain.mark(bottom2, 1)
    BV = FunctionSpace(SubMesh(cube_boundary, bottom2, 1), "CG", 1)
    bu = TrialFunction(BV)
    bv = TestFunction(BV)
    bar = MPI.sum(cube_boundary.mpi_comm(),
                  abs(assemble(inner(grad(bu)[0],
                                     grad(bv)[0])*dx).array()).sum())

    # Should give same result
    assert round(bar - foo, 7) == 0


@fixture
def base():
    n = 16
    plane = CompiledSubDomain("near(x[1], 1.0)")
    square = UnitSquareMesh(n, n)

    square3d = SubMesh(BoundaryMesh(UnitCubeMesh(n, n, n), "exterior"), plane)
    global_normal = Expression(("0.0", "1.0", "0.0"), degree=0)
    square3d.init_cell_orientations(global_normal)

    RT2 = FiniteElement("RT", square.ufl_cell(), 1)
    RT3 = FiniteElement("RT", square3d.ufl_cell(), 1)
    DG2 = FiniteElement("DG", square.ufl_cell(), 0)
    DG3 = FiniteElement("DG", square3d.ufl_cell(), 0)

    return [(RT2, RT3), (DG2, DG3), (square, square3d)]


@fixture
def RT2(base):
    return FunctionSpace(base[2][0], base[0][0])


@fixture
def RT3(base):
    return FunctionSpace(base[2][1], base[0][1])


@fixture
def W2(base):
    """ RT2 * DG2 """
    return FunctionSpace(base[2][0], base[0][0] * base[1][0])


@fixture
def W3(base):
    """ RT3 * DG3 """
    return FunctionSpace(base[2][1], base[0][1] * base[1][1])


@fixture
def QQ2(base):
    """ DG2 * DG2 """
    return FunctionSpace(base[2][0], base[1][0] * base[1][0])


@fixture
def QQ3(base):
    """ DG3 * DG3 """
    return FunctionSpace(base[2][1], base[1][1] * base[1][1])


@skip_in_parallel
def test_basic_rt(RT2, RT3):

    f2 = Expression(("2.0", "1.0"), degree=0)
    f3 = Expression(("1.0", "0.0", "2.0"), degree=0)

    u2 = TrialFunction(RT2)
    u3 = TrialFunction(RT3)
    v2 = TestFunction(RT2)
    v3 = TestFunction(RT3)

    # Project
    pw2 = project(f2, RT2)
    pw3 = project(f3, RT3)
    pa2 = assemble(pw2**2*dx)
    pa3 = assemble(pw3**2*dx)

    # Project explicitly
    a2 = inner(u2, v2)*dx
    a3 = inner(u3, v3)*dx
    L2 = inner(f2, v2)*dx
    L3 = inner(f3, v3)*dx
    w2 = Function(RT2)
    w3 = Function(RT3)
    A2 = assemble(a2)
    b2 = assemble(L2)
    A3 = assemble(a3)
    b3 = assemble(L3)
    solve(A2, w2.vector(), b2)
    solve(A3, w3.vector(), b3)
    a2 = assemble(w2**2*dx)
    a3 = assemble(w3**2*dx)

    # Compare various results
    assert round((w2.vector() - pw2.vector()).norm("l2"), 5) == 0
    assert round((w3.vector() - pw3.vector()).norm("l2"), 5) == 0
    # 2d
    assert round(a2 - 5.0, 7) == 0
    assert round(pa2 - 5.0, 7) == 0
    # 3d
    assert round(a3 - 5.0, 7) == 0
    assert round(pa3 - 5.0, 6) == 0


@skip_in_parallel
def test_mixed_poisson_solve(W2, W3):

    f = Constant(1.0)

    # Solve mixed Poisson on standard unit square
    (sigma2, u2) = TrialFunctions(W2)
    (tau2, v2) = TestFunctions(W2)
    a = (inner(sigma2, tau2) + div(tau2)*u2 + div(sigma2)*v2)*dx
    L = f*v2*dx
    w2 = Function(W2)
    solve(a == L, w2)

    # Solve mixed Poisson on unit square in 3D
    (sigma3, u3) = TrialFunctions(W3)
    (tau3, v3) = TestFunctions(W3)
    a = (inner(sigma3, tau3) + div(tau3)*u3 + div(sigma3)*v3)*dx
    L = f*v3*dx
    w3 = Function(W3)
    solve(a == L, w3)

    # Check that results are about the same
    assert round(assemble(inner(w2, w2)*dx) - assemble(inner(w3, w3)*dx)) == 0


@fixture
def m():
    return 3


@fixture
def cube(m):
    return UnitCubeMesh(m, m, m)


@fixture
def cube_boundary(cube):
    return BoundaryMesh(cube, "exterior")


@fixture
def plane():
    return CompiledSubDomain("near(x[1], 0.0)")


@fixture
def square_boundary_(m):
    square = UnitSquareMesh(m, m)
    return BoundaryMesh(square, "exterior")


@fixture
def line():
    return CompiledSubDomain("near(x[0], 0.0)")


@fixture
def mesh3(cube_boundary, plane):
    return BoundaryMesh(SubMesh(cube_boundary, plane), "exterior")


@fixture
def bottom1(square_boundary_, plane):
    return SubMesh(square_boundary_, plane)


@fixture
def bottom2(cube_boundary, plane):
    return SubMesh(cube_boundary, plane)


@fixture
def bottom3(mesh3, line):
    return SubMesh(mesh3, line)


@skip_in_parallel
def test_normals_2D_1D(bottom1, m):
    "Testing assembly of normals for 1D meshes embedded in 2D"

    n = FacetNormal(bottom1)
    a = inner(n, n)*ds
    value_bottom1 = assemble(a)
    assert round(value_bottom1 - 2.0, 7) == 0
    b = inner(n('+'), n('+'))*dS
    b1 = assemble(b)
    c = inner(n('+'), n('-'))*dS
    c1 = assemble(c)
    assert round(b1 - m + 1, 7) == 0
    assert round(c1 + b1, 7) == 0


@skip_in_parallel
def test_normals_3D_1D(bottom3, m):
    "Testing assembly of normals for 1D meshes embedded in 3D"

    n = FacetNormal(bottom3)
    a = inner(n, n)*ds
    v1 = assemble(a)
    assert round(v1 - 2.0, 7) == 0
    b = inner(n('+'), n('+'))*dS
    b1 = assemble(b)
    c = inner(n('+'), n('-'))*dS
    c1 = assemble(c)
    assert round(b1 - m + 1, 7) == 0
    assert round(c1 + b1, 7) == 0


@skip_in_parallel
def test_normals_3D_2D(bottom2):
    "Testing assembly of normals for 2D meshes embedded in 3D"

    n = FacetNormal(bottom2)
    a = inner(n, n)*ds
    v1 = assemble(a)
    assert round(v1 - 4.0, 7) == 0

    b = inner(n('+'), n('+'))*dS
    b1 = assemble(b)
    c = inner(n('+'), n('-'))*dS
    c1 = assemble(c)
    assert round(c1 + b1, 7) == 0


@skip_in_parallel
def test_cell_volume(m, bottom1, bottom2, bottom3):
    "Testing assembly of volume for embedded meshes"

    volume = CellVolume(bottom1)
    a = volume*dx
    b = assemble(a)
    assert round(b - 1.0/m, 7) == 0

    volume = CellVolume(bottom3)
    a = volume*dx
    b = assemble(a)
    assert round(b - 1.0/m, 7) == 0

    volume = CellVolume(bottom2)
    a = volume*dx
    b = assemble(a)
    assert round(b - 1.0/(2*m*m), 7) == 0


@skip_in_parallel
def test_circumradius(m, bottom1, bottom2, bottom3):
    "Testing assembly of circumradius for embedded meshes"

    r = Circumradius(bottom1)
    a = r*dx
    b = assemble(a)
    assert round(b - 0.5*(1.0/m), 7) == 0

    r = Circumradius(bottom3)
    a = r*dx
    b = assemble(a)
    assert round(b - 0.5*(1.0/m), 7) == 0

    square = UnitSquareMesh(m, m)
    r = Circumradius(square)
    a = r*dx
    b0 = assemble(a)

    r = Circumradius(bottom2)
    a = r*dx
    b1 = assemble(a)
    assert round(b0 - b1, 7) == 0


@skip_in_parallel
def test_facetarea(bottom1, bottom2, bottom3, m):
    "Testing assembly of facet area for embedded meshes"

    area = FacetArea(bottom1)
    a = area*ds
    b = assemble(a)
    assert round(b - 2.0, 7) == 0

    area = FacetArea(bottom3)
    a = area*ds
    b = assemble(a)
    assert round(b - 2.0, 7) == 0

    square = UnitSquareMesh(m, m)
    area = FacetArea(square)
    a = area*ds
    b0 = assemble(a)

    area = FacetArea(bottom2)
    a = area*ds
    b1 = assemble(a)
    assert round(b0 - b1, 7) == 0


@skip_in_parallel
def test_derivative(QQ2, QQ3):
    for W in [QQ2, QQ3]:
        w = Function(W)
        dim = w.value_dimension(0)
        w.interpolate(Constant([42.0*(i+1) for i in range(dim)]))

        # Derivative w.r.t. mixed space
        u, v = split(w)
        F = u*v*dx
        dF = derivative(F, w)
        b1 = assemble(dF)


def test_coefficient_derivatives(V1, V2):
    for V in [V1, V2]:
        f = Function(V)
        g = Function(V)
        v = TestFunction(V)
        u = TrialFunction(V)

        f.interpolate(Expression("1.0 + x[0] + x[1]", degree=1))
        g.interpolate(Expression("2.0 + x[0] + x[1]", degree=1))

        # Since g = f + 1, define dg/df = 1
        cd = {g: 1}

        # Handle relation between g and f in derivative
        M = g**2*dx
        L = derivative(M, f, v, coefficient_derivatives=cd)
        a = derivative(L, f, u, coefficient_derivatives=cd)
        A0 = assemble(a).norm('frobenius')
        b0 = assemble(L).norm('l2')

        # Manually construct the above case
        M = g**2*dx
        L = 2*g*v*dx
        a = 2*u*v*dx
        A1 = assemble(a).norm('frobenius')
        b1 = assemble(L).norm('l2')

        # Compare
        assert round(A0 - A1, 7) == 0.0
        assert round(b0 - b1, 7) == 0.0
