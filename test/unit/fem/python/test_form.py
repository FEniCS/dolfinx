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

square_ = UnitSquareMesh(2, 2)
square_boundary_ = BoundaryMesh(square_, "exterior")
cube_ = UnitCubeMesh(2, 2, 2)
cube_boundary_ = BoundaryMesh(cube_, "exterior")

skip_parallel = pytest.mark.skipif(MPI.size(mpi_comm_world()) > 1, 
                             reason="Skipping unit test(s) not working in parallel")

@pytest.fixture
def square_boundary():
    return square_boundary_

@pytest.fixture
def cube_boundary():
    return cube_boundary_

@pytest.fixture
def square():
    return square_

@pytest.fixture
def cube():
    return cube_

@pytest.fixture
def V1():
    return FunctionSpace(square_boundary_, "CG", 1) 

@pytest.fixture
def VV1():   
    return VectorFunctionSpace(square_boundary_, "CG", 1)

@pytest.fixture
def Q1():   
    return FunctionSpace(square_boundary_, "DG", 0)

@pytest.fixture
def V2():   
    return FunctionSpace(cube_boundary_, "CG", 1)      

@pytest.fixture
def VV2():
    return VectorFunctionSpace(cube_boundary_, "CG", 1)

@pytest.fixture
def Q2():   
    return FunctionSpace(cube_boundary_, "DG", 0)


def test_assemble_functional(V1, V2):

    u = Function(V1)
    u.vector()[:] = 1.0
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 4.0, 7) == 0

    u = Function(V2)
    u.vector()[:] = 1.0
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 6.0, 7) == 0

    f = Expression("1.0")
    u = interpolate(f, V1)
    surfacearea = assemble(u*dx)
    assert round(surfacearea - 4.0, 7) == 0

    f = Expression("1.0")
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


@skip_parallel
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
    dss = ds[bottom]
    foo = MPI.sum(square.mpi_comm(),
               abs(assemble(inner(grad(u)[0], grad(v)[0])*dss(1)).array()).sum())
    # Assemble over all cells of submesh created from subset of boundary mesh
    bottom2 = CellFunctionSizet(square_boundary)
    bottom2.set_all(0)
    subdomain.mark(bottom2, 1)
    BV = FunctionSpace(SubMesh(square_boundary, bottom2, 1), "CG", 1)
    bu = TrialFunction(BV)
    bv = TestFunction(BV)
    bar = MPI.sum(square_boundary.mpi_comm(),
                  abs(assemble(inner(grad(bu)[0], grad(bv)[0])*dx).array()).sum())
    # Should give same result
    assert round(bar - foo, 7) == 0


@skip_parallel
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
    dss = ds[bottom]
    foo = MPI.sum(cube.mpi_comm(),
               abs(assemble(inner(grad(u)[0], grad(v)[0])*dss(1)).array()).sum())
    #foo = MPI.sum(cube.mpi_comm(),
    #              abs(assemble(inner(grad(u)[0], grad(v)[0])*ds(1),
    #                           exterior_facet_domains=bottom).array()).sum())
    # Assemble over all cells of submesh created from subset of boundary mesh
    bottom2 = CellFunctionSizet(cube_boundary)
    bottom2.set_all(0)
    subdomain.mark(bottom2, 1)
    BV = FunctionSpace(SubMesh(cube_boundary, bottom2, 1), "CG", 1)
    bu = TrialFunction(BV)
    bv = TestFunction(BV)
    bar = MPI.sum(cube_boundary.mpi_comm(),
                   abs(assemble(inner(grad(bu)[0], grad(bv)[0])*dx).array()).sum())
    # Should give same result
    assert round(bar - foo, 7) == 0


# Set-up meshes
n = 16
plane_ = CompiledSubDomain("near(x[1], 1.0)")
square_ = UnitSquareMesh(n, n)
square3d_ = SubMesh(BoundaryMesh(UnitCubeMesh(n, n, n), "exterior"), plane_)

# Define global normal and create orientation map
global_normal = Expression(("0.0", "1.0", "0.0"))
square3d_.init_cell_orientations(global_normal)

DG2_ = FunctionSpace(square_, "DG", 0)
DG3_ = FunctionSpace(square3d_, "DG", 0)
RT2_ = FunctionSpace(square_, "RT", 1)
RT3_ = FunctionSpace(square3d_, "RT", 1)

@pytest.fixture
def RT2():
    return RT2_

@pytest.fixture
def RT3():
    return RT3_

@pytest.fixture
def W2():
    return RT2_ * DG2_

@pytest.fixture
def W3():
    return RT3_ * DG3_

@skip_parallel
def test_basic_rt(RT2, RT3):

    f2 = Expression(("2.0", "1.0"))
    f3 = Expression(("1.0", "0.0", "2.0"))

    u2 = TrialFunction(RT2)
    u3 = TrialFunction(RT3)
    v2 = TestFunction(RT2)
    v3 = TestFunction(RT3)

    # Project
    pw2 = project(f2, RT2)
    pw3 = project(f3, RT3)
    pa2 = assemble(inner(pw2, pw2)*dx)
    pa3 = assemble(inner(pw3, pw3)*dx)

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
    a2 = assemble(inner(w2, w2)*dx)
    a3 = assemble(inner(w3, w3)*dx)

    # Compare various results
    assert round((w2.vector() - pw2.vector()).norm("l2") - 0.0, 5) == 0
    assert round(a3 - 5.0, 7) == 0
    assert round(a2 - a3, 7) == 0
    assert round(pa2 - a2, 7) == 0
    assert round(pa2 - pa3, 7) == 0


@skip_parallel
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


@pytest.fixture
def m():
    return 3

cube_ = UnitCubeMesh(m(), m(), m())
cube_boundary_ = BoundaryMesh(cube_, "exterior")

plane_ = CompiledSubDomain("near(x[1], 0.0)")
square_ = UnitSquareMesh(m(), m())
square_boundary_ = BoundaryMesh(square_, "exterior")

line_ = CompiledSubDomain("near(x[0], 0.0)")
mesh3_ = BoundaryMesh(SubMesh(cube_boundary_, plane_), "exterior")

@pytest.fixture
def bottom1():
    return SubMesh(square_boundary_, plane_)

@pytest.fixture
def bottom2():
    return SubMesh(cube_boundary_, plane_)

@pytest.fixture
def bottom3():
    return SubMesh(mesh3_, line_)

    
@skip_parallel
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


@skip_parallel
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


@skip_parallel
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


@skip_parallel
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


@skip_parallel
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


@skip_parallel
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


if __name__ == "__main__":
    pytest.main()
