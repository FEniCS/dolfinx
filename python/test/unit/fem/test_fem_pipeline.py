# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import time

import numpy as np
import pytest
from petsc4py import PETSc

import ufl
from dolfinx import MPI, DirichletBC, Function, FunctionSpace, fem
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner)


@pytest.mark.parametrize("filename", [
    "UnitCubeMesh_hexahedron.xdmf",
    "UnitCubeMesh_tetra.xdmf",
    "UnitSquareMesh_quad.xdmf",
    "UnitSquareMesh_triangle.xdmf"
])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_manufactured_poisson(degree, filename, datadir):
    """ Manufactured Poisson problem, solving u = x[1]**p, where p is the
    degree of the Lagrange function space.

    """

    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    V = FunctionSpace(mesh, ("Lagrange", degree))
    u, v = TrialFunction(V), TestFunction(V)

    # Get quadrature degree for bilinear form integrand (ignores effect
    # of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(a)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1]**degree
    f = - div(grad(u_exact))

    # Set quadrature degree for linear form integrand (ignores effect of
    # non-affine map)
    L = inner(f, v) * dx(metadata={"quadrature_degree": -1})
    L.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(L)

    t0 = time.time()
    L = fem.Form(L)
    t1 = time.time()
    print("Linear form compile time:", t1 - t0)

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[1]**degree)

    # Create Dirichlet boundary condition
    mesh.create_connectivity_all()
    facetdim = mesh.topology.dim - 1
    bndry_facets = np.where(np.array(
        mesh.topology.on_boundary(facetdim)) == 1)[0]
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    assert(len(bdofs) < V.dim())
    bc = DirichletBC(u_bc, bdofs)

    t0 = time.time()
    b = assemble_vector(L)
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    t1 = time.time()
    print("Vector assembly time:", t1 - t0)

    t0 = time.time()
    a = fem.Form(a)
    t1 = time.time()
    print("Bilinear form compile time:", t1 - t0)

    t0 = time.time()
    A = assemble_matrix(a, [bc])
    A.assemble()
    t1 = time.time()
    print("Matrix assembly time:", t1 - t0)

    # Create LU linear solver
    solver = PETSc.KSP().create(MPI.comm_world)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    # Solve
    t0 = time.time()
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    t1 = time.time()
    print("Linear solver time:", t1 - t0)

    M = (u_exact - uh)**2 * dx
    t0 = time.time()
    M = fem.Form(M)
    t1 = time.time()
    print("Error functional compile time:", t1 - t0)

    t0 = time.time()
    error = assemble_scalar(M)
    error = MPI.sum(mesh.mpi_comm(), error)
    t1 = time.time()

    print("Error assembly time:", t1 - t0)
    assert np.absolute(error) < 1.0e-14
