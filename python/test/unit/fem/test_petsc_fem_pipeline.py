# Copyright (C) 2021 Jorgen Dokken, Jack S. Hale, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the FEM pipeline requiring PETSc/SLEPc."""

from mpi4py import MPI

import numpy as np
import pytest

import basix
import dolfinx
import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices
from ufl import dx, inner


@pytest.mark.petsc4py
@pytest.mark.parametrize("family", ["N1curl", "N2curl"])
@pytest.mark.parametrize("order", [1])
def test_petsc_curl_curl_eigenvalue(family, order):
    """curl-curl eigenvalue problem.

    Solved using H(curl)-conforming finite element method.
    See https://www-users.cse.umn.edu/~arnold/papers/icm2002.pdf for details.
    """
    if not dolfinx.cpp.common.has_petsc:
        return

    petsc4py = pytest.importorskip("petsc4py")  # noqa: F841
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix

    slepc4py = pytest.importorskip("slepc4py")  # noqa: F841
    from slepc4py import SLEPc

    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([np.pi, np.pi])],
        [24, 24],
        CellType.triangle,
        dtype=default_real_type,
    )

    e = element(family, basix.CellType.triangle, order, dtype=default_real_type)
    V = functionspace(mesh, e)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(ufl.curl(u), ufl.curl(v)) * dx
    b = inner(u, v) * dx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(mesh.topology)
    boundary_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    zero_u = Function(V, dtype=dolfinx.default_scalar_type)
    zero_u.x.array[:] = 0
    bcs = [dirichletbc(zero_u, boundary_dofs)]

    a, b = form(a), form(b)
    A = petsc_assemble_matrix(a, bcs=bcs)
    A.assemble()
    B = petsc_assemble_matrix(b, bcs=bcs, diag=0.01)
    B.assemble()

    eps = SLEPc.EPS().create()
    eps.setOperators(A, B)
    PETSc.Options()["eps_type"] = "krylovschur"
    PETSc.Options()["eps_gen_hermitian"] = ""
    PETSc.Options()["eps_target_magnitude"] = ""
    PETSc.Options()["eps_target"] = 5.0
    PETSc.Options()["eps_view"] = ""
    PETSc.Options()["eps_nev"] = 12
    eps.setFromOptions()
    eps.solve()

    num_converged = eps.getConverged()
    evlas_unsorted = np.zeros(num_converged, dtype=np.complex128)

    for i in range(0, num_converged):
        evlas_unsorted[i] = eps.getEigenvalue(i)

    assert np.isclose(np.imag(evlas_unsorted), 0.0).all()
    evals_sorted = np.sort(np.real(evlas_unsorted))[:-1]
    evals_sorted = evals_sorted[np.logical_not(evals_sorted < 1e-8)]

    evals_exact = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0, 8.0, 9.0])
    assert np.isclose(evals_sorted[0 : evals_exact.shape[0]], evals_exact, rtol=1e-2).all()

    eps.destroy()
    A.destroy()
    B.destroy()
