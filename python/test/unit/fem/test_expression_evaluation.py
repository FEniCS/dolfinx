# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import cffi
import numba
import numpy as np
from petsc4py import PETSc

import ufl
import dolfin


def test_rank0():
    """Test evaluation of UFL expression.

    This test evaluates gradient of P2 function at vertices of reference triangle.
    Because these points coincide with positions of point evaluation degrees-of-freedom
    of vector P1 space, values could be used to interpolate the expression into this space.

    This test also shows simple Numba assembler which accepts the donor P2 function ``f``
    as a coefficient and tabulates vector P1 function into tensor ``b``.

    For a donor function f(x, y) = x^2 + 2*y^2 result is compared with the exact
    gradient grad f(x, y) = [2*x, 4*y].

    """
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)
    P2 = dolfin.FunctionSpace(mesh, ("P", 2))
    vP1 = dolfin.VectorFunctionSpace(mesh, ("P", 1))

    f = dolfin.Function(P2)

    def expr1(x):
        return x[:, 0] ** 2 + 2.0 * x[:, 1] ** 2

    f.interpolate(expr1)

    ufl_expr = ufl.grad(f)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    compiled_expr = dolfin.jit.ffc_jit((ufl_expr, points))

    ffi = cffi.FFI()

    @numba.njit
    def assemble_expression(b, kernel, mesh, x, dofmap, coeff, coeff_dofmap):
        connections, pos = mesh
        geometry = np.zeros((3, 2))
        w = np.zeros(6, dtype=PETSc.ScalarType)
        constants = np.zeros(1, dtype=PETSc.ScalarType)
        b_local = np.zeros(6, dtype=PETSc.ScalarType)

        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            for j in range(3):
                for k in range(2):
                    geometry[j, k] = x[c[j], k]

            for j in range(6):
                w[j] = coeff[coeff_dofmap[i * 6 + j]]

            b_local.fill(0.0)
            kernel(ffi.from_buffer(b_local),
                   ffi.from_buffer(w),
                   ffi.from_buffer(constants),
                   ffi.from_buffer(geometry))
            for j in range(6):
                b[dofmap[i * 6 + j]] = b_local[j]

    # Prepare mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    coeff_dofmap = P2.dofmap.dof_array
    dofmap = vP1.dofmap.dof_array

    # Data structure for the result
    b = dolfin.Function(vP1)

    assemble_expression(b.vector.array, compiled_expr.tabulate_expression,
                        (c, pos), geom, dofmap, f.vector.array, coeff_dofmap)

    def grad_expr1(x):
        values = np.empty((x.shape[0], 2))

        values[:, 0] = 2.0 * x[:, 0]
        values[:, 1] = 4.0 * x[:, 1]

        return values

    b2 = dolfin.Function(vP1)
    b2.interpolate(grad_expr1)

    assert np.isclose((b2.vector - b.vector).norm(), 0.0)
