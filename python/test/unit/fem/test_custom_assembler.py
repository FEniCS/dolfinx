# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import math
import time

import cffi
import numpy as np
import pytest
from numba import jit
from petsc4py import PETSc

import dolfin
from ufl import dx, inner


def test_custom_mesh_loop():

    @jit(nopython=True, cache=True)
    def area(x0, x1, x2):
        a = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2
        b = (x0[0] - x2[0])**2 + (x0[1] - x2[1])**2
        c = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2
        return math.sqrt(2 * (a * b + a * c + b * c) - (a**2 + b**2 + c**2)) / 4.0

    @jit(nopython=True, cache=True)
    def assemble_vector(b, mesh, x, dofmap):
        """Assemble over a mesh into the array b"""
        connections, pos = mesh
        q0, q1 = 1 / 3.0, 1 / 3.0
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            A = area(x[c[0]], x[c[1]], x[c[2]])
            b[dofmap[i * 3 + 0]] += A * (1.0 - q0 - q1)
            b[dofmap[i * 3 + 2]] += A * q1
            b[dofmap[i * 3 + 1]] += A * q0

    ffi = cffi.FFI()

    @jit(nopython=True)
    def assemble_vector_ufc(b, kernel, mesh, x, dofmap):
        """Assemble provided kernel over a mesh into the array b"""
        connections, pos = mesh
        b_local = np.zeros(3)
        geometry = np.zeros((3, 2))
        coeffs = np.zeros(0)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            for j in range(3):
                for k in range(2):
                    geometry[j, k] = x[c[j], k]
            kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs), ffi.from_buffer(geometry), 0)
            for j in range(3):
                b[dofmap[i * 3 + j]] += b_local[j]

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 64, 64)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))
    b0 = dolfin.Function(V)

    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    with b0.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector(_b, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba, 1):", end - start)

    with b0.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector(_b, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba, 2):", end - start)

    b0.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b0.vector().sum() == pytest.approx(1.0))

    # Test against generated code and general assembler
    v = dolfin.TestFunction(V)
    L = inner(1.0, v) * dx

    start = time.time()
    b1 = dolfin.fem.assemble_vector(L)
    end = time.time()
    print("Time (C++, 1):", end - start)

    with b1.localForm() as b_local:
        b_local.set(0.0)
    start = time.time()
    dolfin.fem.assemble_vector(b1, L)
    end = time.time()
    print("Time (C++, 2):", end - start)

    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b1.sum() == pytest.approx(1.0))

    b2 = b1 - b0.vector()
    assert(b2.norm() == pytest.approx(0.0))

    # Complex not supported yet
    # cffi_support.register_type('double _Complex', numba.types.complex128)
    if not dolfin.has_petsc_complex:
        return

    # Assemble using generated tabulate_tensor kernel
    b3 = dolfin.Function(V)
    ufc_form = dolfin.jit.ffc_jit(L)
    print("test", ffi.list_types())
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    with b3.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector_ufc(_b, kernel, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba/cffi, 1):", end - start)

    with b3.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector_ufc(_b, kernel, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba/cffi, 2):", end - start)

    b3.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b3.vector().sum() == pytest.approx(1.0))
