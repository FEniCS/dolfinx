"""Unit tests for the fem interface"""

# Copyright (C) 2016 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

from dolfin import (MPI, Function, FunctionSpace, UnitCubeMesh, UnitSquareMesh,
                    VectorFunctionSpace)
from dolfin.cpp.fem import PETScDMCollection
from ufl import FiniteElement, MixedElement, VectorElement


def test_scalar_p1():
    meshc = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    meshf = UnitCubeMesh(MPI.comm_world, 3, 4, 5)

    Vc = FunctionSpace(meshc, ("CG", 1))
    Vf = FunctionSpace(meshf, ("CG", 1))

    def u(x):
        return x[0] + 2.0 * x[1] + 3.0 * x[2]

    uc, uf = Function(Vc), Function(Vf)
    uc.interpolate(u)
    uf.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)
    Vuc = Function(Vf)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)

    assert diff.norm() < 1.0e-12


def test_scalar_p1_scaled_mesh():
    # Make coarse mesh smaller than fine mesh
    meshc = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    meshc.geometry.points *= 0.9

    meshf = UnitCubeMesh(MPI.comm_world, 3, 4, 5)

    Vc = FunctionSpace(meshc, ("CG", 1))
    Vf = FunctionSpace(meshf, ("CG", 1))

    def u(x):
        return x[0] + 2.0 * x[1] + 3.0 * x[2]
    uc, uf = Function(Vc), Function(Vf)
    uc.interpolate(u)
    uf.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)
    Vuc = Function(Vf)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)

    assert diff.norm() < 1.0e-12

    # Now make coarse mesh larger than fine mesh
    meshc.geometry.points *= 1.5
    uc.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)

    assert diff.norm() < 1.0e-12


def test_scalar_p2():
    meshc = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    meshf = UnitCubeMesh(MPI.comm_world, 3, 4, 5)

    Vc = FunctionSpace(meshc, ("CG", 2))
    Vf = FunctionSpace(meshf, ("CG", 2))

    def u(x):
        return x[0] + 2.0 * x[1] + 3.0 * x[2]

    uc, uf = Function(Vc), Function(Vf)
    uc.interpolate(u)
    uf.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)
    Vuc = Function(Vf)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)

    assert diff.norm() < 1.0e-12


def test_vector_p1_2d():
    meshc = UnitSquareMesh(MPI.comm_world, 5, 4)
    meshf = UnitSquareMesh(MPI.comm_world, 7, 8)

    Vc = VectorFunctionSpace(meshc, ("CG", 1))
    Vf = VectorFunctionSpace(meshf, ("CG", 1))

    def u(x):
        return np.stack([x[0] + 2.0 * x[1], 4.0 * x[0]], axis=0)

    uc, uf = Function(Vc), Function(Vf)
    uc.interpolate(u)
    uf.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)

    Vuc = Function(Vf)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)
    assert diff.norm() < 1.0e-12


def test_vector_p2_2d():
    meshc = UnitSquareMesh(MPI.comm_world, 5, 4)
    meshf = UnitSquareMesh(MPI.comm_world, 5, 8)

    Vc = VectorFunctionSpace(meshc, ("CG", 2))
    Vf = VectorFunctionSpace(meshf, ("CG", 2))

    def u(x):
        return np.stack([x[0] + 2.0 * x[1], 4.0 * x[0] * x[1]], axis=0)

    uc, uf = Function(Vc), Function(Vf)
    uc.interpolate(u)
    uf.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)
    Vuc = Function(Vf)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)
    assert diff.norm() < 1.0e-12


def test_vector_p1_3d():
    meshc = UnitCubeMesh(MPI.comm_world, 2, 3, 4)
    meshf = UnitCubeMesh(MPI.comm_world, 3, 4, 5)

    Vc = VectorFunctionSpace(meshc, ("CG", 1))
    Vf = VectorFunctionSpace(meshf, ("CG", 1))

    def u(x):
        values0 = x[0] + 2.0 * x[1]
        values1 = 4.0 * x[0]
        values2 = 3.0 * x[2] + x[0]
        return np.stack([values0, values1, values2], axis=0)

    uc, uf = Function(Vc), Function(Vf)
    uc.interpolate(u)
    uf.interpolate(u)

    mat = PETScDMCollection.create_transfer_matrix(Vc._cpp_object,
                                                   Vf._cpp_object)
    Vuc = Function(Vf)
    mat.mult(uc.vector, Vuc.vector)

    diff = Vuc.vector
    diff.axpy(-1, uf.vector)
    assert diff.norm() < 1.0e-12


@pytest.mark.xfail
def test_taylor_hood_cube():
    pytest.xfail("Problem with Mixed Function Spaces")
    meshc = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    meshf = UnitCubeMesh(MPI.comm_world, 3, 4, 5)

    Ve = VectorElement("CG", meshc.ufl_cell(), 2)
    Qe = FiniteElement("CG", meshc.ufl_cell(), 1)
    Ze = MixedElement([Ve, Qe])

    Zc = FunctionSpace(meshc, Ze)
    Zf = FunctionSpace(meshf, Ze)

    def z(x):
        return np.row_stack((x[0] * x[1],
                             x[1] * x[2],
                             x[2] * x[0],
                             x[0] + 3.0 * x[1] + x[2]))

    zc, zf = Function(Zc), Function(Zf)
    zc.interpolate(z)
    zf.interpolate(z)

    mat = PETScDMCollection.create_transfer_matrix(Zc, Zf)
    Zuc = Function(Zf)
    mat.mult(zc.vector, Zuc.vector)
    Zuc.vector.update_ghost_values()

    diff = Function(Zf)
    diff.assign(Zuc - zf)
    assert diff.vector.norm("l2") < 1.0e-12
