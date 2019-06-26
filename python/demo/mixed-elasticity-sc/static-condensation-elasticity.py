# Copyright (C) 2019 Michal Habera, Andreas Zilian
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# This demo solves a Cook's plane stress elasticity test in a mixed space
# formulation. The test is a sloped cantilever under upward traction force
# at free end. Static condensation of internal (stress) degrees-of-freedom
# is demonstrated.

import os

import cffi
import numpy
import numba
import numba.cffi_support
from petsc4py import PETSc

import dolfin
import dolfin.cpp
import dolfin.io
import ufl


filedir = os.path.dirname(__file__)
infile = dolfin.io.XDMFFile(dolfin.MPI.comm_world,
                            os.path.join(filedir, "cooks_tri_mesh.xdmf"))
mesh = infile.read_mesh(dolfin.MPI.comm_world, dolfin.cpp.mesh.GhostMode.none)
infile.close()

# Se = stress element
# Ue = displacement element
Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
Ue = ufl.VectorElement("CG", mesh.ufl_cell(), 2)

S = dolfin.FunctionSpace(mesh, Se)
U = dolfin.FunctionSpace(mesh, Ue)

# Get local dofmap sizes for later local tensor tabulations
Ssize = len(S.dofmap().cell_dofs(0))
Usize = len(U.dofmap().cell_dofs(0))

sigma, sigmat = dolfin.TrialFunction(S), dolfin.TestFunction(S)
u, ut = dolfin.TrialFunction(U), dolfin.TestFunction(U)

# Homogeneous boundary condition in displacement
u_bc = dolfin.Function(U)
with u_bc.vector().localForm() as loc:
    loc.set(0.0)

# Displacement BC is applied to the right side
bc = dolfin.fem.DirichletBC(U, u_bc, lambda x, only_bndry: numpy.isclose(x[:, 0], 0.0))


def free_end(x):
    """Marks the leftmost points of the cantilever"""
    return numpy.isclose(x[:, 0], 48.0)


# Mark free end facets as 1
mf = dolfin.mesh.MeshFunction("size_t", mesh, 1, 0)
mf.mark(free_end, 1)

ds = ufl.Measure("ds", subdomain_data=mf)
_i, _j, _k, _l = ufl.indices(4)

# Elastic stiffness tensor
E = 1.0
# Poisson ratio
nu = 1.0 / 3
# First Lame coeff
lambda_ = E * nu / ((1. + nu) * (1. - 2 * nu))
# Shear modulus
mu = E / (2. * (1. + nu))


def sigma_u(u):
    """Consitutive relation for stress-strain. Assuming plane-stress in XY"""
    eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    sigma = E / (1. - nu ** 2) * ((1. - nu) * eps + nu * ufl.Identity(2) * ufl.tr(eps))
    return sigma


a00 = ufl.inner(sigma, sigmat) * ufl.dx
a10 = - ufl.inner(sigma, ufl.grad(ut)) * ufl.dx
a01 = - ufl.inner(sigma_u(u), sigmat) * ufl.dx

f = ufl.as_vector([0.0, 1.0 / 16])
b1 = - ufl.inner(f, ut) * ds(1)

# JIT compile individual blocks tabulation kernels
ufc_form00 = dolfin.jit.ffc_jit(a00)
kernel00 = ufc_form00.create_cell_integral(-1).tabulate_tensor

ufc_form01 = dolfin.jit.ffc_jit(a01)
kernel01 = ufc_form01.create_cell_integral(-1).tabulate_tensor

ufc_form10 = dolfin.jit.ffc_jit(a10)
kernel10 = ufc_form10.create_cell_integral(-1).tabulate_tensor

ffi = cffi.FFI()

numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                 numba.types.complex128)

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.int32))


@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, coords_, entity_local_index, cell_orientation):
    # Prepare target condensed local elem tensor
    A = numba.carray(A_, (Usize, Usize), dtype=PETSc.ScalarType)

    # Tabulate all sub blocks locally
    A00 = numpy.zeros((Ssize, Ssize), dtype=PETSc.ScalarType)
    kernel00(ffi.from_buffer(A00), w_, coords_, entity_local_index, cell_orientation)

    A01 = numpy.zeros((Ssize, Usize), dtype=PETSc.ScalarType)
    kernel01(ffi.from_buffer(A01), w_, coords_, entity_local_index, cell_orientation)

    A10 = numpy.zeros((Usize, Ssize), dtype=PETSc.ScalarType)
    kernel10(ffi.from_buffer(A10), w_, coords_, entity_local_index, cell_orientation)

    # A = - A10 * A00^{-1} * A01
    A00inv = numpy.linalg.inv(A00)
    A[:, :] = - numpy.dot(A10, numpy.dot(A00inv, A01))


# Prepare an empty Form and set the condensed tabulation kernel
a_cond = dolfin.cpp.fem.Form([U._cpp_object, U._cpp_object])
a_cond.set_tabulate_cell(-1, tabulate_condensed_tensor_A.address)

A_cond = dolfin.fem.assemble_matrix(a_cond, [bc])
A_cond.assemble()

b = dolfin.fem.assemble_vector(b1)
dolfin.fem.apply_lifting(b, [a_cond], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfin.fem.set_bc(b, [bc])

uc = dolfin.Function(U)
dolfin.la.solve(A_cond, uc.vector(), b)

with dolfin.io.XDMFFile(dolfin.MPI.comm_world, "uc.xdmf") as outfile:
    outfile.write_checkpoint(uc, "uc")

# Pure displacement based formulation
a = - ufl.inner(sigma_u(u), ufl.grad(ut)) * ufl.dx
A = dolfin.fem.assemble_matrix(a, [bc])
A.assemble()

# Extract bounding box for function evaluation
bb_tree = dolfin.cpp.geometry.BoundingBoxTree(mesh, 2)

# Evaluate at free end midpoint
# Keep False value in non-owning processes
uc_midpoint = False
try:
    uc_midpoint = uc([48.0, 52.0], bb_tree)
except RuntimeError:
    pass

# Check against standart table value
if uc_midpoint is not False:
    assert(numpy.isclose(uc_midpoint[1], 23.95, rtol=1.e-2))

# Check the equality of displacement based and mixed condensed
# global matrices, i.e. check that condensation is exact
assert(numpy.isclose((A - A_cond).norm(), 0.0))
