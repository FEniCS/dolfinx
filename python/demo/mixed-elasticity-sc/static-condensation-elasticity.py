# Copyright (C) 2019 Michal Habera and Andreas Zilian
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
import numba
import numba.cffi_support
import numpy
from petsc4py import PETSc

import dolfin
import dolfin.cpp
import dolfin.io
import dolfin.la
import ufl

filedir = os.path.dirname(__file__)
infile = dolfin.io.XDMFFile(dolfin.MPI.comm_world,
                            os.path.join(filedir, "cooks_tri_mesh.xdmf"),
                            encoding=dolfin.cpp.io.XDMFFile.Encoding.ASCII)
mesh = infile.read_mesh(dolfin.cpp.mesh.GhostMode.none)
infile.close()

# Stress (Se) and displacement (Ue) elements
Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
Ue = ufl.VectorElement("CG", mesh.ufl_cell(), 2)

S = dolfin.FunctionSpace(mesh, Se)
U = dolfin.FunctionSpace(mesh, Ue)

# Get local dofmap sizes for later local tensor tabulations
Ssize = S.dolfin_element().space_dimension()
Usize = U.dolfin_element().space_dimension()

sigma, tau = dolfin.TrialFunction(S), dolfin.TestFunction(S)
u, v = dolfin.TrialFunction(U), dolfin.TestFunction(U)

# Homogeneous boundary condition in displacement
u_bc = dolfin.Function(U)
with u_bc.vector.localForm() as loc:
    loc.set(0.0)

# Displacement BC is applied to the right side
bc = dolfin.fem.DirichletBC(U, u_bc, lambda x: numpy.isclose(x[:, 0], 0.0))


def free_end(x):
    """Marks the leftmost points of the cantilever"""
    return numpy.isclose(x[:, 0], 48.0)


# Mark free end facets as 1
mf = dolfin.mesh.MeshFunction("size_t", mesh, 1, 0)
mf.mark(free_end, 1)

ds = ufl.Measure("ds", subdomain_data=mf)

# Elastic stiffness tensor and Poisson ratio
E, nu = 1.0, 1.0 / 3.0


def sigma_u(u):
    """Consitutive relation for stress-strain. Assuming plane-stress in XY"""
    eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    sigma = E / (1. - nu ** 2) * ((1. - nu) * eps + nu * ufl.Identity(2) * ufl.tr(eps))
    return sigma


a00 = ufl.inner(sigma, tau) * ufl.dx
a10 = - ufl.inner(sigma, ufl.grad(v)) * ufl.dx
a01 = - ufl.inner(sigma_u(u), tau) * ufl.dx

f = ufl.as_vector([0.0, 1.0 / 16])
b1 = - ufl.inner(f, v) * ds(1)

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
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.int32))


@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, c_, coords_, entity_local_index, cell_orientation):
    # Prepare target condensed local elem tensor
    A = numba.carray(A_, (Usize, Usize), dtype=PETSc.ScalarType)

    # Tabulate all sub blocks locally
    A00 = numpy.zeros((Ssize, Ssize), dtype=PETSc.ScalarType)
    kernel00(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index, cell_orientation)

    A01 = numpy.zeros((Ssize, Usize), dtype=PETSc.ScalarType)
    kernel01(ffi.from_buffer(A01), w_, c_, coords_, entity_local_index, cell_orientation)

    A10 = numpy.zeros((Usize, Ssize), dtype=PETSc.ScalarType)
    kernel10(ffi.from_buffer(A10), w_, c_, coords_, entity_local_index, cell_orientation)

    # A = - A10 * A00^{-1} * A01
    A[:, :] = - A10 @ numpy.linalg.solve(A00, A01)


# Prepare an empty Form and set the condensed tabulation kernel
a_cond = dolfin.cpp.fem.Form([U._cpp_object, U._cpp_object])
a_cond.set_tabulate_tensor(dolfin.fem.FormIntegrals.Type.cell, -1, tabulate_condensed_tensor_A.address)

A_cond = dolfin.fem.assemble_matrix(a_cond, [bc])
A_cond.assemble()

b = dolfin.fem.assemble_vector(b1)
dolfin.fem.apply_lifting(b, [a_cond], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfin.fem.set_bc(b, [bc])

uc = dolfin.Function(U)
dolfin.la.solve(A_cond, uc.vector, b)

with dolfin.io.XDMFFile(dolfin.MPI.comm_world, "uc.xdmf") as outfile:
    outfile.write_checkpoint(uc, "uc")

# Pure displacement based formulation
a = - ufl.inner(sigma_u(u), ufl.grad(v)) * ufl.dx
A = dolfin.fem.assemble_matrix(a, [bc])
A.assemble()

# Create bounding box for function evaluation
bb_tree = dolfin.cpp.geometry.BoundingBoxTree(mesh, 2)

# Check against standard table value
p = [48.0, 52.0, 0.0]
cell = dolfin.cpp.geometry.compute_first_collision(bb_tree, p)
if cell >= 0:
    value = uc.eval(p, numpy.asarray(cell))
    assert numpy.isclose(value[1], 23.95, rtol=1.e-2)

# Check the equality of displacement based and mixed condensed global
# matrices, i.e. check that condensation is exact
assert numpy.isclose((A - A_cond).norm(), 0.0)
