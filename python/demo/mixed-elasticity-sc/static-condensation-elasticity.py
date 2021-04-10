#
# .. _demo_static_condensation:
#
# Static condensation of linear elasticity
# ========================================
# Copyright (C) 2020  Michal Habera and Andreas Zilian
#
# This demo solves a Cook's plane stress elasticity test in a mixed space
# formulation. The test is a sloped cantilever under upward traction force
# at free end. Static condensation of internal (stress) degrees-of-freedom
# is demonstrated. ::

import os

import cffi
import dolfinx
import dolfinx.cpp
import dolfinx.geometry
import dolfinx.io
import dolfinx.la
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy
import ufl
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc

filedir = os.path.dirname(__file__)
infile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                             os.path.join(filedir, "cooks_tri_mesh.xdmf"),
                             "r",
                             encoding=dolfinx.cpp.io.XDMFFile.Encoding.ASCII)
mesh = infile.read_mesh(name="Grid")
infile.close()

# Stress (Se) and displacement (Ue) elements
Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
Ue = ufl.VectorElement("CG", mesh.ufl_cell(), 2)

S = dolfinx.FunctionSpace(mesh, Se)
U = dolfinx.FunctionSpace(mesh, Ue)

# Get local dofmap sizes for later local tensor tabulations
Ssize = S.dolfin_element().space_dimension()
Usize = U.dolfin_element().space_dimension()

sigma, tau = ufl.TrialFunction(S), ufl.TestFunction(S)
u, v = ufl.TrialFunction(U), ufl.TestFunction(U)


def free_end(x):
    """Marks the leftmost points of the cantilever"""
    return numpy.isclose(x[0], 48.0)


def left(x):
    """Marks left part of boundary, where cantilever is attached to wall"""
    return numpy.isclose(x[0], 0.0)


# Locate all facets at the free end and assign them value 1
free_end_facets = locate_entities_boundary(mesh, 1, free_end)
mt = dolfinx.mesh.MeshTags(mesh, 1, free_end_facets, 1)

ds = ufl.Measure("ds", subdomain_data=mt)

# Homogeneous boundary condition in displacement
u_bc = dolfinx.Function(U)
with u_bc.vector.localForm() as loc:
    loc.set(0.0)

# Displacement BC is applied to the left side
left_facets = locate_entities_boundary(mesh, 1, left)
bdofs = locate_dofs_topological(U, 1, left_facets)
bc = dolfinx.fem.DirichletBC(u_bc, bdofs)

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
ufc_form00 = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a00)
kernel00 = ufc_form00.create_cell_integral(-1).tabulate_tensor

ufc_form01 = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a01)
kernel01 = ufc_form01.create_cell_integral(-1).tabulate_tensor

ufc_form10 = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a10)
kernel10 = ufc_form10.create_cell_integral(-1).tabulate_tensor

ffi = cffi.FFI()

cffi_support.register_type(ffi.typeof('double _Complex'),
                           numba.types.complex128)

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8), numba.types.uint32)


@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL,
                                cell_permutation_info=0):
    # Prepare target condensed local elem tensor
    A = numba.carray(A_, (Usize, Usize), dtype=PETSc.ScalarType)

    # Tabulate all sub blocks locally
    A00 = numpy.zeros((Ssize, Ssize), dtype=PETSc.ScalarType)
    kernel00(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index, permutation,
             cell_permutation_info)

    A01 = numpy.zeros((Ssize, Usize), dtype=PETSc.ScalarType)
    kernel01(ffi.from_buffer(A01), w_, c_, coords_, entity_local_index, permutation,
             cell_permutation_info)

    A10 = numpy.zeros((Usize, Ssize), dtype=PETSc.ScalarType)
    kernel10(ffi.from_buffer(A10), w_, c_, coords_, entity_local_index, permutation,
             cell_permutation_info)

    # A = - A10 * A00^{-1} * A01
    A[:, :] = - A10 @ numpy.linalg.solve(A00, A01)


# Prepare a Form with a condensed tabulation kernel
integrals = {dolfinx.fem.IntegralType.cell: ([(-1, tabulate_condensed_tensor_A.address)], None)}
a_cond = dolfinx.cpp.fem.Form([U._cpp_object, U._cpp_object], integrals, [], [], False, None)

A_cond = dolfinx.fem.assemble_matrix(a_cond, [bc])
A_cond.assemble()

b = dolfinx.fem.assemble_vector(b1)
dolfinx.fem.apply_lifting(b, [a_cond], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, [bc])

uc = dolfinx.Function(U)
solver = PETSc.KSP().create(A_cond.getComm())
solver.setOperators(A_cond)
solver.solve(b, uc.vector)

# Pure displacement based formulation
a = - ufl.inner(sigma_u(u), ufl.grad(v)) * ufl.dx
A = dolfinx.fem.assemble_matrix(a, [bc])
A.assemble()

# Create bounding box for function evaluation
bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, 2)

# Check against standard table value
p = numpy.array([48.0, 52.0, 0.0], dtype=numpy.float64)
cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, p)
cell = dolfinx.cpp.geometry.select_colliding_cells(mesh, cell_candidates, p, 1)

uc.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
if len(cell) > 0:
    value = uc.eval(p, cell)
    print(value[1])
    assert numpy.isclose(value[1], 23.95, rtol=1.e-2)

# Check the equality of displacement based and mixed condensed global
# matrices, i.e. check that condensation is exact
assert numpy.isclose((A - A_cond).norm(), 0.0)
