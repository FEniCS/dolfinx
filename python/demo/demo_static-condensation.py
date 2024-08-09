# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# (demo-static-condensation)=
#
# # Static condensation of linear elasticity
#
# Copyright (C) 2020  Michal Habera and Andreas Zilian
#
# This demo solves a Cook's plane stress elasticity test in a mixed
# space formulation. The test is a sloped cantilever under upward
# traction force at free end. Static condensation of internal (stress)
# degrees-of-freedom is demonstrated.

# +
from pathlib import Path

try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.common.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)


from mpi4py import MPI

import cffi
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np

import ufl
from basix.ufl import element
from dolfinx import geometry
from dolfinx.fem import (
    Form,
    Function,
    IntegralType,
    dirichletbc,
    form,
    form_cpp_class,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.io import XDMFFile
from dolfinx.jit import ffcx_jit
from dolfinx.mesh import locate_entities_boundary, meshtags
from ffcx.codegeneration.utils import numba_ufcx_kernel_signature as ufcx_signature

if np.issubdtype(PETSc.RealType, np.float32):  # type: ignore
    print("float32 not yet supported for this demo.")
    exit(0)


infile = XDMFFile(
    MPI.COMM_WORLD,
    Path(Path(__file__).parent, "data", "cooks_tri_mesh.xdmf"),
    "r",
    encoding=XDMFFile.Encoding.ASCII,
)
msh = infile.read_mesh(name="Grid")
infile.close()

# Stress (Se) and displacement (Ue) elements
Se = element("DG", msh.basix_cell(), 1, shape=(2, 2), symmetry=True, dtype=PETSc.RealType)  # type: ignore
Ue = element("Lagrange", msh.basix_cell(), 2, shape=(2,), dtype=PETSc.RealType)  # type: ignore

S = functionspace(msh, Se)
U = functionspace(msh, Ue)


sigma, tau = ufl.TrialFunction(S), ufl.TestFunction(S)
u, v = ufl.TrialFunction(U), ufl.TestFunction(U)

# Locate all facets at the free end and assign them value 1. Sort the
# facet indices (requirement for constructing MeshTags)
free_end_facets = np.sort(locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], 48.0)))
mt = meshtags(msh, 1, free_end_facets, 1)

ds = ufl.Measure("ds", subdomain_data=mt)

# Homogeneous boundary condition in displacement
u_bc = Function(U)
u_bc.x.array[:] = 0

# Displacement BC is applied to the left side
left_facets = locate_entities_boundary(msh, 1, lambda x: np.isclose(x[0], 0.0))
bdofs = locate_dofs_topological(U, 1, left_facets)
bc = dirichletbc(u_bc, bdofs)

# Elastic stiffness tensor and Poisson ratio
E, nu = 1.0, 1.0 / 3.0


def sigma_u(u):
    """Constitutive relation for stress-strain. Assuming plane-stress in XY"""
    eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    sigma = E / (1.0 - nu**2) * ((1.0 - nu) * eps + nu * ufl.Identity(2) * ufl.tr(eps))
    return sigma


a00 = ufl.inner(sigma, tau) * ufl.dx
a10 = -ufl.inner(sigma, ufl.grad(v)) * ufl.dx
a01 = -ufl.inner(sigma_u(u), tau) * ufl.dx

f = ufl.as_vector([0.0, 1.0 / 16])
b1 = form(-ufl.inner(f, v) * ds(1), dtype=PETSc.ScalarType)  # type: ignore

# JIT compile individual blocks tabulation kernels
ufcx00, _, _ = ffcx_jit(msh.comm, a00, form_compiler_options={"scalar_type": PETSc.ScalarType})  # type: ignore
kernel00 = getattr(ufcx00.form_integrals[0], f"tabulate_tensor_{np.dtype(PETSc.ScalarType).name}")  # type: ignore

ufcx01, _, _ = ffcx_jit(msh.comm, a01, form_compiler_options={"scalar_type": PETSc.ScalarType})  # type: ignore
kernel01 = getattr(ufcx01.form_integrals[0], f"tabulate_tensor_{np.dtype(PETSc.ScalarType).name}")  # type: ignore

ufcx10, _, _ = ffcx_jit(msh.comm, a10, form_compiler_options={"scalar_type": PETSc.ScalarType})  # type: ignore
kernel10 = getattr(ufcx10.form_integrals[0], f"tabulate_tensor_{np.dtype(PETSc.ScalarType).name}")  # type: ignore

ffi = cffi.FFI()
cffi_support.register_type(ffi.typeof("double _Complex"), numba.types.complex128)

# Get local dofmap sizes for later local tensor tabulations
Ssize = S.element.space_dimension
Usize = U.element.space_dimension


@numba.cfunc(ufcx_signature(PETSc.ScalarType, PETSc.RealType), nopython=True)  # type: ignore
def tabulate_A(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    """Element kernel that applies static condensation."""

    # Prepare target condensed local element tensor
    A = numba.carray(A_, (Usize, Usize), dtype=PETSc.ScalarType)

    # Tabulate all sub blocks locally
    A00 = np.zeros((Ssize, Ssize), dtype=PETSc.ScalarType)
    kernel00(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index, permutation)

    A01 = np.zeros((Ssize, Usize), dtype=PETSc.ScalarType)
    kernel01(ffi.from_buffer(A01), w_, c_, coords_, entity_local_index, permutation)

    A10 = np.zeros((Usize, Ssize), dtype=PETSc.ScalarType)
    kernel10(ffi.from_buffer(A10), w_, c_, coords_, entity_local_index, permutation)

    # A = - A10 * A00^{-1} * A01
    A[:, :] = -A10 @ np.linalg.solve(A00, A01)


# Prepare a Form with a condensed tabulation kernel
formtype = form_cpp_class(PETSc.ScalarType)  # type: ignore
cells = np.arange(msh.topology.index_map(msh.topology.dim).size_local)
integrals = {IntegralType.cell: [(-1, tabulate_A.address, cells, np.array([], dtype=np.int8))]}
a_cond = Form(formtype([U._cpp_object, U._cpp_object], integrals, [], [], False, {}, None))

A_cond = assemble_matrix(a_cond, bcs=[bc])
A_cond.assemble()
b = assemble_vector(b1)
apply_lifting(b, [a_cond], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
set_bc(b, [bc])

uc = Function(U)
solver = PETSc.KSP().create(A_cond.getComm())  # type: ignore
solver.setOperators(A_cond)
solver.solve(b, uc.x.petsc_vec)

# Pure displacement based formulation
a = form(-ufl.inner(sigma_u(u), ufl.grad(v)) * ufl.dx)
A = assemble_matrix(a, bcs=[bc])
A.assemble()

# Create bounding box for function evaluation
bb_tree = geometry.bb_tree(msh, 2)

# Check against standard table value
p = np.array([[48.0, 52.0, 0.0]], dtype=np.float64)
cell_candidates = geometry.compute_collisions_points(bb_tree, p)
cells = geometry.compute_colliding_cells(msh, cell_candidates, p).array

uc.x.scatter_forward()
if len(cells) > 0:
    value = uc.eval(p, cells[0])
    print(value[1])
    assert np.isclose(value[1], 23.95, rtol=1.0e-2)

# Check the equality of displacement based and mixed condensed global
# matrices, i.e. check that condensation is exact
assert np.isclose((A - A_cond).norm(), 0.0)
