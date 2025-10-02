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
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_static-condensation.py>`
# * {download}`Jupyter notebook <./demo_static-condensation.ipynb>`
# ```
# This demo solves a Cook's plane stress elasticity test in a mixed
# space formulation. The test is a sloped cantilever under upward
# traction force at free end. Static condensation of internal (stress)
# degrees-of-freedom is demonstrated.
#
# This demo illustrates how to:
# - Use static condensation with [numba](https://numba.pydata.org/)
#   on variational forms made with UFL.
# - Extracting JIT compiled C-kernels using {py:func}`ffcx_jit
# <dolfinx.jit.ffcx_jit>`

# This demo requires more modules than usual, as it uses `numba` for
# efficient static condensation.

# +
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import cffi
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np

import ufl
from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type, geometry
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
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.jit import ffcx_jit
from dolfinx.mesh import locate_entities_boundary, meshtags
from ffcx.codegeneration.utils import empty_void_pointer
from ffcx.codegeneration.utils import numba_ufcx_kernel_signature as ufcx_signature

# -

rtype = default_real_type
dtype = default_scalar_type
if np.issubdtype(rtype, np.float32):  # type: ignore
    print("float32 not yet supported for this demo.")
    exit(0)

# We start by reading in the Cook's mesh [cooks_tri_mesh.xdmf](
# https://github.com/FEniCS/dolfinx/blob/main/python/demo/data/cooks_tri_mesh.xdmf)
# using {py:meth}`XDMFFile.read_mesh <dolfinx.io.XDMFFile.read_mesh>`.
# Note that the mesh is written in plain-text format, which means we use
# {py:attr}`XDMFFile.Encoding.ASCII <dolfinx.io.XDMFFile.Encoding.ASCII>`.

infile = XDMFFile(
    MPI.COMM_WORLD,
    Path(Path(__file__).parent, "data", "cooks_tri_mesh.xdmf"),
    "r",
    encoding=XDMFFile.Encoding.ASCII,
)
msh = infile.read_mesh(name="Grid")
infile.close()

# We create the Stress (Se) and displacement (Ue) elements and
# corresponding function spaces. Note that the stress element is symmetric.

gdim = msh.geometry.dim
Se = element("DG", msh.basix_cell(), 1, shape=(gdim, gdim), symmetry=True, dtype=rtype)  # type: ignore
Ue = element("Lagrange", msh.basix_cell(), 2, shape=(gdim,), dtype=rtype)  # type: ignore
S = functionspace(msh, Se)
U = functionspace(msh, Ue)

# Next, we define the trial and test functions for stress and displacement,

sigma, tau = ufl.TrialFunction(S), ufl.TestFunction(S)
u, v = ufl.TrialFunction(U), ufl.TestFunction(U)

# Locate all facets at the free end and assign them value 1. Sort the
# facet indices (requirement for constructing {py:class}`MeshTags
# <dolfinx.mesh.MeshTags>`).

tdim = msh.topology.dim
free_end_facets = np.sort(locate_entities_boundary(msh, tdim - 1, lambda x: np.isclose(x[0], 48.0)))
mt = meshtags(msh, tdim - 1, free_end_facets, 1)

# Next, we create an integration measure with the facet markers.

ds = ufl.Measure("ds", subdomain_data=mt)

# Homogeneous boundary condition in displacement

u_bc = Function(U)
u_bc.x.array[:] = 0

# Displacement {py:class}`BC <dolfinx.fem.dirichletbc>` is applied to
# the left side

left_facets = locate_entities_boundary(msh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
bdofs = locate_dofs_topological(U, tdim - 1, left_facets)
bc = dirichletbc(u_bc, bdofs)

# Elastic stiffness tensor and Poisson ratio

# +
E, nu = 1.0, 1.0 / 3.0


def sigma_u(u):
    """Constitutive relation for stress-strain. Assuming plane-stress in
    XY"""
    eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    sigma = E / (1.0 - nu**2) * ((1.0 - nu) * eps + nu * ufl.Identity(2) * ufl.tr(eps))
    return sigma


# -

# With the definitions above, we can define the different blocks
# of the variational formulation

# +
a00 = ufl.inner(sigma, tau) * ufl.dx
a10 = -ufl.inner(sigma, ufl.grad(v)) * ufl.dx
a01 = -ufl.inner(sigma_u(u), tau) * ufl.dx

f = ufl.as_vector([0.0, 1.0 / 16])
b1 = form(-ufl.inner(f, v) * ds(1), dtype=dtype)  # type: ignore
# -

# To generate (C-code) and JIT compile the kernels, we use
# {py:func}`ffcx_jit <dolfinx.jit.ffcx_jit>` for each individual block.
# We extract the kernel function from the compiled form object by
# getting the `tabulate_tensor_{dtype}` attribute of the compiled form.

# +
ufcx00, _, _ = ffcx_jit(msh.comm, a00, form_compiler_options={"scalar_type": dtype})  # type: ignore
kernel00 = getattr(ufcx00.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")  # type: ignore

ufcx01, _, _ = ffcx_jit(msh.comm, a01, form_compiler_options={"scalar_type": dtype})  # type: ignore
kernel01 = getattr(ufcx01.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")  # type: ignore

ufcx10, _, _ = ffcx_jit(msh.comm, a10, form_compiler_options={"scalar_type": dtype})  # type: ignore
kernel10 = getattr(ufcx10.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")  # type: ignore
# -

ffi = cffi.FFI()
if np.issubdtype(dtype, np.complexfloating):
    if cffi.__version_info__ > (1, 16, 99) and cffi.__version_info__ <= (1, 17, 1):
        print(
            "CFFI 1.17.0 and 1.17.1 has a bug for complex type."
            "See https://github.com/FEniCS/dolfinx/pull/3635. Exiting."
        )
        exit(0)
    cffi_support.register_type(ffi.typeof("double _Complex"), numba.types.complex128)

# Get local dofmap sizes for later local tensor tabulations

Ssize = S.element.space_dimension
Usize = U.element.space_dimension


# Next, we define a static condensation kernel that uses the
# previously defined kernels to compute the condensed local element
# tensor. The kernel is decorated with {py:func}`numba.cfunc` using the
# appropriate signature obtained from {py:func}`ufcx_signature`.`


@numba.cfunc(ufcx_signature(dtype, rtype), nopython=True)  # type: ignore
def tabulate_A(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL, custom_data=None):
    """Element kernel that applies static condensation."""

    # Prepare target condensed local element tensor
    A = numba.carray(A_, (Usize, Usize), dtype=dtype)

    # Tabulate all sub blocks locally
    A00 = np.zeros((Ssize, Ssize), dtype=dtype)
    kernel00(
        ffi.from_buffer(A00),
        w_,
        c_,
        coords_,
        entity_local_index,
        permutation,
        empty_void_pointer(),
    )

    A01 = np.zeros((Ssize, Usize), dtype=dtype)
    kernel01(
        ffi.from_buffer(A01),
        w_,
        c_,
        coords_,
        entity_local_index,
        permutation,
        empty_void_pointer(),
    )

    A10 = np.zeros((Usize, Ssize), dtype=dtype)
    kernel10(
        ffi.from_buffer(A10),
        w_,
        c_,
        coords_,
        entity_local_index,
        permutation,
        empty_void_pointer(),
    )

    # A = - A10 * A00^{-1} * A01
    A[:, :] = -A10 @ np.linalg.solve(A00, A01)


# Prepare a {py:class}`Form<dolfinx.fem.Form>` with a condensed
# tabulation kernel. We specify the integration domains to be the
# cells owned by the current process

formtype = form_cpp_class(dtype)  # type: ignore
cells = np.arange(msh.topology.index_map(msh.topology.dim).size_local)
integrals = {IntegralType.cell: [(0, tabulate_A.address, cells, np.array([], dtype=np.int8))]}
a_cond = Form(
    formtype([U._cpp_object, U._cpp_object], integrals, [], [], False, [], mesh=msh._cpp_object)
)

# Next, we pass the compiled kernel to the standard {py:func}`
# assemble_matrix <dolfinx.fem.petsc.assemble_matrix>` function to assemble
# to the global condensed stiffness matrix. We also assemble the right-hand
# side vector using {py:func}`assemble_vector
# <dolfinx.fem.petsc.assemble_vector>` and apply the boundary conditions by
# {py:func}`applying lifting <dolfinx.fem.petsc.apply_lifting>` and
# {py:meth}`set bc<dolfinx.fem.DirichletBC.set>`.

A_cond = assemble_matrix(a_cond, bcs=[bc])
A_cond.assemble()
b = assemble_vector(b1)
apply_lifting(b, [a_cond], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
bc.set(b)

# We use a {py:class}`PETSc.KSP <petsc4py.PETSc.KSP>` solver to solve the
# condensed linear system. The solution is stored in a
# {py:class}`Function<dolfinx.fem.Function>`, while we pass the
# underlying data wrapped as a {py:class}`PETSc.Vec <petsc4py.PETSc.Vec>`
# to the solver by calling {py:meth}`petsc_vec
# <dolfinx.la.Vector.petsc_vec>` on the
# {py:meth}`vector <dolfinx.fem.Function.x>` attribute of the function.

uc = Function(U, name="u_from_condensation")
solver = PETSc.KSP().create(A_cond.getComm())  # type: ignore
solver.setOperators(A_cond)
solver.solve(b, uc.x.petsc_vec)
solver.destroy()

# We verify the condensed solution by comparing against a standard,
# pure displacement based formulation

a = form(-ufl.inner(sigma_u(u), ufl.grad(v)) * ufl.dx)
A = assemble_matrix(a, bcs=[bc])
A.assemble()

# Create {py:class}`BoundingBoxTree <dolfinx.geometry.BoundingBoxTree>`
# using {py:meth}`bb_tree <dolfinx.geometry.bb_tree>` constructor
# for efficient computation of the ownership of a set of evaluation points

bb_tree = geometry.bb_tree(msh, tdim, padding=0.0)

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
