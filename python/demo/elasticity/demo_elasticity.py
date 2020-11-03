#
# .. _demo_elasticity:
#
# Elasticity equation
# ===================
# Copyright (C) 2020 Garth N. Wells and Michal Habera
#
# This demo solves the equations of static linear elasticity. The solver uses
# smoothed aggregation algebraic multigrid. ::

from contextlib import ExitStack
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import BoxMesh, DirichletBC, Function, VectorFunctionSpace, cpp
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (Form, apply_lifting, assemble_matrix, assemble_vector,
                         locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.la import VectorSpaceBasis
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, dx, grad, inner, sym, tr)

# Nullspace and problem setup
# ---------------------------
#
# Prepare a helper which builds PETSc' NullSpace.
# Nullspace (or near nullspace) is needed to improve the
# performance of algebraic multigrid.
#
# In the case of small deformation linear elasticity the nullspace
# contains rigid body modes. ::


def build_nullspace(V):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    nullspace_basis = [cpp.la.create_vector(index_map) for i in range(6)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

        # Build translational null space basis
        for i in range(3):
            basis[i][V.sub(i).dofmap.list.array] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs = [V.sub(i).dofmap.list.array for i in range(3)]
        basis[3][dofs[0]] = -x[dofs[0], 1]
        basis[3][dofs[1]] = x[dofs[1], 0]
        basis[4][dofs[0]] = x[dofs[0], 2]
        basis[4][dofs[2]] = -x[dofs[2], 0]
        basis[5][dofs[2]] = x[dofs[2], 1]
        basis[5][dofs[1]] = -x[dofs[1], 2]

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(6)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp


mesh = BoxMesh(
    MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                     np.array([2.0, 1.0, 1.0])], [12, 12, 12],
    CellType.tetrahedron, dolfinx.cpp.mesh.GhostMode.none)


def boundary(x):
    return np.logical_or(np.isclose(x[0], 0.0),
                         np.isclose(x[1], 1.0))


# Rotation rate and mass density
omega = 300.0
rho = 10.0

# Loading due to centripetal acceleration (rho*omega^2*x_i)
x = SpatialCoordinate(mesh)
f = as_vector((rho * omega**2 * x[0], rho * omega**2 * x[1], 0.0))

# Elasticity parameters
E = 1.0e9
nu = 0.0
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def sigma(v):
    return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(
        len(v))


# Create function space
V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v)) * dx
L = inner(f, v) * dx

u0 = Function(V)
with u0.vector.localForm() as bc_local:
    bc_local.set(0.0)

# Set up boundary condition on inner surface
bc = DirichletBC(u0, locate_dofs_geometrical(V, boundary))

# Controlling compilation parameters
# ----------------------------------
#
# Parameters which control FFCX and JIT compilation could be set
# directly with the interface of :py:class:`Form <dolfinx.fem.Form>` or
# via environmental variables.
#
# This demo shows a mixed approach, where C compilation
# flags are set with environmental variables.
# Some parameters which control FFCX compilation are passed directly to the ``Form``.
# ::

os.environ["DOLFINX_JIT_CFLAGS"] = "-Ofast -march=native"
os.environ["FFCX_VERBOSITY"] = "20"

form = Form(a, form_compiler_parameters={"quadrature_degree": 1})

# The use of such aggresive compiler flags (e.g. ``-Ofast`` violates IEEE floating point standard)
# often results in a faster assembly code, but slower JIT compilation.
# FFCX verbosity levels follow Python std logging library levels, https://docs.python.org/3/library/logging.html.
# To see all available form compiler parameters run ``ffcx --help`` in the commandline.
#
# .. warning::
#    Environmental variables override any other parameters passed to the ``Form``, or directly stated
#    in the metadata of an integral. Please make sure there are no environmental variables set
#    with side-effects.
#
# Assembly and solve
# ------------------
# ::

# Assemble system, applying boundary conditions and preserving symmetry
A = assemble_matrix(form, [bc])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Create solution function
u = Function(V)

# Create near null space basis (required for smoothed aggregation AMG).
null_space = build_nullspace(V)

# Attach near nullspace to matrix
A.setNearNullSpace(null_space)

# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-12
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

# Create CG Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

# Compute solution
solver.setMonitor(lambda ksp, its, rnorm: print("Iteration: {}, rel. residual: {}".format(its, rnorm)))
solver.solve(b, u.vector)
solver.view()

# Save solution to XDMF format
with XDMFFile(MPI.COMM_WORLD, "elasticity.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)

unorm = u.vector.norm()
if mesh.mpi_comm().rank == 0:
    print("Solution vector norm:", unorm)
