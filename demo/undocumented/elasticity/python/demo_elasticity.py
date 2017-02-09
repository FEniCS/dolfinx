"""This demo solves the equations of static linear elasticity for a
pulley subjected to centripetal accelerations. The solver uses
smoothed aggregation algerbaric multigrid."""

# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
from dolfin import *

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"

def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


# Load mesh from file
mesh = Mesh()
XDMFFile(mpi_comm_world(), "../pulley.xdmf").read(mesh)

# Function to mark inner surface of pulley
def inner_surface(x, on_boundary):
    r = 3.75 - x[2]*0.17
    return (x[0]*x[0] + x[1]*x[1]) < r*r and on_boundary

# Rotation rate and mass density
omega = 300.0
rho = 10.0

# Loading due to centripetal acceleration (rho*omega^2*x_i)
f = Expression(("rho*omega*omega*x[0]", "rho*omega*omega*x[1]", "0.0"),
               omega=omega, rho=rho, degree=2)

# Elasticity parameters
E = 1.0e9
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx

# Set up boundary condition on inner surface
c = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, c, inner_surface)

# Assemble system, applying boundary conditions and preserving
# symmetry)
A, b = assemble_system(a, L, bc)

# Create solution function
u = Function(V)

# Create near null space basis (required for smoothed aggregation
# AMG). The solution vector is passed so that it can be copied to
# generate compatible vectors for the nullspace.
null_space = build_nullspace(V, u.vector())

# Attach near nullspace to matrix
as_backend_type(A).set_near_nullspace(null_space)

# Create PETSC smoothed aggregation AMG preconditioner and attach near
# null space
pc = PETScPreconditioner("petsc_amg")

# Use Chebyshev smoothing for multigrid
PETScOptions.set("mg_levels_ksp_type", "chebyshev")
PETScOptions.set("mg_levels_pc_type", "jacobi")

# Improve estimate of eigenvalues for Chebyshev smoothing
PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

# Create CG Krylov solver and turn convergence monitoring on
solver = PETScKrylovSolver("cg", pc)
solver.parameters["monitor_convergence"] = True

# Set matrix operator
solver.set_operator(A);

# Compute solution
solver.solve(u.vector(), b);

# Save solution to VTK format
File("elasticity.pvd", "compressed") << u

# Save colored mesh partitions in VTK format if running in parallel
if MPI.size(mesh.mpi_comm()) > 1:
    File("partitions.pvd") << CellFunction("size_t", mesh, \
                                           MPI.rank(mesh.mpi_comm()))

# Project and write stress field to post-processing file
W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
stress = project(sigma(u), V=W)
File("stress.pvd") << stress

# Plot solution
plot(u, interactive=True)
