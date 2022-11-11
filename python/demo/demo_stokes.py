# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Stokes equations with Taylor-Hood elements
#
# This demo shows how to solve the Stokes problem using Taylor-Hood
# elements with a range of different linear solvers.
#
# ## Equation and problem definition
#
# ### Strong formulation
#
# $$
# - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega,
#
# \nabla \cdot u &= 0 \quad {\rm in} \ \Omega.
# $$
#
# ```{note}
# The sign of the pressure has been flipped from the classical
# definition. This is done in order to have a symmetric system
# of equations rather than a non-symmetric system of equations.
# ```
#
# A typical set of boundary conditions on the boundary $\partial
# \Omega = \Gamma_{D} \cup \Gamma_{N}$ can be:
#
# $$
# u &= u_0 \quad {\rm on} \ \Gamma_{D},
#
# \nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \ \Gamma_{N}.
# $$
#
# ### Weak formulation
#
# We formulate the Stokes equations' mixed variational form; that is, a
# form where the two variables, the velocity and the pressure, are
# approximated. We have the problem: find $(u, p) \in W$ such that
#
# $$
# a((u, p), (v, q)) = L((v, q))
# $$
#
# for all $(v, q) \in W$, where
#
# $$
# a((u, p), (v, q)) &:= \int_{\Omega} \nabla u \cdot \nabla v -
#            \nabla \cdot v \ p + \nabla \cdot u \ q \, {\rm d} x,
#
# L((v, q)) &:= \int_{\Omega} f \cdot v \, {\rm d} x + \int_{\partial
#            \Omega_N} g \cdot v \, {\rm d} s.
# $$
#
# The space $W$ is a mixed (product) function space $W = V
# \times Q$, such that $u \in V$ and $q \in Q$.
#
# ### Domain and boundary conditions
#
# We define the lid-driven cavity problem with the following
# domain and boundary conditions:
#
# - $\Omega = [0,1]\times[0,1]$ (a unit square)
# - $\Gamma_D = \partial \Omega$
# - $u_0 = (1, 0)^\top$ at $x_1 = 1$ and $u_0 = (0,
#   0)^\top$ otherwise
# - $f = (0, 0)^\top$
#
# ## Implementation
#
# We first import the modules and functions that the program uses:

# +
import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form,
                         locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_rectangle,
                          locate_entities_boundary)
from ufl import div, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc

# -

# We create a {py:class}`Mesh <dolfinx.mesh.Mesh>`, define functions to
# geometrically locate subsets of its boundary and define a function
# describing the velocity to be imposed as a boundary condition in a lid
# driven cavity problem:

# +
# Create mesh
msh = create_rectangle(MPI.COMM_WORLD,
                       [np.array([0, 0]), np.array([1, 1])],
                       [32, 32],
                       CellType.triangle, GhostMode.none)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))
# -

# We define two {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# instances with different finite elements. `P2` corresponds to a continuous
# piecewise quadratic basis for the velocity field and `P1` to a continuous
# piecewise linear basis for the pressure field:


P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)

# We define boundary conditions:

# +
# No-slip boundary condition for velocity field (`V`) on boundaries
# where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(V)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]
# -

# We now define the bilinear and linear forms corresponding to the weak
# mixed formulation of the Stokes equations in a blocked structure:

# +
# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))

a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
          [inner(div(u), q) * dx, None]])
L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])
# -

# We will use a block-diagonal preconditioner to solve this problem:

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None],
       [None, a_p11]]

# ### Nested matrix solver
#
# We now assemble the bilinear form into a nested matrix `A`, and call
# the `assemble()` method to communicate shared entries in parallel.
# Rows and columns in `A` that correspond to degrees-of-freedom with
# Dirichlet boundary conditions are zeroed and a value of 1 is set on
# the diagonal.

A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
A.assemble()

# We create a nested matrix `P` to use as the preconditioner. The
# top-left block of `P` is shared with the top-left block of `A`. The
# bottom-right diagonal entry is assembled from the form `a_p11`:

P11 = fem.petsc.assemble_matrix(a_p11, [])
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
P.assemble()

# Next, the right-hand side vector is assembled and then modified to
# account for non-homogeneous Dirichlet boundary conditions:

# +
b = fem.petsc.assemble_vector_nest(L)

# Modify ('lift') the RHS for Dirichlet boundary conditions
fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

# Sum contributions from ghost entries on the owner
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
fem.petsc.set_bc_nest(b, bcs0)
# -

# The pressure field for this problem is determined only up to a
# constant. We can supply the vector that spans the nullspace and any
# component of the solution in this direction will be eliminated during
# the iterative linear solution process.

# +
# Create nullspace vector
null_vec = fem.petsc.create_vector_nest(L)

# Set velocity part to zero and the pressure part to a non-zero constant
null_vecs = null_vec.getNestSubVecs()
null_vecs[0].set(0.0), null_vecs[1].set(1.0)

# Normalize the vector, create a nullspace object, and attach it to the
# matrix
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)
# -

# Now we create a Krylov Subspace Solver `ksp`. We configure it to use
# the MINRES method, and a block-diagonal preconditioner using PETSc's
# additive fieldsplit type preconditioner:

# +
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-9)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

# Define the matrix blocks in the preconditioner with the velocity and
# pressure matrix index sets
nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(
    ("u", nested_IS[0][0]),
    ("p", nested_IS[0][1]))

# Set the preconditioners for each block
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")

# Monitor the convergence of the KSP
ksp.setFromOptions()
# -

# To compute the solution, we create finite element {py:class}`Function
# <dolfinx.fem.Function>` for the velocity (on the space `V`) and
# for the pressure (on the space `Q`). The vectors for `u` and `p` are
# combined to form a nested vector and the system is solved:

u, p = Function(V), Function(Q)
x = PETSc.Vec().createNest([_cpp.la.petsc.create_vector_wrap(u.x), _cpp.la.petsc.create_vector_wrap(p.x)])
ksp.solve(b, x)

# Norms of the solution vectors are computed:

norm_u_0 = u.x.norm()
norm_p_0 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(A) Norm of velocity coefficient vector (nested, iterative): {}".format(norm_u_0))
    print("(A) Norm of pressure coefficient vector (nested, iterative): {}".format(norm_p_0))

# The solution fields can be saved to file in XDMF format for
# visualization, e.g. with ParaView. Before writing to file, ghost values
# are updated.

# +
with XDMFFile(MPI.COMM_WORLD, "out_stokes/velocity.xdmf", "w") as ufile_xdmf:
    u.x.scatter_forward()
    ufile_xdmf.write_mesh(msh)
    ufile_xdmf.write_function(u)

with XDMFFile(MPI.COMM_WORLD, "out_stokes/pressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p)
# -

# ### Monolithic block iterative solver
#
# Next, we solve same problem, but now with monolithic (non-nested)
# matrices and iterative solvers.

# +
A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
P = fem.petsc.assemble_matrix_block(a_p, bcs=bcs)
P.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Set near nullspace for pressure
null_vec = A.createVecLeft()
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
null_vec.array[offset:] = 1.0
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

# Build IndexSets for each field (global dof indices for each field)
V_map = V.dofmap.index_map
Q_map = Q.dofmap.index_map
offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

# Create Krylov solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setTolerances(rtol=1e-9)
ksp.setType("minres")
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(
    ("u", is_u),
    ("p", is_p))

# Configure velocity and pressure sub KSPs
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")

# Monitor the convergence of the KSP
opts = PETSc.Options()
opts["ksp_monitor"] = None
opts["ksp_view"] = None
ksp.setFromOptions()
# -

# We also need to create a block vector, `x`, to store the (full)
# solution, which we initialize using the block RHS form `L`.

# +
# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Create Functions and scatter x solution
u, p = Function(V), Function(Q)
offset = V_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
p.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
# -

# We can calculate the $L^2$ norms of u and p as follows:

norm_u_1 = u.x.norm()
norm_p_1 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(B) Norm of velocity coefficient vector (blocked, iterative): {}".format(norm_u_1))
    print("(B) Norm of pressure coefficient vector (blocked, iterative): {}".format(norm_p_1))
assert np.isclose(norm_u_1, norm_u_0)
assert np.isclose(norm_p_1, norm_p_0)

# ### Monolithic block direct solver
#
# Solve same problem, but now with monolithic matrices and a direct solver

# Create LU solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# We also need to create a block vector, `x`, to store the (full)
# solution, which we initialize using the block RHS form `L`.

# +
# Compute solution
x = A.createVecLeft()
ksp.solve(b, x)

# Create Functions and scatter x solution
u, p = Function(V), Function(Q)
offset = V_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
p.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
# -

# We can calculate the $L^2$ norms of u and p as follows:

norm_u_2 = u.x.norm()
norm_p_2 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(C) Norm of velocity coefficient vector (blocked, direct): {}".format(norm_u_2))
    print("(C) Norm of pressure coefficient vector (blocked, direct): {}".format(norm_p_2))
assert np.isclose(norm_u_2, norm_u_0)
assert np.isclose(norm_p_2, norm_p_0)

# ### Non-blocked direct solver
#
# Again, solve the same problem but this time with a non-blocked direct
# solver approach

# +
# Create the function space
TH = P2 * P1
W = FunctionSpace(msh, TH)
W0, _ = W.sub(0).collapse()

# No slip boundary condition
noslip = Function(V)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
dofs = locate_dofs_topological((W.sub(0), V), 1, facets)
bc0 = dirichletbc(noslip, dofs, W.sub(0))


# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(W0)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
dofs = locate_dofs_topological((W.sub(0), V), 1, facets)
bc1 = dirichletbc(lid_velocity, dofs, W.sub(0))


# Since for this problem the pressure is only determined up to a
# constant, we pin the pressure at the point (0, 0)
zero = Function(Q)
zero.x.set(0.0)
dofs = locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
bc2 = dirichletbc(zero, dofs, W.sub(1))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(W0)
a = form((inner(grad(u), grad(v)) + inner(p, div(v)) + inner(div(u), q)) * dx)
L = form(inner(f, v) * dx)


# Assemble LHS matrix and RHS vector
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L)

fem.petsc.apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
fem.petsc.set_bc(b, bcs)

# Create and configure solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute the solution
U = Function(W)
ksp.solve(b, U.vector)

# Split the mixed solution and collapse
u = U.sub(0).collapse()
p = U.sub(1).collapse()

# Compute norms
norm_u_3 = u.x.norm()
norm_p_3 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(D) Norm of velocity coefficient vector (monolithic, direct): {}".format(norm_u_3))
    print("(D) Norm of pressure coefficient vector (monolithic, direct): {}".format(norm_p_3))
assert np.isclose(norm_u_3, norm_u_0)

# Write the solution to file
with XDMFFile(MPI.COMM_WORLD, "out_stokes/new_velocity.xdmf", "w") as ufile_xdmf:
    u.x.scatter_forward()
    ufile_xdmf.write_mesh(msh)
    ufile_xdmf.write_function(u)

with XDMFFile(MPI.COMM_WORLD, "out_stokes/my.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p)
