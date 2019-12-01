#
# .. _demo_pde_stokes-taylor-hood_python_documentation:
#
# Stokes equations with Taylor-Hood elements
# ==========================================
#
# This demo show how to solve the Stokes problem using Taylor-Hood
# elements with a range of different linbear solvers.
#
# Equation and problem definition
# -------------------------------
#
# Strong formulation
# ^^^^^^^^^^^^^^^^^^
#
# .. math:: - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \
#         \Omega, \\
#                         \nabla \cdot u &= 0 \quad {\rm in} \ \Omega.
#                         \\
#
#
# .. note:: The sign of the pressure has been flipped from the classical
#         definition. This is done in order to have a symmetric system
#         of equations rather than a non-symmetric system of equations.
#
# A typical set of boundary conditions on the boundary :math:`\partial
# \Omega = \Gamma_{D} \cup \Gamma_{N}` can be:
#
# .. math:: u &= u_0 \quad {\rm on} \ \Gamma_{D}, \\
#         \nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \
#         \Gamma_{N}. \\
#
#
# Weak formulation
# ^^^^^^^^^^^^^^^^
#
# We formulate the Stokes equations mixed variational form; that is, a
# form where the two variables, the velocity and the pressure, are
# approximated. We have the problem: find :math:`(u, p) \in W` such that
#
# .. math:: a((u, p), (v, q)) = L((v, q))
#
# for all :math:`(v, q) \in W`, where
#
# .. math::
#
#         a((u, p), (v, q))
#                                 &= \int_{\Omega} \nabla u \cdot \nabla v
#                  - \nabla \cdot v \ p
#                  + \nabla \cdot u \ q \, {\rm d} x, \\
#         L((v, q))
#                                 &= \int_{\Omega} f \cdot v \, {\rm d} x
#                         + \int_{\partial \Omega_N} g \cdot v \, {\rm d} s. \\
#
# The space :math:`W` is mixed (product) function space :math:`W = V
# \times Q`, such that :math:`u \in V` and :math:`q \in Q`.
#
# Domain and boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We shall consider the following definitions of the input functions,
# the domain, and the boundaries:
#
# * :math:`\Omega = [0,1]\times[0,1] \setminus {\rm dolphin}` (a unit
#   square)
# * :math:`\Gamma_N = \{ 0 \} \times [0, 1]`
# * :math:`\Gamma_D = \partial \Omega \setminus \Gamma_N`
# * :math:`u_0 = (- \sin(\pi x_1), 0)^\top` at :math:`x_0 = 1` and
#   :math:`u_0 = (0, 0)^\top` otherwise
# * :math:`f = (0, 0)^\top`
# * :math:`g = (0, 0)^\top`
#
#
# Implementation
# --------------
#
# In this example, different boundary conditions are prescribed on
# different parts of the domain's exterior. Each sub-region is tagged
# with a different (integer) label. For this purpose, DOLFIN provides a
# :py:class:`MeshFunction <dolfin.cpp.mesh.MeshFunction>` class
# representing functions over mesh entities (such as over cells or over
# facets). Meshes and mesh functions can be read from file in the
# following way::

import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc

import dolfin
import ufl
from dolfin import MPI, DirichletBC, Function, FunctionSpace, RectangleMesh
from dolfin.cpp.mesh import CellType
from dolfin.io import XDMFFile
from dolfin.la import VectorSpaceBasis
from dolfin.plotting import plot
from ufl import div, dx, grad, inner

# Create mesh
mesh = RectangleMesh(
    MPI.comm_world,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [3, 2],
    CellType.triangle, dolfin.cpp.mesh.GhostMode.none)

cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
mesh.geometry.coord_mapping = cmap

# We define two :py:class:`FunctionSpace
# <dolfin.functions.functionspace.FunctionSpace>` instances with
# different finite elements. ``P2`` corresponds to piecewise quadratics
# for the velocity field and ``P1`` to continuous piecewise linears for
# the pressure field::

# Define function spaces
P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, P2)
Q = FunctionSpace(mesh, P1)

# The product of these finite element spaces is known as the Taylor–Hood
# mixed element. It is a standard stable element pair for the Stokes
# equations. Now we can define boundary conditions::


# No-slip boundary condition for velocity on boundaries where x = 0, x =
# 1, and y = 0.
noslip = Function(V)
bc0 = DirichletBC(V, noslip,
                  lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                        np.isclose(x[0], 1.0)),
                                          np.isclose(x[1], 0.0)))


# Driving velocity condition for velocity on y = 1
lid_velocity = Function(V)
lid_velocity.interpolate(lambda x: np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
bc1 = DirichletBC(V, lid_velocity, lambda x: np.isclose(x[1], 1.0))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

# We now define the bilinear and linear forms corresponding to the weak
# mixed formulation of the Stokes equations. In our implementation we
# write these formulations in a blocked structure::

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = dolfin.Constant(mesh, (0, 0))

a = [[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
     [inner(div(u), q) * dx, None]]

L = [inner(f, v) * dx,
     inner(dolfin.Constant(mesh, 0), q) * dx]

a_p11 = inner(p, q) * dx
a_p = [[a[0][0], None],
       [None, a_p11]]

# With the bilinear form ``a``, preconditioner bilinear form ``prec``
# and linear right hand side (RHS) ``L``, we may now assembly the finite
# element linear system. We exploit the structure of the Stokes system
# and assemble the finite element system into block matrices and a block
# vector. Provision of the ``bcs`` argument to
# :py:func:`assemble_matrix_nest <dolfin.fem.assemble_matrix_nest>`
# ensures the rows and columns associated with the boundary conditions
# are zeroed and the diagonal set to the identity, preserving symmetry::

A = dolfin.fem.assemble_matrix_nest(a, bcs)
A.assemble()

# The preconditioner P can share the A_00 block
P11 = dolfin.fem.assemble_matrix(a_p11, [])
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
P.assemble()


# Assemble the RHS vector
b = dolfin.fem.assemble.assemble_vector_nest(L)

# Modify ('lift') the RHS for Dirichlet boundary conditions
dolfin.fem.assemble.apply_lifting_nest(b, a, bcs)
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet bc values in the RHS
bcs0 = dolfin.cpp.fem.bcs_rows(dolfin.fem.assemble._create_cpp_form(L), bcs)
dolfin.fem.assemble.set_bc_nest(b, bcs0)

# Ths pressure field is determined only up to a constant. We can supply
# the this vector and it will be eliminated during the iterative linear
# solution process.
null_vec = dolfin.fem.create_vector_nest(L)
null_vecs = null_vec.getNestSubVecs()
null_vecs[0].set(0.0), null_vecs[1].set(1.0)
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

# Now we are ready to create a Krylov Subspace Solver ``ksp``. We
# configure it for block Jacobi preconditioning using PETSc's additive
# fieldsplit composite type.

ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setOperators(A, P)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-12)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

# Define the matrix blocks in the preconditioner with the velocity and
# pressure matrix index sets
nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(
    ("u", nested_IS[0][0]),
    ("p", nested_IS[0][1]))

ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("hypre")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")

# Monitor the convergence of the KSP
opts = PETSc.Options()
# opts["ksp_monitor"] = None
# opts["ksp_view"] = None

# Configure velocity and pressure sub KSPs
ksp.setFromOptions()

# Compute solution
u, p = Function(V), Function(Q)
x = PETSc.Vec().createNest([u.vector, p.vector])
ksp.solve(b, x)

# We can calculate the :math:`L^2` norms of u and p as follows::

norm_u_0 = u.vector.norm()
norm_p_0 = p.vector.norm()
print("(A) Norm of velocity coefficient vector: {}".format(norm_u_0))
print("(A) Norm of pressure coefficient vector: {}".format(norm_p_0))

# Save solution in XDMF format
with XDMFFile(MPI.comm_world, "velocity.xdmf") as ufile_xdmf:
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ufile_xdmf.write(u)

with XDMFFile(MPI.comm_world, "pressure.xdmf") as pfile_xdmf:
    p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    pfile_xdmf.write(p)

# Next, we solve same problem, but now with monolithic (non-nested)
# matrices and iterative solvers

A = dolfin.fem.assemble_matrix_block(a, bcs)
A.assemble()
P = dolfin.fem.assemble_matrix_block(a_p, bcs)
P.assemble()
b = dolfin.fem.assemble.assemble_vector_block(L, a, bcs)

# Set near null space for pressure
null_vec = A.createVecLeft()
offset = V.dofmap.index_map.size_local * V.dofmap.index_map.block_size
null_vec.array[offset:] = 1.0
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

# Build IndexSets for each field (global dof indices for each field)
V_map = V.dofmap.index_map
Q_map = Q.dofmap.index_map
offset_u = V_map.local_range[0] * V_map.block_size + Q_map.local_range[0]
offset_p = offset_u + V_map.size_local * V_map.block_size
is_u = PETSc.IS().createStride(V_map.size_local * V_map.block_size, offset_u, 1, comm=PETSc.COMM_SELF)
is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

# Create Krylov solver
ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setOperators(A, P)
ksp.setTolerances(rtol=1e-12)
ksp.setType("minres")
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(
    ("u", is_u),
    ("p", is_p))

# Configure velocity and pressure sub KSPs
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("hypre")
ksp_p.setType("preonly")
ksp_p.getPC().setType("hypre")

# Monitor the convergence of the KSP
opts = PETSc.Options()
# opts["ksp_monitor"] = None
# opts["ksp_view"] = None

ksp.setFromOptions()

# We also need to create a block vector,``x``, to store the (full)
# solution, which we initialize using the block RHS form ``L``.

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Create Functions and scatter x solution
u, p = Function(V), Function(Q)
offset = V_map.size_local * V_map.block_size
u.vector.array[:] = x.array_r[:offset]
p.vector.array[:] = x.array_r[offset:]

# We can calculate the :math:`L^2` norms of u and p as follows::

norm_u_1 = u.vector.norm()
norm_p_1 = p.vector.norm()
print("(B) Norm of velocity coefficient vector: {}".format(norm_u_1))
print("(B) Norm of pressure coefficient vector: {}".format(norm_p_1))
assert np.isclose(norm_u_1, norm_u_0)
assert np.isclose(norm_p_1, norm_p_0)

# Solve same problem, but now with monolithic matrices and a direct solver

# Create LU solver
ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Monitor the convergence of the KSP
opts = PETSc.Options()
# opts["ksp_monitor"] = None
# opts["ksp_view"] = None

ksp.setFromOptions()

# We also need to create a block vector,``x``, to store the (full)
# solution, which we initialize using the block RHS form ``L``.

# Compute solution
x = A.createVecLeft()
ksp.solve(b, x)

# Create Functions and scatter x solution
u, p = Function(V), Function(Q)
offset = V_map.size_local * V_map.block_size
u.vector.array[:] = x.array_r[:offset]
p.vector.array[:] = x.array_r[offset:]

# We can calculate the :math:`L^2` norms of u and p as follows::

norm_u_2 = u.vector.norm()
norm_p_2 = p.vector.norm()
print("(C) Norm of velocity coefficient vector: {}".format(norm_u_2))
print("(C) Norm of pressure coefficient vector: {}".format(norm_p_2))
assert np.isclose(norm_u_2, norm_u_0)
assert np.isclose(norm_p_2, norm_p_0)
