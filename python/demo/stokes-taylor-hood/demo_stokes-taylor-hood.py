#
# .. _demo_pde_stokes-taylor-hood_python_documentation:
#
# Stokes equations with Taylor-Hood elements
# ==========================================
#
# This demo is implemented in a single Python file,
# :download:`demo_stokes-taylor-hood.py`, which contains both the
# variational form and the solver.
#
# Equation and problem definition
# -------------------------------
#
# Strong formulation
# ^^^^^^^^^^^^^^^^^^
#
# .. math::
#         - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega, \\
#                         \nabla \cdot u &= 0 \quad {\rm in} \ \Omega. \\
#
#
# .. note::
#         The sign of the pressure has been flipped from the classical
#         definition. This is done in order to have a symmetric (but not
#         positive-definite) system of equations rather than a
#         non-symmetric (but positive-definite) system of equations.
#
# A typical set of boundary conditions on the boundary :math:`\partial
# \Omega = \Gamma_{D} \cup \Gamma_{N}` can be:
#
# .. math::
#         u &= u_0 \quad {\rm on} \ \Gamma_{D}, \\
#         \nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \ \Gamma_{N}. \\
#
#
# Weak formulation
# ^^^^^^^^^^^^^^^^
#
# The Stokes equations can easily be formulated in a mixed variational
# form; that is, a form where the two variables, the velocity and the
# pressure, are approximated simultaneously. Using the abstract
# framework, we have the problem: find :math:`(u, p) \in W` such that
#
# .. math::
#         a((u, p), (v, q)) = L((v, q))
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
# The space :math:`W` should be a mixed (product) function space
# :math:`W = V \times Q`, such that :math:`u \in V` and :math:`q \in Q`.
#
# Domain and boundary conditions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In this demo, we shall consider the following definitions of the input functions, the domain, and the boundaries:
#
# * :math:`\Omega = [0,1]\times[0,1] \setminus {\rm dolphin}` (a unit square)
# * :math:`\Gamma_N = \{ 0 \} \times [0, 1]`
# * :math:`\Gamma_D = \partial \Omega \setminus \Gamma_N`
# * :math:`u_0 = (- \sin(\pi x_1), 0)^\top` at :math:`x_0 = 1` and :math:`u_0 = (0, 0)^\top` otherwise
# * :math:`f = (0, 0)^\top`
# * :math:`g = (0, 0)^\top`
#
#
# Implementation
# --------------
#
# In this example, different boundary conditions are prescribed on
# different parts of the domain's exterior. Each sub-region is tagged with a
# different (integer) label. For this purpose, DOLFIN provides
# a :py:class:`MeshFunction <dolfin.cpp.mesh.MeshFunction>` class
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

# Load mesh and subdomains
# xdmf = XDMFFile(MPI.comm_world, "../dolfin_fine.xdmf")
# mesh = xdmf.read_mesh(dolfin.cpp.mesh.GhostMode.none)
mesh = RectangleMesh(
    MPI.comm_world,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [3, 2],
    CellType.triangle, dolfin.cpp.mesh.GhostMode.none)

# sub_domains = xdmf.read_mf_size_t(mesh)

cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
mesh.geometry.coord_mapping = cmap

# Next, we define two :py:class:`FunctionSpace
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

# Extract subdomain facet arrays
# mf = sub_domains.values
# mf0 = np.where(mf == 0)
# mf1 = np.where(mf == 1)

# No-slip boundary condition for velocity
# x1 = 0, x1 = 1 and around the dolphin
noslip = Function(V)
# noslip.interpolate(lambda x: np.zeros_like(x[:mesh.geometry.dim]))
bc0 = DirichletBC(V, noslip,
                  lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                        np.isclose(x[0], 1.0)),
                                          np.isclose(x[1], 0.0)))


# Inflow boundary condition for velocity at x0 = 1
def inflow_eval(x):
    values = np.zeros((2, x.shape[1]))
    values[0] = 1.0
    return values


inflow = Function(V)
inflow.interpolate(inflow_eval)
bc1 = DirichletBC(V, inflow, lambda x: np.isclose(x[1], 1.0))

# Collect boundary conditions
bcs = [bc0, bc1]

# The first argument to :py:class:`DirichletBC
# <dolfin.cpp.fem.DirichletBC>` specifies the :py:class:`FunctionSpace
# <dolfin.cpp.function.FunctionSpace>`. The second argument specifies
# the value on the Dirichlet boundary. The last argument specifies the
# marking of the subdomains: ``mf0`` and ``mf1`` contain the ``0`` and
# ``1`` subdomain markers, respectively.
#
# We now define the bilinear and linear forms corresponding to the weak
# mixed formulation of the Stokes equations. In our implementation we
# write these formulations in a blocked structure::

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = dolfin.Constant(mesh, (0, 0))

a = [[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
     [inner(div(u), q) * dx, None]]

prec = [[a[0][0], None],
        [None, inner(p, q) * dx]]

L = [inner(f, v) * dx,
     inner(dolfin.Constant(mesh, 0), q) * dx]

# With the bilinear form ``a``, preconditioner bilinear form ``prec``
# and linear right hand side (RHS) ``L``, we may now assembly the finite
# element linear system. We exploit the structure of the Stokes system
# and assemble the finite element system into block matrices and a block
# vector. Provision of the ``bcs`` argument to
# :py:func:`assemble_matrix_nest <dolfin.fem.assemble_matrix_nest>`
# ensures the rows and columns associated with the boundary conditions
# are zeroed and the diagonal set to the identity, preserving symmetry::

A = dolfin.fem.create_matrix_nest(a)
dolfin.fem.assemble_matrix_nest(A, a, bcs)
A.assemble()

P = dolfin.fem.create_matrix_nest(prec)
dolfin.fem.assemble_matrix_nest(P, prec, bcs)
P.assemble()

b = dolfin.fem.create_vector_nest(L)
b.set(0.0)

# The boundary conditions we collected in ``bcs`` may now be applied to
# the RHS block vector ``b``. In this case we must apply the lifting
# operator to remove the columns of ``a`` corresponding to the boundary
# conditions from ``b`` with :py:func:`apply_lifting_nest
# <dolfin.fem.assemble.apply_lifting_nest>`. Thereafter we prescribe the
# values of the boundary conditions using :py:func:`set_bc_nest
# <dolfin.fem.assemble.set_bc_nest>`::

dolfin.fem.assemble.assemble_vector_nest(b, L)
dolfin.fem.assemble.apply_lifting_nest(b, a, bcs)
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = dolfin.cpp.fem.bcs_rows(dolfin.fem.assemble._create_cpp_form(L), bcs)
dolfin.fem.assemble.set_bc_nest(b, bcs0)

b.assemble()

# Now we are ready to create a Krylov Subspace Solver ``ksp``. We
# configure it for block Jacobi preconditioning using PETSc's additive
# fieldsplit composite type.

ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setOperators(A, P)

# Setup the parent KSP
ksp.setTolerances(rtol=1e-12)
ksp.setType("minres")

# Set near null space for pressure
null_vec = dolfin.fem.create_vector_nest(L)
null_vecs = null_vec.getNestSubVecs()
null_vecs[0].set(0.0)
null_vecs[1].set(1.0)
null_vec.normalize()
nsp = PETSc.NullSpace().create(False, [null_vec])
A.setNullSpace(nsp)
assert np.isclose((A * null_vec).norm(), 0.0)

# Monitor the convergence of the KSP
opts = PETSc.Options()
# opts["ksp_monitor"] = None
# opts["ksp_view"] = None

ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

# Supply the KSP with the velocity and pressure matrix index sets
nested_IS = A.getNestISs()
ksp.getPC().setFieldSplitIS(
    ("u", nested_IS[0][0]),
    ("p", nested_IS[0][1]))

# nested_IS[0][0].view()
# nested_IS[0][1].view()
# exit(0)

# Configure velocity and pressure sub KSPs
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("hypre")
ksp_p.setType("preonly")
ksp_p.getPC().setType("hypre")

ksp.setFromOptions()

# We also need to create a block vector,``x``, to store the (full)
# solution, which we initialize using the block RHS form ``L``.

# Compute solution
u, p = Function(V), Function(Q)
x = PETSc.Vec().createNest([u.vector, p.vector])
ksp.solve(b, x)

# We can calculate the :math:`L^2` norms of u and p as follows::

print("NN Norm of whole solution vector: {}".format(x.norm()))
print("NN Norm of velocity coefficient vector: {}".format(u.vector.norm()))
print("NN Norm of pressure coefficient vector: {}".format(p.vector.norm()))
ref = x.norm()

# Check pressure norm
# assert np.isclose(p.vector.norm(), 4147.69457577)

# Finally, we can save and plot the solutions::

# Save solution in XDMF format
with XDMFFile(MPI.comm_world, "velocity.xdmf") as ufile_xdmf:
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ufile_xdmf.write(u)

with XDMFFile(MPI.comm_world, "pressure.xdmf") as pfile_xdmf:
    p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    pfile_xdmf.write(p)

# # Plot solution
# plt.figure()
# plot(u, title="velocity")
# plot(p, title="pressure" + str(MPI.rank(mesh.mpi_comm())))

# # Display plots
# plt.show()


print("----------------------")

# Solve same problem, but now with monolithic matrices and iterative solvers

A = dolfin.fem.create_matrix_block(a)
dolfin.fem.assemble_matrix_block(A, a, bcs)
A.assemble()

P = dolfin.fem.create_matrix_block(prec)
dolfin.fem.assemble_matrix_block(P, prec, bcs)
P.assemble()

b = dolfin.fem.create_vector_block(L)
b.set(0.0)
dolfin.fem.assemble.assemble_vector_block(b, L, a, bcs)

# Set near null space for pressure

# FIXME: using createVecRight doesn't add ghosts, which breaks at the
# scatter_local_vectors step, which assumes that vectors are ghosted.
# null_vec = A.createVecRight()
null_vec = b.copy()
Vsize = V.dofmap.index_map.block_size * (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
xu = np.zeros(Vsize)
xp = np.ones(Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts)
dolfin.cpp.la.scatter_local_vectors(null_vec, [xu, xp], [V.dofmap.index_map, Q.dofmap.index_map])
null_vec.normalize()
nsp = PETSc.NullSpace().create(False, [null_vec])
A.setNullSpace(nsp)
assert np.isclose((A * null_vec).norm(), 0.0)

# Build IndexSets for each field (global dof indices for each field)
V_map = V.dofmap.index_map
Q_map = Q.dofmap.index_map
proc_offset_u, _ = V_map.local_range
proc_offset_p, _ = Q_map.local_range

offset_u = proc_offset_u * V_map.block_size + proc_offset_p
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
# x = A.createVecRight()
x = b.copy()
x.set(0.0)
ksp.solve(b, x)

# u, p = Function(V), Function(Q)
# u_local, p_local = dolfin.cpp.la.get_local_vectors(x, [V.dofmap.index_map, Q.dofmap.index_map])
# u.vector.array = u_local
# p.vector.array = p_local

# We can calculate the :math:`L^2` norms of u and p as follows::

print("RR Norm of whole solution vector: {}".format(ref))
print("BB Norm of whole solution vector: {}".format(x.norm()))
# print("XX Norm of velocity coefficient vector: {}".format(u.vector.norm()))
# print("XX Norm of pressure coefficient vector: {}".format(p.vector.norm()))


print("----------------------")

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
# x = A.createVecRight()
x = b.copy()
x.set(0.0)
ksp.solve(b, x)

# u, p = Function(V), Function(Q)
# u_local, p_local = dolfin.cpp.la.get_local_vectors(x, [V.dofmap.index_map, Q.dofmap.index_map])
# u.vector.array = u_local
# p.vector.array = p_local

# We can calculate the :math:`L^2` norms of u and p as follows::

print("RR Norm of whole solution vector: {}".format(ref))
print("DD Norm of whole solution vector: {}".format(x.norm()))
