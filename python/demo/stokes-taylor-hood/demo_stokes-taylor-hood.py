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
# * :math:`\Omega = [0,1]\times[0,1] \backslash {\rm dolphin}` (a unit cube)
# * :math:`\Gamma_D =`
# * :math:`\Gamma_N =`
# * :math:`u_0 = (- \sin(\pi x_1), 0.0)` for :math:`x_0 = 1` and :math:`u_0 = (0.0, 0.0)` otherwise
# * :math:`f = (0.0, 0.0)`
# * :math:`g = (0.0, 0.0)`
#
#
# Implementation
# --------------
#
# In this example, different boundary conditions are prescribed on
# different parts of the boundaries. Each sub-regions is tagged with
# different (integer) labels. For this purpose, DOLFIN provides
# a :py:class:`MeshFunction <dolfin.cpp.mesh.MeshFunction>` class
# representing functions over mesh entities (such as over cells or over
# facets). Mesh and mesh functions can be read from file in the
# following way::

from petsc4py import PETSc

import matplotlib.pyplot as plt
import numpy as np

import dolfin
from dolfin import (MPI, DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction)
from dolfin.io import XDMFFile
from dolfin.plotting import plot
from ufl import (FiniteElement, VectorElement,
                 div, dx, grad, inner)

# Load mesh and subdomains
xdmf = XDMFFile(MPI.comm_world, "../dolfin_fine.xdmf")
mesh = xdmf.read_mesh(dolfin.cpp.mesh.GhostMode.none)

sub_domains = xdmf.read_mf_size_t(mesh)

cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
mesh.geometry.coord_mapping = cmap

# Next, we define a :py:class:`FunctionSpace
# <dolfin.functions.functionspace.FunctionSpace>` built on a mixed
# finite element ``TH`` which consists of continuous
# piecewise quadratics and continuous piecewise
# linears::

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, P2)
Q = FunctionSpace(mesh, P1)

# The mixed finite element space is known as Taylor–Hood.
# It is a stable, standard element pair for the Stokes
# equations. Now we can define boundary conditions::


# Extract subdomain facet arrays
mf = sub_domains.values
mf0 = np.where(mf == 0)
mf1 = np.where(mf == 1)

# No-slip boundary condition for velocity
# x1 = 0, x1 = 1 and around the dolphin
noslip = Function(V)
noslip.interpolate(lambda x: np.zeros_like(x))

bc0 = DirichletBC(V, noslip, mf0[0])

# Inflow boundary condition for velocity
# x0 = 1


def inflow_eval(x):
    values = np.zeros((2, x.shape[1]))
    values[0] = - np.sin(x[1] * np.pi)
    return values


inflow = Function(V)
inflow.interpolate(inflow_eval)
bc1 = DirichletBC(V, inflow, mf1[0])

# Collect boundary conditions
bcs = [bc0, bc1]

# The first argument to
# :py:class:`DirichletBC <dolfin.cpp.fem.DirichletBC>`
# specifies the :py:class:`FunctionSpace
# <dolfin.cpp.function.FunctionSpace>`. Since we have a
# mixed function space, we write
# ``W.sub(0)`` for the velocity component of the space, and
# ``W.sub(1)`` for the pressure component of the space.
# The second argument specifies the value on the Dirichlet
# boundary. The last two arguments specify the marking of the subdomains:
# ``sub_domains`` contains the subdomain markers, and the final argument is the subdomain index.
#
# The bilinear and linear forms corresponding to the weak mixed
# formulation of the Stokes equations are defined as follows::

# Define variational problem
(u, p) = TrialFunction(V), TrialFunction(Q)
(v, q) = TestFunction(V), TestFunction(Q)
f = Function(V)
g = Function(Q)

a = [[inner(grad(u), grad(v)) * dx,  inner(p, div(v)) * dx],
     [inner(div(u), q) * dx, None]]

prec = [[inner(grad(u), grad(v)) * dx, None],
        [None, p * q * dx]]

L = [inner(f, v) * dx, dolfin.Constant(mesh, 0) * q * dx]

# We also need to create a :py:class:`Function
# <dolfin.cpp.function.Function>` to store the solution(s). The (full)
# solution will be stored in ``w``, which we initialize using the mixed
# function space ``W``. The actual
# computation is performed by calling solve with the arguments ``a``,
# ``L``, ``w`` and ``bcs``. The separate components ``u`` and ``p`` of
# the solution can be extracted by calling the :py:meth:`split
# <dolfin.functions.function.Function.split>` function. Here we use an
# optional argument True in the split function to specify that we want a
# deep copy. If no argument is given we will get a shallow copy. We want
# a deep copy for further computations on the coefficient vectors::

# Compute solution
x = dolfin.fem.create_vector_nest(L)

A = dolfin.fem.create_matrix_nest(a)
dolfin.fem.assemble_matrix_nest(A, a, bcs)
A.assemble()

P = dolfin.fem.create_matrix_nest(prec)
dolfin.fem.assemble_matrix_nest(P, prec, bcs)
P.assemble()

b = dolfin.fem.create_vector_nest(L)
b.zeroEntries()

dolfin.fem.assemble.assemble_vector_nest(b, L)
dolfin.fem.assemble.apply_lifting_nest(b, a, bcs)
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = dolfin.cpp.fem.bcs_rows(dolfin.fem.assemble._create_cpp_form(L), bcs)
dolfin.fem.assemble.set_bc_nest(b, bcs0)

b.assemble()

# -- Nullspace
x_nullspace = dolfin.fem.create_vector_nest(L)
x_nullspace.zeroEntries()
x_nullspace.getNestSubVecs()[1].set(1.0)
x_nullspace.getNestSubVecs()[1].array /= x_nullspace.getNestSubVecs()[1].norm()
x_nullspace.getNestSubVecs()[1].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
nullspace = PETSc.NullSpace().create(vectors=(x_nullspace,), comm=mesh.mpi_comm())
A.setNullSpace(nullspace)
P.setNullSpace(nullspace)

# -- Near nullspace
rbm_functions = [lambda x: np.row_stack((np.ones_like(x[1]), np.zeros_like(x[1]))),
                 lambda x: np.row_stack((np.zeros_like(x[1]), np.ones_like(x[1]))),
                 lambda x: np.row_stack((-x[1], x[0]))]

rbms = [Function(V) for _ in range(3)]

for (rbm, rbm_function) in zip(rbms, rbm_functions):
    rbm.interpolate(rbm_function)

rbms = dolfin.cpp.la.VectorSpaceBasis(list(map(lambda rbm: rbm.vector, rbms)))
rbms.orthonormalize()

near_nullspace = PETSc.NullSpace().create(vectors=tuple(rbms[i] for i in range(3)), comm=mesh.mpi_comm())
A.getNestSubMatrix(0, 0).setNearNullSpace(near_nullspace)
P.getNestSubMatrix(0, 0).setNearNullSpace(near_nullspace)

ksp = PETSc.KSP().create(mesh.mpi_comm())

ksp.setOperators(A, P)

ksp.setTolerances(rtol=1e-8)
ksp.setType("fgmres")
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

nested_IS = A.getNestISs()
ksp.getPC().setFieldSplitIS(
    ("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

# ksp.getPC().setUp()
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.setTolerances(rtol=1e-3)
ksp_u.getPC().setType("gamg")
ksp_u.getPC().setGAMGType(PETSc.PC.GAMGType.AGG)
ksp_p.setType("cg")
ksp_p.setTolerances(rtol=1e-3)
ksp_p.getPC().setType("hypre")

opts = PETSc.Options()
opts["ksp_monitor"] = None
opts["ksp_monitor_lg_residualnorm"] = None
ksp.setFromOptions()

ksp.solve(b, x)

ksp.view()
# We can calculate the :math:`L^2` norms of u and p as follows::

u, p = Function(V), Function(Q)
(u.vector.array, p.vector.array) = \
    map(lambda vec: vec.array_r, x.getNestSubVecs())

print("Norm of velocity coefficient vector: %.15g" % u.vector.norm())
print("Norm of pressure coefficient vector: %.15g" % p.vector.norm())

# Check pressure norm
assert np.isclose(p.vector.norm(), 4147.69457577)

# Finally, we can save and plot the solutions::

# Save solution in XDMF format
with XDMFFile(MPI.comm_world, "velocity.xdmf") as ufile_xdmf:
    ufile_xdmf.write(u)

with XDMFFile(MPI.comm_world, "pressure.xdmf") as pfile_xdmf:
    pfile_xdmf.write(p)

# Plot solution
plt.figure()
plot(u, title="velocity")

# plt.figure()
plot(p, title="pressure" + str(MPI.rank(mesh.mpi_comm())))

# Display plots
plt.show()
