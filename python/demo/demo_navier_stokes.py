# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # A divergence conforming discontinuous Galerkin method for the Navier-Stokes equations
# This demo illustrates how to implement a divergence conforming
# discontinuous Galerkin method for the Navier-Stokes equations in
# FEniCSx. The method conserves mass exactly and uses upwinding. The
# formulation is inspired by a combination of "A fully divergence-free
# finite element method for magnetohydrodynamic equations" by Himpmair
# et al., "A Note on Discontinuous Galerkin Divergence-free Solutions
# of the Navier-Stokes Equations" by Cockburn et al, and "On the Divergence
# Constraint in Mixed Finite Element Methods for Incompressible Flows" by
# John et al.

# ## Governing equations
# We consider incompressible Navier-Stokes equations in a domain
# $\Omega \subset \mathbb{R}^d$, $d \in \{2, 3\}$, and time interval
# $(0, \infty)$, given by
# $$
#     \partial_t u - \nu \Delta u + (u \cdot \nabla)u + \nabla p = f
#     \textnormal{ in } \Omega_t,
# $$
# $$
#     \nabla \cdot u = 0
#     \textnormal{ in } \Omega_t,
# $$
# where $u: \Omega_t \to \mathbb{R}^d$ is the velocity field,
# $p: \Omega_t \to \mathbb{R}$ is the pressure field,
# $f: \Omega_t \to \mathbb{R}^d$ is a prescribed force, $\nu \in \mathbb{R}^+$
# is the kinematic viscosity, and
# $\Omega_t \coloneqq \Omega \times (0, \infty)$.

# The problem is supplemented with the initial condition
# $$
#     u(x, 0) = u_0(x) \textnormal{ in } \Omega
# $$
# and boundary condition
# $$
#     u = u_D \textnormal{ on } \partial \Omega \times (0, \infty),
# $$
# where $u_0: \Omega \to \mathbb{R}^d$ is a prescribed initial velocity field
# which satisfies the divergence free condition. The pressure field is only
# determined up to a constant, so we seek the unique pressure field satisfying
# $$
#     \int_\Omega p = 0.
# $$

# ## Discrete problem
# We begin by introducing the function spaces
# $$
#     V_h^g \coloneqq \left\{v \in H(\textnormal{div}; \Omega);
#     v|_K \in V_h(K) \; \forall K \in \mathcal{T}, v \cdot n = g \cdot n
#     \textnormal{ on } \partial \Omega \right\}
# $$,
# and
# $$
#     Q_h \coloneqq \left\{q \in L^2_0(\Omega);
#     q|_K \in Q_h(K) \; \forall K \in \mathcal{T} \right\}.
# $$
# The local spaces $V_h(K)$ and $Q_h(K)$ should satisfy
# $$
#     \nabla \cdot V_h(K) \subseteq Q_h(K),
# $$
# in order to conserve mass exactly. Suitable choices on affine simplex cells
# include
# $$
#     V_h(K) \coloneqq \mathbb{RT}_k(K) \textnormal{ and }
#     Q_h(K) \coloneqq \mathbb{P}_k(K),
# $$
# or
# $$
#     V_h(K) \coloneqq \mathbb{BDM}_k(K) \textnormal{ and }
#     Q_h(K) \coloneqq \mathbb{P}_{k-1}(K).
# $$

# Let two cells $K^+$ and $K^-$ share a facet $F$. The trace of a piecewise
# smooth vector valued function $\phi$ on F taken approaching from inside $K^+$
# (resp. $K^-$) is denoted $\phi^{+}$ (resp. $\phi^-$). We now introduce the
# average
# $\renewcommand{\avg}[1]{\left\{\!\!\left\{#1\right\}\!\!\right\}}$
# $$
#     \avg{\phi} = \frac{1}{2} \left(\phi^+ + \phi^-\right)
# $$
# $\renewcommand{\jump}[1]{\llbracket #1 \rrbracket}$
# and jump
# $$
#     \jump{\phi} = \phi^+ \otimes n^+ + \phi^- \otimes n^-,
# $$
# operators, where $n$ denotes the outward unit normal to $\partial K$.
# Finally, let the upwind flux of $\phi$ with respect to a vector field
# $\psi$ be defined as
# $$
#     \hat{\phi}^\psi \coloneqq
#     \begin{cases}
#         \lim_{\epsilon \downarrow 0} \phi(x - \epsilon \psi(x)), \;
#         x \in \partial K \setminus \Gamma^\psi, \\
#         0, \qquad \qquad \qquad \qquad x \in \partial K \cap \Gamma^\psi,
#     \end{cases}
# $$
# where $\Gamma^\psi = \left\{x \in \Gamma; \; \psi(x) \cdot n(x) < 0\right\}$.

# The semi-discrete version problem is: find $(u_h, p_h) \in V_h^{u_D} \times Q_h$
# such that
# $$
#     \int_\Omega \partial_t u_h \cdot v + a_h(u_h, v_h) + c_h(u_h; u_h, v_h)
#     + b_h(v_h, p_h) = \int_\Omega f \cdot v_h + L_{a_h}(v_h) + L_{c_h}(v_h)
#      \quad \forall v_h \in V_h^0,
# $$
# $$
#     b_h(u_h, q_h) = 0 \quad \forall q_h \in Q_h,
# $$
# where
# $\renewcommand{\sumK}[0]{\sum_{K \in \mathcal{T}_h}}$
# $\renewcommand{\sumF}[0]{\sum_{F \in \mathcal{F}_h}}$
# $$
#     a_h(u, v) = Re^{-1} \left(\sumK \int_K \nabla u : \nabla v
#     - \sumF \int_F \avg{\nabla u} : \jump{v}
#     - \sumF \int_F \avg{\nabla v} : \jump{u} \\
#     + \sumF \int_F \frac{\alpha}{h_K} \jump{u} : \jump{v}\right),
# $$
# $$
#     c_h(w; u, v) = - \sumK \int_K u \cdot \nabla \cdot (v \otimes w)
#     + \sumK \int_{\partial_K} w \cdot n \hat{u}^{w} \cdot v,
# $$
# $$
# L_{a_h}(v_h) = Re^{-1} \left(- \int_{\partial \Omega} u_D \otimes n :
#   \nabla_h v_h + \frac{\alpha}{h} u_D \otimes n : v_h \otimes n \right),
# $$
# $$
#     L_{c_h}(v_h) = - \int_{\partial \Omega} u_D \cdot n \hat{u}_D \cdot v_h,
# $$
# and
# $$
#     b_h(v, q) = - \int_K \nabla \cdot v q.
# $$

from dolfinx import mesh, fem, io
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, dS, ds, avg, outer, div, conditional,
                 gt, dot)


def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)


def u_e_expr(x):
    """Expression for the exact velocity solution to Kovasznay flow"""
    return np.vstack((1 - np.exp(
        (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2)) * x[0])
        * np.cos(2 * np.pi * x[1]),
        (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2))
        / (2 * np.pi) * np.exp(
            (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2)) * x[0])
        * np.sin(2 * np.pi * x[1])))


def p_e_expr(x):
    """Expression for the exact pressure solution to Kovasznay flow"""
    return (1 / 2) * (1 - np.exp(
        2 * (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2)) * x[0]))


def f_expr(x):
    """Expression for the applied force"""
    return np.vstack((np.zeros_like(x[0]),
                      np.zeros_like(x[0])))


def boundary_marker(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | \
        np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


# Simulation parameters
n = 16
num_time_steps = 25
t_end = 10
R_e = 25  # Reynolds Number
k = 1  # Polynomial degree

msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

# Function space for the velocity
V = fem.FunctionSpace(msh, ("Raviart-Thomas", k + 1))
# Function space for the pressure
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
# Funcion space for visualising the velocity field
W = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(msh, PETSc.ScalarType(6.0 * k**2))
R_e_const = fem.Constant(msh, PETSc.ScalarType(R_e))

h = CellDiameter(msh)
n = FacetNormal(msh)


def jump(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


# We solve the Stokes problem for the initial condition
a_00 = 1 / R_e_const * (inner(grad(u), grad(v)) * dx
                        - inner(avg(grad(u)), jump(v, n)) * dS
                        - inner(jump(u, n), avg(grad(v))) * dS
                        + alpha / avg(h) * inner(jump(u, n), jump(v, n)) * dS
                        - inner(grad(u), outer(v, n)) * ds
                        - inner(outer(u, n), grad(v)) * ds
                        + alpha / h * inner(outer(u, n), outer(v, n)) * ds)
a_01 = - inner(p, div(v)) * dx
a_10 = - inner(div(u), q) * dx
a_11 = fem.Constant(msh, PETSc.ScalarType(0.0)) * inner(p, q) * dx

a = fem.form([[a_00, a_01],
              [a_10, a_11]])

f = fem.Function(W)
u_bc = fem.Function(V)
u_bc.interpolate(u_e_expr)
L_0 = inner(f, v) * dx + \
    1 / R_e_const * (- inner(outer(u_bc, n), grad(v)) * ds
                     + alpha / h * inner(outer(u_bc, n), outer(v, n)) * ds)
L_1 = inner(fem.Constant(msh, PETSc.ScalarType(0.0)), q) * dx

L = fem.form([L_0,
              L_1])

# Boundary conditions
boundary_facets = mesh.locate_entities_boundary(
    msh, msh.topology.dim - 1, boundary_marker)
boundary_vel_dofs = fem.locate_dofs_topological(
    V, msh.topology.dim - 1, boundary_facets)
bc_u = fem.dirichletbc(u_bc, boundary_vel_dofs)

# The pressure is only determined up to a constant, so pin a single degree
# of freedom
# TODO TIDY
pressure_dofs = fem.locate_dofs_geometrical(
    Q, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                np.isclose(x[1], 0.0)))
if len(pressure_dofs) > 0:
    pressure_dof = [pressure_dofs[0]]
else:
    pressure_dof = []
bc_p = fem.dirichletbc(PETSc.ScalarType(0.0),
                       np.array(pressure_dof, dtype=np.int32),
                       Q)

bcs = [bc_u, bc_p]

# Assemble Stokes problem
A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Create and configure solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
# See https://graal.ens-lyon.fr/MUMPS/doc/userguide_5.5.1.pdf
# TODO Check
opts["mat_mumps_icntl_6"] = 2
opts["mat_mumps_icntl_14"] = 100
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

x = A.createVecRight()

# Solve Stokes for initial condition
ksp.solve(b, x)

# Split the solution
u_h = fem.Function(V)
p_h = fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()

u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file
u_file = io.VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
p_file = io.VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])

t = 0.0
u_file.write(t)
p_file.write(t)

# Solution and previous time step
u_n = fem.Function(V)
u_n.x.array[:] = u_h.x.array

# Add time stepping and convective terms
lmbda = conditional(gt(dot(u_n, n), 0), 1, 0)
u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")
a_00 += inner(u / delta_t, v) * dx - \
    inner(u, div(outer(v, u_n))) * dx + \
    inner((dot(u_n, n))("+") * u_uw, v("+")) * dS + \
    inner((dot(u_n, n))("-") * u_uw, v("-")) * dS + \
    inner(dot(u_n, n) * lmbda * u, v) * ds
a = fem.form([[a_00, a_01],
              [a_10, a_11]])

L_0 += inner(u_n / delta_t, v) * dx - \
    inner(dot(u_n, n) * (1 - lmbda) * u_bc, v) * ds
L = fem.form([L_0,
              L_1])

# Time stepping loop
for n in range(num_time_steps):
    t += delta_t.value

    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)
    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()

# Function spaces for exact velocity and pressure
V_e = fem.VectorFunctionSpace(msh, ("Lagrange", k + 3))
Q_e = fem.FunctionSpace(msh, ("Lagrange", k + 2))

u_e = fem.Function(V_e)
u_e.interpolate(u_e_expr)

p_e = fem.Function(Q_e)
p_e.interpolate(p_e_expr)

# Compute errors
e_u = norm_L2(msh.comm, u_h - u_e)
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
p_h_avg = domain_average(msh, p_h)
p_e_avg = domain_average(msh, p_e)
e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e - p_e_avg))

if msh.comm.rank == 0:
    print(f"e_u = {e_u}")
    print(f"e_div_u = {e_div_u}")
    print(f"e_p = {e_p}")
