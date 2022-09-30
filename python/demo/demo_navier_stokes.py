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
a_00 = 1 / R_e_const * (inner(grad(u), grad(v)) * dx -
                        inner(avg(grad(u)), jump(v, n)) * dS
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
