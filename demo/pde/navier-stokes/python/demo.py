"""This demo program solves the incompressible Navier-Stokes equations
on an L-shaped domain using Chorin's splitting method."""

__author__ = "Anders Logg <logg@simula.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2010-08-30

from dolfin import *

# Load mesh from file
mesh = Mesh("lshape.xml.gz")

# Refine mesh
mesh = refine(mesh)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
v = TestFunction(V)
q = TestFunction(Q)
u = TrialFunction(V)
p = TrialFunction(Q)

# Set parameter values
dt = 0.01
T = 3
nu = 0.01

# Define time-dependent pressure boundary condition
p_in = Expression("sin(3.0*t)")

# Define boundary conditions
noslip  = DirichletBC(V, (0, 0), "on_boundary && x[1] < 1.0 - DOLFIN_EPS && x[0] < 1.0 - DOLFIN_EPS")
inflow  = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
bcu = [noslip]
bcp = [inflow, outflow]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1/k)*inner(v, u - u0)*dx + inner(v, grad(u0)*u0)*dx \
    + nu*inner(grad(v), grad(u))*dx - inner(v, f)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(q), grad(p))*dx
L2 = -(1/k)*q*div(u1)*dx

# Velocity update
a3 = inner(v, u)*dx
L3 = inner(v, u1)*dx - k*inner(v, grad(p1))*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Create files for storing solution
ufile = File("velocity.pvd")
pfile = File("pressure.pvd")

# Time-stepping
t = dt
p = Progress("Time-stepping")
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    p_in.t = t

    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "ilu")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "gmres", "amg_hypre")
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "ilu")
    end()

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True)

    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1)
    p.update(t / T)
    t += dt

# Hold plot
interactive()
