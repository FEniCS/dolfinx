"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2010-08-08"
__copyright__ = "Copyright (C) 2010 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Test for PETSc and SLEPc
if not has_la_backend("PETSc") or not has_la_backend("Epetra"):
    print "DOLFIN has not been configured with Trilinos or PETSc. Exiting."
    exit()

print "This demo is unlikely to converge if PETSc is not configured with Hypre or ML."

# Load mesh and subdomains
mesh = UnitCube(16, 16, 16)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, right)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, left)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(v, q) = TestFunctions(W)
(u, p) = TrialFunctions(W)
f = Constant((0.0, 0.0, 0.0))
a = inner(grad(v), grad(u))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(v, f)*dx

# Form for use in constructing preconditioner matrix
b = inner(grad(v), grad(u))*dx + q*p*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

# Create Krylov solver and AMG preconditioner
solver = KrylovSolver("tfqmr", "amg_ml")

# Associate opeartor (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
