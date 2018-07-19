"""This demo uses PETSc's TAO solver for nonlinear (bound-constrained)
optimisation problems to solve a buckling problem in FEniCS.
We consider here a hyperelastic beam constrained in a box
under axial compression.

The box is designed such that this beam will lose stability and move
upwards (and not downwards) in order to minimise the potential energy."""

# Copyright (C) 2014 Tianyi Li
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *
from dolfin.io import XDMFFile
import matplotlib.pyplot as plt


# Read mesh and refine once
mesh = Mesh("../buckling.xml.gz")
mesh = refine(mesh)

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create solution, trial and test functions
u, du, v = Function(V), TrialFunction(V), TestFunction(V)

# Elasticity parameters
E, nu = 10.0, 0.3
mu = Constant(E/(2.0*(1.0+nu)))
lmbda = Constant(E*nu/((1.0+nu)*(1.0-2.0*nu)))

# Compressible neo-Hookean model
I = Identity(mesh.geometry.dim)
F = I + grad(u)
C = F.T*F
Ic = tr(C)
J  = det(F)
psi = (mu/2)*(Ic-2)-mu*ln(J)+(lmbda/2)*(ln(J))**2

# Surface force
f = Constant((-0.08, 0.0))

# The displacement u must be such that the current configuration
# doesn't escape the box [xmin, xmax] x [ymin, ymax]
constraint_u = Expression(("xmax-x[0]", "ymax-x[1]"), xmax=10.0, ymax=2.0, degree=1)
constraint_l = Expression(("xmin-x[0]", "ymin-x[1]"), xmin=0.0, ymin=-0.2, degree=1)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)

# Symmetry condition (to block rigid body rotations)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 10)
boundaries = MeshFunction("size_t", mesh, mesh.topology.dim-1)
boundaries.set_all(0)
left = Left()
left.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
bc = DirichletBC(V, Constant([0.0, 0.0]), boundaries, 1)
bc.apply(u_min.vector())
bc.apply(u_max.vector())

# Variational formulation
elastic_energy = psi*dx - dot(f, u)*ds(2)
grad_elastic_energy = derivative(elastic_energy, u, v)
H_elastic_energy = derivative(grad_elastic_energy, u, du)

# Define the minimisation problem by using OptimisationProblem class
class BucklingProblem(OptimisationProblem):

    def __init__(self):
        OptimisationProblem.__init__(self)

    # Objective function
    def f(self, x):
        u.vector()[:] = x
        return assemble(elastic_energy)

    # Gradient of the objective function
    def F(self, b, x):
        u.vector()[:] = x
        assemble(grad_elastic_energy, tensor=b)

    # Hessian of the objective function
    def J(self, A, x):
        u.vector()[:] = x
        assemble(H_elastic_energy, tensor=A)

# Create the PETScTAOSolver
solver = PETScTAOSolver()

# Set some parameters
solver.parameters["method"] = "tron"
solver.parameters["monitor_convergence"] = True
solver.parameters["report"] = True

# Uncomment this line to see the available parameters
# info(parameters, True)

# Parse (PETSc) parameters
parameters.parse()

# Solve the problem
solver.solve(BucklingProblem(), u.vector(), u_min.vector(), u_max.vector())

# Save solution in XDMF format if available
out = XDMFFile(mesh.mpi_comm(), "u.xdmf")
if has_hdf5():
    out.write(u)
elif MPI.size(mesh.mpi_comm()) == 1:
    encoding = XDMFFile.Encoding.ASCII
    out.write(u, encoding)
else:
    # Save solution in vtk format
    out = File("u.pvd")
    out << u

# Plot the current configuration
plot(u, mode="displacement", wireframe=True, title="Displacement field")
plt.show()
