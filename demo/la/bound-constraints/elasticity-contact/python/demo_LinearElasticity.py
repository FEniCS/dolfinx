# An example of use of the interface to TAO to solve a contact mechanics problems in FEnics.
# This is a 2D linear elastic beam with gravity with an unilateral contact condition with a basement (at x[0]=-1)
#
# Corrado Maurini 12/03/2012
#
from dolfin import *
# Create mesh
mesh = Rectangle(0,0,10,1,100,10)
# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)
# Create test and trial functions, and source term
u, w = TrialFunction(V), TestFunction(V)
b = Constant((.0, -0.01))

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 -2.0*nu))

# Stress and strains
def eps(u):
    return sym(grad(u))

def sigma(epsilon):
    return  2*mu*epsilon + lmbda*tr(epsilon)*Identity(w.cell().d)
    
# Weak formulation
F = inner(sigma(eps(u)), eps(w))*dx - dot(b, w)*dx

# Extract bilinear and linear forms from F
a, L = lhs(F), rhs(F)

# Dirichlet boundary condition 
def left_boundary(x, on_boundary):
    return on_boundary and x[0] < DOLFIN_EPS
    c = Constant((0.0, 0.0))
    
bc = DirichletBC(V, c, left_boundary)

# Assemble the linear system
A=assemble(a)
b=assemble(L)
bc.apply(A)
bc.apply(b)

# Define the constraints
lowerbound = interpolate(Expression(("(-30)-x[0]","(-1.)-x[1]")), V)
upperbound = interpolate(Expression(("(30)-x[0]","(30)-x[0]")), V)
xu=upperbound.vector()
xl=lowerbound.vector()

# Define the function to store the solution and the related vector
usol=Function(V);
xsol=usol.vector()

# Create the TAOLinearBoundSolver and solve the problem
solver=TAOLinearBoundSolver()
solver.solve(A,xsol,b,xl,xu)

# Save solution in VTK format
file = File("displacement.pvd")
file << usol
#plot(usol, mode = "displacement",wireframe=False, title="Displacement field")

