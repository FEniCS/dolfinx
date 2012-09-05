# An example of use of the interface to TAO to solve a contact mechanics problems in FEnics.
# 
# The example considers a heavy elastic sphere in contact with a rigid half space (at the bottom)
#
# Corrado Maurini 03/09/2012
#
from dolfin import *
parameters["allow_extrapolation"]=True

# Create mesh
mesh = UnitCircle(30)

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create test and trial functions, and source term
u, w = TrialFunction(V), TestFunction(V)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 -2.0*nu))
b = Constant((0.,-.2))

# Stress and strains
def eps(u):
    return sym(grad(u))

def sigma(epsilon):
    return  2*mu*epsilon + lmbda*tr(epsilon)*Identity(w.cell().d)

# Weak formulation
F = inner(sigma(eps(u)), eps(w))*dx - dot(b, w)*dx

# Extract bilinear and linear forms from F
a, L = lhs(F), rhs(F)

# Boundary condition (null horizontal displacement of the center)
tol=0.001
def center(x):
    return x[0]**2+x[1]**2 < tol**2
bc = DirichletBC(V.sub(0), 0., center,method="pointwise")

# Assemble the linear system
A, b = assemble_system(a, L, bc)

# Define the constraints
constraint_u = Expression( ("xmax-x[0]","ymax-x[1]"), xmax =  2., ymax =  2.)
constraint_l = Expression( ("xmin-x[0]","ymin-x[1]"), xmin = -2., ymin = -1.)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)

# Define the function to store the solution 
usol=Function(V)

# Create the TAOLinearBoundSolver
solver=TAOLinearBoundSolver("tao_tron","gmres")

#Set some parameters
solver.parameters["monitor_convergence"]=True
solver.parameters["report"]=True
solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-8
solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-8
solver.parameters["krylov_solver"]["monitor_convergence"]= False
#info(solver.parameters,True)

# Solve the problem
solver.solve(A, usol.vector(), b , u_min.vector(), u_max.vector())

# Calculate the contact area
delta=project(usol-u_min,V)

class ContactZone(SubDomain):
    def inside(self, x, on_boundary):
        return (delta(x))[1] < 0.001 and on_boundary

sub_domains = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
sub_domains.set_all = 0
contact_zone = ContactZone()
contact_zone.mark(sub_domains,5)
dsb = ds[sub_domains]
one = interpolate(Constant(1.),FunctionSpace(mesh,'CG',1))
area = assemble(one*dsb(5))
print "The contact area is ", area

# Save solution in VTK format
file = File("displacement.pvd")
file << usol

# plot the stress
stress=sigma(eps(usol))
plot(sqrt(inner(stress,stress)))

# plot the current configuration
plot(usol, mode = "warp",wireframe=False, title="Displacement field")
interactive()