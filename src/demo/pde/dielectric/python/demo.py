# This demo program calculates the electrostatic potential
# in the unit square by solving Laplace's equation
#
#      div \epsilon_r grad u(x, y) = 0
#
# The lower half is filled with a dielectric material
# with dielectric constant \epsilon_r.
#
# The boundary conditions are:
#
#     u(x, y)     = 0  for y > 0
#     u(x,y)      = V  for y = 0

__author__ = "Kristen Kaasbjerg (cosby@fys.ku.dk)"
__date__ = "2008-02-14"
__copyright__ = ""
__license__  = "GNU LGPL Version 2.1"


from dolfin import *

l = 1. ; h = 1. # unit square 
h_ = .5 # position of the dielectric interface
e_r = 10 # dielectric constant for y<h_ (e_r=1 for y>h_
V = 1. # applied voltage at the y=0 boundary

# Create mesh and finite element
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh,"triangle", 2,2) 
editor.initVertices(4)
editor.initCells(2)
editor.addVertex(0, 0.0, 0.0)
editor.addVertex(1, 0.0, h)
editor.addVertex(2, l, h)
editor.addVertex(3, l,0)
editor.addCell(0, 0, 1, 2)
editor.addCell(1, 0, 2, 3)
editor.close()

# Refine mesh
for refine in range(4):
    mesh.refine()

# Finite elements
element = FiniteElement("Lagrange", "triangle", 2) # last argument: order of the polynomial
# Discontinuous element needed for the dielectric function
DG0 = FiniteElement("DG", "triangle", 0) # 0'th order are possible for DG elements

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        values[0] = 0.0

# Exact solution
class Exact(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        u = 0
        N = 20 # N needs to be rather large (~200) in other to have u=V for y=0 !
        pi = DOLFIN_PI
        if x[1]<=h_:
            for n in range(N):
                n_ = 2*n+1 #n>0!
                X = (1.-exp(2*n_*pi*(1-h_)))/(1+exp(2*n_*pi*(1-h_)))
                u -= sin(n_*pi*x[0])*4*V/(pi*n_) * ( (1+e_r*X)*exp(n_*pi*(x[1]-h_))+
                                                     (-1.+e_r*X)*exp(-n_*pi*(x[1]-h_)) ) /\
                                                     ( (1.-e_r*X)*exp(n_*pi*h_) -
                                                       (1.+e_r*X)*exp(-n_*pi*h_))
        else:
            for n in range(N):
                n_ = 2*n+1 #n>0!
                X = (1.-exp(2*n_*pi*(1-h_)))/(1+exp(2*n_*pi*(1-h_)))
                X1 = 1./(1+exp(2*n_*pi*(1-h_)))
                X2 = exp(2*n_*pi*(1-h_))/(1+exp(2*n_*pi*(1-h_)))
                u += sin(n_*pi*x[0])*4*V/(pi*n_) * ( 2*e_r*X1*exp(n_*pi*(x[1]-h_)) -
                                                     2*e_r*X2*exp(-n_*pi*(x[1]-h_))) /\
                                                     ( (1.+e_r*X)*exp(-n_*pi*h_) -
                                                       (1.-e_r*X)*exp(n_*pi*h_) )
        values[0] = u 
        
# Dielectric constant
class Coefficient(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        if x[1] <= h_:
            values[0] = e_r
        else:
            values[0] = 1.0

# Dirichlet boundary condition
class DirichletFunction(Function):
    def __init__(self, element,mesh):
        Function.__init__(self, element,mesh)
    def eval(self, values, x):
        if x[1] < DOLFIN_EPS:
            values[0] = V
        else:
            values[0] = 0.0
            
# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and True)

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
f = Source(element,mesh)
b = Coefficient(DG0, mesh)

a = b*dot(grad(v), grad(u))*dx
L = v*f*dx

# Define boundary condition
u0 = DirichletFunction(element,mesh)
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)

# Solve PDE and plot solution
pde = LinearPDE(a, L, mesh, bc)
u = pde.solve()

plot(u)

#Calculate difference between exact and FEM solution
# Use higher order element for exact solution,
# because it is an interpolation of the exact solution
# in the finite element space!
Pk = FiniteElement("Lagrange", "triangle", 5)
exact = Exact(Pk,mesh) 

e = u - exact
L2_norm = e*e*dx
norm = assemble(L2_norm,mesh)

print "L2-norm is: ", norm








