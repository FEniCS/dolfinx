""" Steady state advection-diffusion equation,
discontinuous formulation using full upwinding.

Constant velocity field with homogeneous Dirichlet boundary conditions
on all boundaries.

Implemented in python from cpp demo by Johan Hake

"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2008-06-19"
__copyright__ = "Copyright (C) 2008 Johan Hake"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

def forms(scalar, vector, constant):
    # Import local ffc to be able to define forms
    # Note: This is needed because the compiled functions that is used below are
    # only cpp_Functions not ffc_Functions, and cannot be used to define forms
    import ffc
    
    # Test and trial functions
    v  = TestFunction(scalar)
    u  = TrialFunction(scalar)
    
    # Coeffisient functions
    f  = ffc.Function(scalar)
    b  = ffc.Function(vector)    # Note: b('+') == b('-')
    of = ffc.Function(constant)  # This function is a switch that determines if a
                                 # facet is an outflow facet or not 1.0 or 0.0
    n  = ffc.FacetNormal("triangle")
    h  = ffc.MeshSize("triangle")
                            
    kappa = Constant("triangle")
    alpha = Constant("triangle")
    
    def upwind(u, b):
        return [b[i]('+')*(of('+')*u('+') + of('-')*u('-')) for i in range(len(b))]
    
    # Bilinear form
    a_int = dot( grad(v), mult(kappa, grad(u)) - mult(b,u) )*dx
    
    a_fac = kappa('+')*alpha('+')/h('+')*dot(jump(v, n), jump(u, n))*dS \
            - kappa('+')*dot(avg(grad(v)), jump(u, n))*dS \
            - kappa('+')*dot(jump(v, n), avg(grad(u)))*dS
    
    a_gd = kappa*alpha/h*v*u*ds \
           - kappa*dot(grad(v), mult(u,n))*ds \
           - kappa*dot(mult(v,n), grad(u))*ds
    
    a_vel = dot(jump(v, n), upwind(u, b))*ffc.dS + dot(mult(v, n), mult(b, of*u))*ds
    
    a = a_int + a_fac + a_gd + a_vel
    
    # Linear form
    L = v*f*dx
    
    return a, L

mesh =  UnitSquare (64, 64)

# Defining the finite element room
scalar_DG = FiniteElement("Discontinuous Lagrange", "triangle", 2)
scalar_CG = FiniteElement("Lagrange", "triangle", 2)
vector_CG = VectorElement("Lagrange", "triangle", 2)
constant  = FiniteElement("Discontinuous Lagrange", "triangle", 0)

# Defining dolfin coeffisient functions
file_string = open('functions2D.h').read()
coeffisients = compile_functions(file_string,mesh)

# Extracting the compiled functions
source   = coeffisients.pop(0)
velocity = coeffisients[0]
outflow_facet = coeffisients[1]

# Setting members of the compiled functions
outflow_facet.velocity = velocity
source.C               = 0.0

coeffisients.append(FacetNormal("triangle", mesh))
coeffisients.append(AvgMeshSize("triangle", mesh)) 
coeffisients.append(Function(constant,mesh,0.0))   # The diffusivity
coeffisients.append(Function(constant,mesh,20.0))  # The penalty term

a, L = forms(scalar_DG, vector_CG, constant)

# Assembly and solve system
A = assemble(a, mesh, coeffisients)
b = assemble(L, mesh, [source])

uh = Function(scalar_DG, mesh, Vector())

solve(A, uh.vector(), b)

# Defining projection forms 
vp = TestFunction(scalar_CG)
up = TrialFunction(scalar_CG)

u0 = Function(scalar_DG)

ap = dot(vp,up)*dx
Lp = dot(vp,uh)*dx

# Define and solve PDE
pde = LinearPDE(ap, Lp, mesh)

up = pde.solve()

file = File("temperature.pvd")
file << up

# Plot solution
plot(up, interactive=True)
