from dolfin import *
from diffpde import *

class Source(Function):
    def eval(self, point, i):
        if(i == 1):
            return -10.0
        else:
            return 0.0

class Density(Function):
    def eval(self, point, i):
        return 3.0e0

class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 0.0:
            value.set(0.0)

class InitialVelocity(Function):
    def eval(self, point, i):
        return 0.0


set("ODE method", "dg");
set("ODE order", 0);
set("ODE nonlinear solver", "fixed-point");
set("ODE linear solver", "direct");
set("ODE tolerance", 1.0e3);
set("ODE discrete tolerance", 1.0e3);

set("ODE fixed time step", True);
set("ODE initial time step", 1.0e-2);
set("ODE maximum time step", 1.0e-2);

set("ODE save solution", False);
set("ODE solution file name", "primal.py");
set("ODE number of samples", 100);


nu  = 0.3 # Poisson's ratio
E   = 500.0 * 0.7 * 1.0 * 3.0 # Young's modulus
nuv = 1.0 * 1.0e2 * 1.0e-3 * 100.0 * 1.0 # Viscosity

yld = 4.0e3 * 1.0e-3 # Yield strength
nuplast = 1.0e-4 * 1.0e3 * 4.0e-1 # Plastic viscosity

import geometry

T = 10.0
#T = 1.0e-4 / 1.0
#k = 1.0e-3 / 4.0
#k = 1.0e-3 * 3.333
k = 3.333e-3 * 3


#load_parameters("parameters.xml")

# Coefficients

f = Source()
rho = Density()
v0 = InitialVelocity()
bc = UtilBC1()

pde = DiffPDE(geometry.mesh, f, E, nu, nuv, bc, T, v0, rho)
pde.solve(pde.U)

print "fcount: ", pde.fcount
