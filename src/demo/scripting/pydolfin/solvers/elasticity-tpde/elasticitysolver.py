from math import *
from dolfin import *
from elasticitypde import *

class Source(Function):
    def eval(self, point, i):
        if(i == 1):
            return -2.0
        else:
            return 0.0

class InitialDisplacement(Function):
    def eval(self, point, i):
        return 0.0

class InitialVelocity(Function):
    def eval(self, point, i):
        if(i == 1 and point.x > 0.0):
            return 1.0
        else:
            return 0.0
        
class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 0.0:
            value.set(0.0)
        return value

f = Source()
u0 = InitialDisplacement()
v0 = InitialVelocity()

bc = SimpleBC()

mesh = Mesh("tetmesh-4.xml.gz")
#mesh = Mesh("cell.xml.gz")

E = 20.0 # Young's modulus
nu = 0.3 # Poisson's ratio

lmbdaval = E * nu / ((1 + nu) * (1 - 2 * nu))
muval = E / (2 * (1 + nu))

lmbda = Function(lmbdaval)
mu = Function(muval)

k = 1e-3
T = 5.0

t = doublep()
t.assign(0.0)
f.sync(t)


set("ODE method", "dg");
set("ODE order", 0);
set("ODE nonlinear solver", "fixed-point");
set("ODE linear solver", "direct");
set("ODE tolerance", 1.0e3);
set("ODE discrete tolerance", 1.0e3);

set("ODE fixed time step", True);
set("ODE initial time step", 2.0e-2);
set("ODE maximum time step", 2.0e-2);

set("ODE save solution", False);
set("ODE solution file name", "primal.py");
set("ODE number of samples", 100);

k = get("ODE initial time step")

pde = ElasticityPDE(mesh, f, lmbda, mu, u0, v0, bc, k, T, t)

pde.solve(pde.U)

