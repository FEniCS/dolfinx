from math import *
from dolfin import *
from nspde import *

coeffs = import_header("Coefficients.h")

f = coeffs.MySource()
vbc = coeffs.VelocityBC()
pbc = coeffs.PressureBC()

#mesh = Mesh("tetmesh-4.xml.gz")
#mesh = Mesh("cell.xml.gz")
#mesh = UnitSquare(10, 10)
mesh = UnitSquare(10, 10)

nu = 1.0/3900.0
d1 = 0.0
d2 = 0.0

T = 5.0

set("ODE method", "cg");
set("ODE order", 1);
set("ODE nonlinear solver", "fixed-point");
#set("ODE linear solver", "direct");
set("ODE tolerance", 1.0e-2);
set("ODE discrete tolerance", 1.0e-9);

set("ODE fixed time step", True);
set("ODE initial time step", 1.0e-3);
set("ODE maximum time step", 1.0e-3);

set("ODE save solution", False);
set("ODE solution file name", "primal.py");
set("ODE number of samples", 100);

pde = NSPDE(mesh, f, nu, d1, d2, vbc, pbc, T)

pde.solve(pde.U)

