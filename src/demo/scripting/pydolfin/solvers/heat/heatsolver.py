from math import *
from dolfin import *
from heatpde import *

# Define right-hand side
class Source(Function):
    def eval(self, point, i):
        return 10.0 * point.x * sin(point.y)

# Define boundary condition
class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 1.0:
            value.set(0.0)

f = Source()
bc = SimpleBC()

# Create a mesh of the unit square
mesh = UnitSquare(16, 16)


k = 1e-3
T = 10.0


set("ODE method", "dg");
set("ODE order", 0);
set("ODE nonlinear solver", "newton");
set("ODE tolerance", 1e-1);

#set("ODE fixed time step", True);
set("ODE initial time step", 1e-3);
set("ODE maximum time step", 1e+0);

#set("ODE save solution", 1);
set("ODE solution file name", "primal.py");
set("ODE number of samples", 400);

pde = HeatPDE(mesh, f, bc, k, T)

pde.solve(pde.U)

# Plot with Mayavi

# Load mayavi
from mayavi import *

# Plot solution
#v = mayavi()
#d = v.open_vtk_xml("poisson000000.vtu")
#m = v.load_module("Axes")
#m = v.load_module("BandedSurfaceMap")
#f = v.load_filter('WarpScalar', config=0)

# Wait until window is closed
#v.master.wait_window()
