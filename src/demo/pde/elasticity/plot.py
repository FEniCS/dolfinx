# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("elasticity000000.vtu", 0)
f = v.load_filter("ExtractVectorNorm", 0)
m = v.load_module("BandedSurfaceMap", 0)

# Wait until window is closed
v.master.wait_window()
