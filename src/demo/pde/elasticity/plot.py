# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("elasticity000000.vtu")
m = v.load_module("BandedSurfaceMap")

# Wait until window is closed
v.master.wait_window()
