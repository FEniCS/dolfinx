# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("velocity000000.vtu")
m = v.load_module("VelocityVector")

# Wait until window is closed
v.master.wait_window()
