# Load mayavi
from mayavi import *

# Plot 2D mesh
v0 = mayavi()
v1 = mayavi()
d0 = v0.open_vtk_xml("mesh2D000003.vtu")
d1 = v1.open_vtk_xml("mesh3D000003.vtu")
m0 = v0.load_module("BandedSurfaceMap")
m1 = v1.load_module("BandedSurfaceMap")

# Use wireframe (perhaps there's a nicer way to do this)
m0.rep_var.set(1)
m0.represent_config()
m1.rep_var.set(1)
m1.represent_config()

# Wait until window is closed
v0.master.wait_window()
v1.master.wait_window()
