# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("elasticity000000.vtu")

f = v.load_filter('WarpVector', config=0) 

m = v.load_module("BandedSurfaceMap")
m.actor.GetProperty().SetColor((0.8, 0.8, 1.0))

camera = v.renwin.camera
camera.Zoom(0.7)
camera.SetPosition(8.5, 2.5, 15.5)
camera.SetFocalPoint(4.0, -2.0, 0.0)
camera.SetRoll(0.0)

# Turn on sweeping
d.sweep_delay.set(0.01)
d.sweep_var.set(1)
d.do_sweep()

# Wait until window is closed
v.master.wait_window()
